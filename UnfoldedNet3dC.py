# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 11:12:43 2018

@author: Yi Zhang
@author: Sam Ehrenstein

Modified by Sam (Bayat group, CWRU) to only calculate L from the assumption that S=D-L.
"""

import torch
import torch.nn as nn
from scipy.linalg import svd
import numpy as np
from torch.autograd import Variable
from CORONA.classes.Dataset import Converter

def to_var(X,CalInGPU):
    if CalInGPU and torch.cuda.is_available():
        X = X.cuda()
    return Variable(X)

# https://github.com/smortezavi/Randomized_SVD_GPU/blob/master/pytorch_randomized_svd.ipynb
def simple_randomized_torch_svd(M, k=10):
    B = torch.tensor(M).cuda(0)
    m, n = B.size()
    transpose = False
    if m < n:
        transpose = True
        B = B.transpose(0, 1).cuda(0)
        m, n = B.size()
    rand_matrix = torch.rand((n,k), dtype=torch.double).cuda(0)  # short side by k
    Q, _ = torch.qr(B @ rand_matrix)                              # long side by k
    Q.cuda(0)
    smaller_matrix = (Q.transpose(0, 1) @ B).cuda(0)             # k by short side
    U_hat, s, V = torch.svd(smaller_matrix,False)
    U_hat.cuda(0)
    U = (Q @ U_hat)
    
    if transpose:
        return V.transpose(0, 1), s, U.transpose(0, 1)
    else:
        return U, s, V
    
class Conv3dC(nn.Module):
    def __init__(self,kernel):
        super(Conv3dC,self).__init__()
        
        pad0=int((kernel[0]-1)/2)
        pad1=int((kernel[1]-1)/2)
        self.convR=nn.Conv3d(1,1,(kernel[0],kernel[0],kernel[1]),(1,1,1),(pad0,pad0,pad1))
        self.convI=nn.Conv3d(1,1,(kernel[0],kernel[0],kernel[1]),(1,1,1),(pad0,pad0,pad1))
        
    def forward(self,x):
        n=x.shape[-1]
        nh=int(n/2)
        xR,xI=x[None,None,:,:,0:nh],x[None,None,:,:,nh:n]

        xR,xI=self.convR(xR)-self.convI(xI),self.convR(xI)+self.convI(xR)

        xR,xI=xR.squeeze(),xI.squeeze()
        x=torch.cat((xR,xI),-1)
        
        return x

class ISTACell(nn.Module):
    def __init__(self,kernel,exp_L,exp_S,coef_L,coef_S,CalInGPU):
        super(ISTACell,self).__init__()
        
        self.converter = Converter()

        self.conv1=Conv3dC(kernel)
        self.conv2=Conv3dC(kernel)
        self.conv3=Conv3dC(kernel)
        self.conv4=Conv3dC(kernel)
        self.conv5=Conv3dC(kernel)
        self.conv6=Conv3dC(kernel)
        
        self.exp_L=nn.Parameter(exp_L)
        # self.exp_S=nn.Parameter(exp_S)
        
        self.coef_L=coef_L
        self.coef_S=coef_S
        self.CalInGPU=CalInGPU
        self.relu=nn.ReLU()
        self.sig=nn.Sigmoid()
        
    def forward(self,data):
        x=data[0]
        L=data[1]
        # S=data[2]
        S = x-L
        H,W,T2=x.shape
        
        Ltmp=self.conv1(x)+self.conv2(L)+self.conv3(S)         
        # Stmp=self.conv4(x)+self.conv5(L)+self.conv6(S)
        # Stmp = x-Ltmp

        thL=self.sig(self.exp_L)*self.coef_L
        # thS=self.sig(self.exp_S)*self.coef_S
        
        L=self.svtC(Ltmp.view(H*W,T2),thL)
        # S=self.mixthre(Stmp.view(H*W,T2),thS)
        
        data[1]=L.view(H,W,T2)
        # data[2]=S.view(H,W,T2)
        
        return data
            
    def svtC(self,x,th):
        m,n=x.shape

        # Perform the SVD using the Scipy implementation
        # This gets around an issue with Intel MKL

        # form_out={'pre':'concat','shape':[m,n]}
        # U,S,V=svd(self.converter.torch2np([x],[form_out])[0], full_matrices=False)
        # U,S,V=svd(x.cpu().detach().numpy(), full_matrices=False)
        # # S = np.diag(S)
        # U = torch.from_numpy(U).reshape((m,n)).cuda()
        # S = torch.from_numpy(S).reshape((n,)).cuda()
        # V = torch.from_numpy(V).reshape((n,n)).cuda()

        U,S,V = simple_randomized_torch_svd(x, k=10)
        
        S=self.relu(S-th*S[0])

        US=to_var(torch.zeros(m,n),self.CalInGPU)
        stmp=to_var(torch.zeros(n),self.CalInGPU)
        stmp[0:S.shape[0]]=S
        # stmp=S
        minmn=min(m,n)
        US[:,0:minmn]=U[:,0:minmn]

        x=(US*stmp)@V.t()
        return x
    
    def mixthre(self,x,th):
        n=x.shape[-1]
        nh=int(n/2)
        xR,xI=x[:,0:nh],x[:,nh:n]
        normx=xR**2+xI**2
        normx=torch.cat((normx,normx),-1)
        
        x = self.relu((1-th*torch.mean(normx)/normx))*x
                
        return x
        
class UnfoldedNet3dC(nn.Module):
    def __init__(self,params=None):
        super(UnfoldedNet3dC, self).__init__()
        
        self.layers=params['layers']
        self.kernel=params['kernel']
        self.CalInGPU=params['CalInGPU']
        self.coef_L=to_var(torch.tensor(params['coef_L'],dtype=torch.float),
                           self.CalInGPU)
        self.coef_S=to_var(torch.tensor(params['coef_S'],dtype=torch.float),
                           self.CalInGPU)
        self.exp_L=to_var(torch.zeros(self.layers,requires_grad=True),
                          self.CalInGPU)
        self.exp_S=to_var(torch.zeros(self.layers,requires_grad=True),
                          self.CalInGPU)
        self.sig=nn.Sigmoid()

        self.relu=nn.ReLU()
        
        self.filter=self.makelayers()
        
    def makelayers(self):
        filt=[]
        for i in range(self.layers):
            filt.append(ISTACell(self.kernel[i],self.exp_L[i],self.exp_S[i],
                             self.coef_L,self.coef_S,self.CalInGPU))
            
        return nn.Sequential(*filt)
    
    def forward(self,x):
        data=to_var(torch.zeros([2]+list(x.shape)),self.CalInGPU)
        data[0]=x
        
        data=self.filter(data)
        L=data[1]
        # S=data[2]
        
        return L
    
    def getexp_LS(self):
        exp_L,exp_S=self.sig(self.exp_L)*self.coef_L,self.sig(self.exp_S)*self.coef_S
        if torch.cuda.is_available():
            exp_L=exp_L.cpu().detach().numpy()
            exp_S=exp_S.cpu().detach().numpy()
        else:
            exp_L=exp_L.detach().numpy()
            exp_S=exp_S.detach().numpy()
            
        return exp_L,exp_S
    
if __name__=='__main__':
    tmp=torch.tensor(0)

    params_net={'layers':10,
            'kernel':[(5,1)]*3+[(3,1)]*7,
            'coef_L':0.4,
            'coef_S':1.8,
            'CalInGPU':True}
    #print(ISTACell((3,3),tmp,tmp,tmp,tmp,tmp))
    net=(UnfoldedNet3dC(params_net))
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name)
