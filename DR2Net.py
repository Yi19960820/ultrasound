import torch
import torch.nn as nn
from ResNet3dC import Conv3dC
from torchsummary import summary
import numpy as np

# https://github.com/smortezavi/Randomized_SVD_GPU/blob/master/pytorch_randomized_svd.ipynb
def simple_randomized_torch_svd(Mr, Mi, k=10):
    Br = torch.tensor(Mr).cuda(0)
    Bi = torch.tensor(Mr).cuda(0)
    m, n = Br.size()
    transpose = False
    if m < n:
        transpose = True
        Br = Br.transpose(0, 1).cuda(0)
        Bi = Bi.transpose(0, 1).cuda(0)
        m, n = Br.size()
    rand_matrix_R = torch.rand((n,k), dtype=torch.double).cuda(0)  # short side by k
    rand_matrix_I = torch.rand((n,k), dtype=torch.double).cuda(0)
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

class DR2ResBlock(nn.Module):
    '''
    Res block from DR^2-Net: https://arxiv.org/pdf/1702.05743.pdf
    '''
    def __init__(self):
        super(DR2ResBlock, self).__init__()
        # c1 = 64
        # c2 = 32
        c1 = 16
        c2 = 8
        c3 = 1
        # w1 = 11
        # w2 = 1
        # w3 = 7
        w1 = 7
        w2 = 1
        w3 = 3
        p1 = w1//2
        p2 = w2//2
        p3 = w3//2

        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm3d(c1)
        self.bn2 = nn.BatchNorm3d(c2)
        self.conv1 = Conv3dC(1,c1,(w1,w1),(1,1),(p1,p1))
        self.conv2 = Conv3dC(c1,c2,(w2,w2),(1,1),(p2,p2))
        self.conv3 = Conv3dC(c2,c3,(w3,w3),(1,1),(p3,p3))

    def forward(self, xR, xI):
        yR, yI = self.conv1(xR, xI)
        yR, yI = self.relu(yR), self.relu(yI)
        yR, yI = self.bn1(yR), self.bn1(yI)
        yR, yI = self.conv2(yR, yI)
        yR, yI = self.relu(yR), self.relu(yI)
        yR, yI = self.bn2(yR), self.bn2(yI)
        yR, yI = self.conv3(yR, yI)
        yR = yR+xR
        yI = yI+xI

        return yR, yI
    
class DR2Net(nn.Module):
    def __init__(self, gpu=True):
        super(DR2Net, self).__init__()

        self.rb1 = DR2ResBlock()
        self.rb2 = DR2ResBlock()
        self.rb3 = DR2ResBlock()
        self.rb4 = DR2ResBlock()
    
    def svtC(self,xR, xI,th):
        m,n=xR.shape

        # form_out={'pre':'concat','shape':[m,n]}
        # U,S,V=svd(self.converter.torch2np([x],[form_out])[0], full_matrices=False)
        # U,S,V=svd(x.cpu().detach().numpy(), full_matrices=False)
        # # S = np.diag(S)
        # U = torch.from_numpy(U).reshape((m,n)).cuda()
        # S = torch.from_numpy(S).reshape((n,)).cuda()
        # V = torch.from_numpy(V).reshape((n,n)).cuda()

        U,S,V = simple_randomized_torch_svd(xR, xI, k=10)
        
        S=self.relu(S-th*S[0])

        US=torch.zeros(m,n)
        stmp=torch.zeros(n)
        stmp[0:S.shape[0]]=S
        # stmp=S
        minmn=min(m,n)
        US[:,0:minmn]=U[:,0:minmn]

        x=(US*stmp)@V.t()
        return x

    def forward(self, x):
        T2=x.shape[-1]
        T=int(T2/2)
        xR=x[:,:,:,:,0:T]
        xI=x[:,:,:,:,T:T2]

        yR, yI = self.rb1(xR, xI)
        yR, yI = self.rb2(yR, yI)
        # yR, yI = self.rb3(yR, yI)
        # yR, yI = self.rb4(yR, yI)

        y = torch.cat((yR, yI), -1)
        return y

if __name__=='__main__':
    model = DR2Net()
    x = torch.zeros([1,1, 33, 33,40])
    summary(model, x)