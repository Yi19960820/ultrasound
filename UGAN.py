import torch
import torch.nn as nn
import numpy as np
from ResNet3dC import Conv3dC, ResBlock3dC

class ConvBlock3dC(nn.Module):
    '''
    Convolutional (encoding) block for the U-Net.
    '''
    def __init__(self, Cin, Cout, w, p):
        super(ConvBlock3dC, self).__init__()

        w1, w2 = w
        s1, s2 = 1,1
        p1, p2 = p

        self.relu = nn.ReLU()
        self.conv0 = Conv3dC(Cin,Cout,(w1,w1,w2),(2*s1,2*s1,s2),(p1,p1,p2))
        self.conv1 = Conv3dC(Cout,Cout,(w1,w1,w2),(s1,s1,s2),(p1,p1,p2))
        self.conv2 = Conv3dC(Cout,Cout,(w1,w1,w2),(s1,s1,s2),(p1,p1,p2))
        self.conv3 = Conv3dC(Cout,Cout,(w1,w1,w2),(s1,s1,s2),(p1,p1,p2))
        
        def forward(self, xR, xI):
            yR, yI = self.conv0(xR, xI)
            yRp, yIp = self.conv1(yR, yI)
            yRp, yIp = self.conv2(yRp, yIp)
            yR += yRp
            yI += yIp
            yR = self.conv3(yR, yI)
            return yR, yI

class Deconv3dC(nn.Module):
    """
    Conv block for 3d complex computation
    """
    def __init__(self,Cin,Cout,kernel,stride,padding):
        """
        Args:
            Cin: number of input panes
            Cout: number of output panes
            kernel: (w1,w2), w1 for X and Y dimension, w2 for T dimension
            stride: (s1,s2), s1 for X and Y dimension, s2 for T dimension
            padding: (p1,p2), p1 for X and Y dimension, p2 for T dimension
        """
        super(Deconv3dC,self).__init__()
        
        w1,w2=kernel
        s1,s2=stride
        p1,p2=padding
        self.convR=nn.ConvTranspose3d(Cin,Cout,(w1,w1,w2),(s1,s1,s2),(p1,p1,p2))
        self.convI=nn.ConvTranspose3d(Cin,Cout,(w1,w1,w2),(s1,s1,s2),(p1,p1,p2))
    
    def forward(self,xR,xI):
        xR,xI=self.convR(xR)-self.convI(xI),self.convR(xI)+self.convI(xR)

        return xR,xI

class DeconvBlock3dC(nn.Module):
    def __init__(self, Cin, Cout, w, p):
        w1, w2 = w
        p1, p2 = p
        s1, s2 = 1,1

        self.relu = nn.ReLU()
        self.conv0 = Deconv3dC(Cin,Cin,(w1,w1,w2),(s1,s1,s2),(p1,p1,p2))
        self.conv1 = Deconv3dC(Cin,Cin,(w1,w1,w2),(s1,s1,s2),(p1,p1,p2))
        self.conv2 = Deconv3dC(Cin,Cin,(w1,w1,w2),(s1,s1,s2),(p1,p1,p2))
        self.conv3 = Deconv3dC(Cin,Cout,(w1,w1,w2),(2*s1,2*s1,s2),(p1,p1,p2))
        
        def forward(self, xR, xI):
            yR, yI = self.conv0(xR, xI)
            yRp, yIp = self.conv1(yR, yI)
            yRp, yIp = self.conv2(yRp, yIp)
            yR += yRp
            yI += yIp
            yR = self.conv3(yR, yI)
            return yR, yI

class UGenerator3dC(nn.Module):
    def __init__(self, gpu=True):
        super(UGenerator3dC, self).__init__()
        
        c  = [1, 4, 8, 16, 32]
        w1 = [0 ,5, 3,  3,  3]
        w2 = [0 ,5, 3,  3,  3]
        p1 = [0, 2, 1,  1,  1]
        p2 = [0, 2, 1,  1,  1]
        
        self.relu = nn.ReLU()
        self.enc0 = ConvBlock3dC(c[0], c[1], (w1[1],w2[1]), (p1[1],p2[1]))
        self.enc1 = ConvBlock3dC(c[1], c[2], (w1[2],w2[2]), (p1[2],p2[2]))
        self.enc2 = ConvBlock3dC(c[2], c[3], (w1[3],w2[3]), (p1[3],p2[3]))
        self.enc3 = ConvBlock3dC(c[3], c[4], (w1[4],w2[4]), (p1[4],p2[4]))
        self.dec4 = DeconvBlock3dC(c[4], c[3], (w1[3],w2[3]), (p1[3], p2[3]))
        self.dec5 = DeconvBlock3dC(c[3], c[2], (w1[2],w2[2]), (p1[2], p2[2]))
        self.dec6 = DeconvBlock3dC(c[2], c[1], (w1[1],w2[1]), (p1[1], p2[1]))
        self.dec7 = DeconvBlock3dC(c[1], c[0], (w1[1],w2[1]), (p1[1], p2[1]))
    
    def forward(self, x):
        T2=x.shape[-1]
        T=int(T2/2)
        xRi=x[:,:,:,:,0:T]
        xIi=x[:,:,:,:,T:T2]

        xR0, xI0 = self.enc0(xRi, xIi)
        xR1, xI1 = self.enc1(xR0, xI0)
        xR2, xI2 = self.enc0(xR1, xI1)
        xR3, xI3 = self.enc1(xR2, xI2)
        xR, xI = self.dec4(xR3, xI3)
        xR, xI = self.dec5(xR+xR2, xI+xI2)
        xR, xI = self.dec6(xR+xR1, xI+xI1)
        xR, xI = self.dec7(xR+xR0, xI+xI0)

        x=torch.cat((xR,xI),-1)
        return x
