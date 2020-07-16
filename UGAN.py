import torch
import torch.nn as nn
import numpy as np
from ResNet3dC import Conv3dC, ResBlock3dC

class ConvBlock3dC(nn.Module):
    '''
    Convolutional (encoding) block for the U-Net.
    '''
    def __init__(self, Cin, Cmid, Cout, w, p):
        super(ConvBlock3dC, self).__init__()

        w1, w2 = w
        s1, s2 = 1,1
        p1, p2 = p

        self.relu = nn.ReLU()
        self.conv0 = Conv3dC(Cin,Cmid,(w1,w1,w2),(2,2,s2),(p1,p1,p2))
        self.conv1 = Conv3dC(Cin,Cmid,(w1,w1,w2),(s1,s1,s2),(p1,p1,p2))
        self.conv2 = Conv3dC(Cmid,Cmid,(w1,w1,w2),(s1,s1,s2),(p1,p1,p2))
        self.conv3 = Conv3dC(Cmid,Cmid,(w1,w1,w2),(s1,s1,s2),(p1,p1,p2))
        
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
    def __init__(self, Cin, Cmid, Cout, w, p):
        w1, w2 = w
        p1, p2 = p
        s1, s2 = 1,1

        self.relu = nn.ReLU()
        self.conv0 = Deconv3dC(Cin,Cmid,(w1,w1,w2),(2,2,s2),(p1,p1,p2))
        self.conv1 = Deconv3dC(Cin,Cmid,(w1,w1,w2),(s1,s1,s2),(p1,p1,p2))
        self.conv2 = Deconv3dC(Cmid,Cmid,(w1,w1,w2),(s1,s1,s2),(p1,p1,p2))
        self.conv3 = Deconv3dC(Cmid,Cmid,(w1,w1,w2),(s1,s1,s2),(p1,p1,p2))
        
        def forward(self, xR, xI, xRc, xIc):
            yR += xRc
            yI += xIc
            yR, yI = self.conv0(yR, yI)
            yRp, yIp = self.conv1(yR, yI)
            yRp, yIp = self.conv2(yRp, yIp)
            yR += yRp
            yI += yIp
            yR = self.conv3(yR, yI)
            return yR, yI

class UGenerator3dC(nn.Module):
    def __init__(self, gpu=True, nchans=8):
        super(UGenerator3dC, self).__init__()

        self.relu = nn.ReLU()
        