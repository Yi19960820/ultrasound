import numpy as np
import torch
import torch.nn as nn
from UGAN import ConvBlock3dC, DeconvBlock3dC, MaxPool3dC

class UNet(nn.Module):
    def __init__(self, gpu=True):
        super(UNet, self).__init__()

        c  = [1, 8, 16, 32, 64]
        w1 = [0 ,5, 3,  3,  3]
        w2 = [0 ,5, 3,  3,  3]
        p1 = [0, 2, 1,  1,  1]
        p2 = [0, 2, 1,  1,  1]

        self.enc0 = ConvBlock3dC(c[0], c[1], (w1[1],w2[1]), (p1[1],p2[1]))
        self.enc1 = ConvBlock3dC(c[1], c[2], (w1[2],w2[2]), (p1[2],p2[2]))
        self.enc2 = ConvBlock3dC(c[2], c[3], (w1[3],w2[3]), (p1[3],p2[3]))
        self.enc3 = ConvBlock3dC(c[3], c[4], (w1[4],w2[4]), (p1[4],p2[4]))
        self.dec4 = DeconvBlock3dC(c[4], c[3], (w1[3],w2[3]), (p1[3], p2[3]))
        self.dec5 = DeconvBlock3dC(c[3], c[2], (w1[2],w2[2]), (p1[2], p2[2]), op=(1,0))
        self.dec6 = DeconvBlock3dC(c[2], c[1], (w1[1],w2[1]), (p1[1], p2[1]), op=(1,0))
        self.dec7 = DeconvBlock3dC(c[1], c[0], (w1[1],w2[1]), (p1[1], p2[1]), op=(1,0))

    def forward(self, x):
        T2=x.shape[-1]
        T=int(T2/2)
        xRi=x[:,:,:,:,0:T]
        xIi=x[:,:,:,:,T:T2]

        xR0, xI0 = self.enc0(xRi, xIi)
        # print('xR0 shape',xR0.shape)
        xR1, xI1 = self.enc1(xR0, xI0)
        # print('xR1 shape',xR1.shape)
        xR2, xI2 = self.enc2(xR1, xI1)
        # print('xR2 shape',xR2.shape)
        xR3, xI3 = self.enc3(xR2, xI2)
        # print('xR3 shape',xR3.shape)
        xR, xI = self.dec4(xR3, xI3)
        # print('xR4 shape',xR.shape)
        xR, xI = self.dec5(xR+xR2, xI+xI2)
        # print('xR5 shape',xR.shape)
        xR, xI = self.dec6(xR+xR1, xI+xI1)
        # print('xR6 shape',xR.shape)
        xR, xI = self.dec7(xR+xR0, xI+xI0)
        # print('xR7 shape',xR.shape)

        x=torch.cat((xR,xI),-1)
        return x