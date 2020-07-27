import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
from UGAN import MaxPool3dC, Conv3dC, Deconv3dC

class ResBlock3dC(nn.Module):
    def __init__(self, Cin, Cout):
        super(ResBlock3dC, self).__init__()

        self.conv0 = Conv3dC(Cin, Cin, (3,3), (1,1), (1,1))
        self.conv1 = Conv3dC(Cin, Cout, (3,3), (1,1), (1,1))

class DownBlock3dC(nn.Module):
    '''
    Convolutional (encoding) block for the U-Net.
    '''
    def __init__(self, Cin, Cout, w, p):
        super(DownBlock3dC, self).__init__()

        w1, w2 = w
        s1, s2 = 1,1
        p1, p2 = p

        self.conv0 = Conv3dC(Cin,Cout,(w1,w2),(s1,s2),(p1,p2))
        self.conv1 = Conv3dC(Cout,Cout,(w1,w2),(s1,s2),(p1,p2))
        # self.conv2 = Conv3dC(Cout,Cout,(w1,w2),(s1,s2),(p1,p2))
        self.bn1 = nn.BatchNorm3d(Cout)
        self.bn2 = nn.BatchNorm3d(Cout)
        self.relu = nn.ReLU()
        
    def forward(self, xR, xI):
        yR, yI = self.conv0(xR, xI)
        yR = self.bn1(yR)
        yI = self.bn1(yI)
        yR = self.relu(yR)
        yI = self.relu(yI)
        yR, yI = self.conv1(yR, yI)
        # yR, yI = self.conv2(yR, yI)
        yR = self.bn2(yR)
        yI = self.bn2(yI)
        # yR = self.relu(yR)
        # yI = self.relu(yI)
        return yR, yI

class UpBlock3dC(nn.Module):
    def __init__(self, Cin, Cmid, Cout, w, p, op=(0,0)):
        super(UpBlock3dC, self).__init__()
        w1, w2 = w
        p1, p2 = p
        s1, s2 = 1,1
        op1, op2 = op

        self.conv0 = Conv3dC(Cin,Cmid,(w1,w2),(s1,s2),(p1,p2))
        self.conv1 = Conv3dC(Cmid,Cmid,(w1,w2),(s1,s2),(p1,p2))
        self.up = Deconv3dC(Cmid, Cout, (2,1), (2,1), op=(op1, op2))
        self.bn1 = nn.BatchNorm3d(Cmid)
        self.bn2 = nn.BatchNorm3d(Cmid)
        self.bn3 = nn.BatchNorm3d(Cout)
        self.relu = nn.ReLU()

    def forward(self, xR, xI):
        yR, yI = self.conv0(xR, xI)
        yR = self.bn1(yR)
        yI = self.bn1(yI)
        yR = self.relu(yR)
        yI = self.relu(yI)
        yR, yI = self.conv1(yR, yI)
        yR = self.bn2(yR)
        yI = self.bn2(yI)
        yR = self.relu(yR)
        yI = self.relu(yI)
        yR, yI = self.up(yR, yI)
        yR = self.bn3(yR)
        yI = self.bn3(yI)
        # yR = self.relu(yR)
        # yI = self.relu(yI)
        return yR, yI

class OutputBlock3dC(nn.Module):
    def __init__(self, Cin, Cmid, Cout):
        super(OutputBlock3dC, self).__init__()

        w1, w2 = 3,3
        s1, s2 = 1,1
        p1, p2 = 1,1
        
        self.conv0 = Conv3dC(Cin,Cmid,(w1,w2),(s1,s2),(p1,p2))
        self.conv1 = Conv3dC(Cmid,Cmid,(w1,w2),(s1,s2),(p1,p2))
        self.conv2 = Conv3dC(Cmid,Cout,(w1,w2),(s1,s2),(p1,p2))
        self.tanh = nn.Tanh()

    def forward(self, xR, xI):
        yR, yI = self.conv0(xR, xI)
        yR, yI = self.conv1(yR, yI)
        yR, yI = self.conv2(yR, yI)
        yR = self.tanh(yR)
        yI = self.tanh(yI)
        return yR, yI

class UNet(nn.Module):
    def __init__(self, gpu=True):
        super(UNet, self).__init__()

        c  = [1, 8, 16, 32, 64, 128]
        w1 = [0 ,3, 3,  3,  3, 3]
        w2 = [0 ,3, 3,  3,  3, 3]
        p1 = [0, 1, 1,  1,  1, 1]
        p2 = [0, 1, 1,  1,  1, 1]

        self.enc0 = DownBlock3dC(c[0], c[1], (w1[1],w2[1]), (p1[1],p2[1]))
        self.mp0 = MaxPool3dC((2,1), (2,1))
        self.enc1 = DownBlock3dC(c[1], c[2], (w1[2],w2[2]), (p1[2],p2[2]))
        self.mp1 = MaxPool3dC((2,1), (2,1))
        self.enc2 = DownBlock3dC(c[2], c[3], (w1[3],w2[3]), (p1[3],p2[3]))
        self.mp2 = MaxPool3dC((2,1), (2,1))
        self.enc3 = DownBlock3dC(c[3], c[4], (w1[4],w2[4]), (p1[4],p2[4]))
        self.mp3 = MaxPool3dC((2,1), (2,1))
        self.bottom = UpBlock3dC(c[4], c[5], c[4], (w1[4], w2[4]), (p1[4], p2[4]))
        self.dec4 = UpBlock3dC(c[4]*2, c[4], c[3], (w1[3],w2[3]), (p1[3], p2[3]))
        self.dec5 = UpBlock3dC(c[3]*2, c[3], c[2], (w1[2],w2[2]), (p1[2], p2[2]))
        self.dec6 = UpBlock3dC(c[2]*2, c[2], c[1], (w1[1],w2[1]), (p1[1], p2[1]))
        self.output = OutputBlock3dC(c[1]*2, c[1], c[0])

    def concat(self, up, skip):
        return torch.cat((up, skip), dim=1)

    def forward(self, x):
        T2=x.shape[-1]
        T=int(T2/2)
        xR=x[:,:,:,:,0:T]
        xI=x[:,:,:,:,T:T2]

        yR0, yI0 = self.enc0(xR, xI)
        yR, yI = self.mp0(yR0, yI0)
        yR1, yI1 = self.enc1(yR, yI)
        yR, yI = self.mp1(yR1, yI1)
        yR2, yI2 = self.enc2(yR, yI)
        yR, yI = self.mp2(yR2, yI2)
        yR3, yI3 = self.enc3(yR, yI)
        yR, yI = self.mp3(yR3, yI3)
        yR, yI = self.bottom(yR, yI)
        yR, yI = self.concat(yR, yR3), self.concat(yI, yI3)
        yR, yI = self.dec4(yR, yI)
        yR, yI = self.concat(yR, yR2), self.concat(yI, yI2)
        yR, yI = self.dec5(yR, yI)
        yR, yI = self.concat(yR, yR1), self.concat(yI, yI1)
        yR, yI = self.dec6(yR, yI)
        yR, yI = self.concat(yR, yR0), self.concat(yI, yI0)
        yR, yI = self.output(yR, yI)

        y=torch.cat((yR,yI),-1)
        return y

if __name__=='__main__':
    model = UNet()
    summary(model, torch.zeros([1,1, 48, 48,40]))