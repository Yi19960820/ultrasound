import torch
import torch.nn as nn
from ResNet3dC import Conv3dC
from torchsummary import summary

class DR2ResBlock(nn.Module):
    '''
    Res block from DR^2-Net: https://arxiv.org/pdf/1702.05743.pdf
    '''
    def __init__(self):
        super(DR2ResBlock, self).__init__()
        c1 = 64
        c2 = 32
        c3 = 1
        w1 = 11
        w2 = 1
        w3 = 7
        p1 = 5
        p2 = 0
        p3 = 3

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
    
    def forward(self, x):
        T2=x.shape[-1]
        T=int(T2/2)
        xR=x[:,:,:,:,0:T]
        xI=x[:,:,:,:,T:T2]

        yR, yI = self.rb1(xR, xI)
        yR, yI = self.rb2(yR, yI)
        yR, yI = self.rb3(yR, yI)
        yR, yI = self.rb4(yR, yI)

        y = torch.cat((yR, yI), -1)
        return y

if __name__=='__main__':
    model = DR2Net()
    x = torch.zeros([1,1, 33, 33,40])
    summary(model, x)