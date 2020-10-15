import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ROILoss(nn.Module):
    '''
    A region-of-interest loss function. Gives 10x more empahsis to the MSE
    in the ROI.
    '''
    def __init__(self):
        super().__init__()
        self.factor = 10
    
    def forward(self, x, y, m):
        return self.factor*torch.sum(((y-x)*m)**2) + torch.sum(((y-x)*(1-m))**2)

if __name__=='__main__':
    from DR2Net import DR2Net
    model = DR2Net()
    x = torch.zeros([1,1, 33, 33,40])