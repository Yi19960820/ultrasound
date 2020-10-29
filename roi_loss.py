import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def ROILoss(x, y, m):
    return 5*torch.sum(((y-x)*m)**2) + torch.sum(((y-x)*(1-m))**2)

if __name__=='__main__':
    signal = torch.cat((torch.ones(1,5), torch.zeros(2,5)), 0)
    signal.requires_grad=True
    noise = torch.ones(3,5)*0.5
    mask = torch.cat((torch.ones(1,5), torch.zeros(2,5)), 0)
    print(signal)
    print(signal+noise)
    l2loss = nn.MSELoss()
    loss = ROILoss(signal, signal+noise, mask)
    loss.backward()
    print(loss)
    l2 = l2loss(signal, signal+noise)
    print(l2)
