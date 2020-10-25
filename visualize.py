import torch
import torch.nn as nn
from ResNet3dC import ResNet3d
import sys, os
from torchvision import utils
import matplotlib.pyplot as plt
import numpy as np

def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1, slice=0): 
    n,c,w,h,d = tensor.shape
    print(tensor.shape)
    # tensor = tensor[:,:,:,:,slice]

    if allkernels: tensor = tensor.view(n*c, w, h, d)
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))    
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    grid = grid.view(grid.shape[0]*grid.shape[1], grid.shape[2])
    plt.figure( figsize=(nrow,rows*d) )
    plt.imshow(grid.numpy(), cmap='gray')

if __name__=='__main__':
    mfile = os.path.abspath(sys.argv[1])
    model=ResNet3d(gpu=False)
    state_dict=torch.load(mfile,map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    filt = model.conv2.weight.detach().clone()
    visTensor(filt, ch=0, allkernels=True)
    plt.axis('off')
    plt.ioff()
    plt.show()