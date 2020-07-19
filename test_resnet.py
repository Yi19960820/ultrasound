# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 15:10:02 2018

@author: Yi Zhang
"""

import numpy as np
import sys
import time
import torch
import matplotlib.pyplot as plt
from scipy.io import savemat
sys.path.append('../')
from CORONA.classes.Player import Player
from CORONA.classes.Dataset import Converter
from CORONA.network.ResNet3dC import ResNet3dC
from DataSet import BigImageDataset
import torch.utils.data as data
import os
from main_resnet import to_var
from plot_mat import psnr, svt
from tqdm import tqdm
import yaml
#from tools.mat2gif import mat2gif

"""Settings"""
"""========================================================================="""
#Model file
# 10 epochs on sim

"""Network Settings: Remember to change the parameters when you change model!"""
gpu=True #if gpu=True, the ResNet will use more parameters
#Directory of input data and its size
m,n,p=39,39,20 #size of data
#Save gif
saveGif=True
save_gif_dir='/results/gifs'
cmap='hot'
note='abs'
#Save matrix
saveMetadata=True
save_mat_dir='/results/mats-invivo'

cfg = yaml.load(open('/data/resnet.yaml'))
data_dir = cfg['datadir']
TestInstances = cfg['ntest']
saveMat = cfg['saveMat']
mfile = cfg['mfile']
"""========================================================================="""

#Converter
form_in={'pre':'concat','shape':[-1,1,m,n,p*2]}
form_out={'pre':'concat','shape':[m,n,p]}
convert=Converter()

# with open('eval_resnet.yml') as f:
#     config = yaml.load(f)
#     mfile = config['mfile']
#     gpu = config['gpu']
#     data_dir = config['data_dir']

#Load the model
device='cuda:0' if torch.cuda.is_available() else 'cpu'
# device='cpu'
if mfile[-3:]=='pkl':
    model=ResNet3dC(gpu)
    state_dict=torch.load(mfile,map_location=device)
    model.load_state_dict(state_dict)
else:
    model=torch.load(mfile)

model = model.cuda()
model.eval()
floss = torch.nn.MSELoss()

#Processing
with torch.no_grad():
    loss_mean = 0
    test_data = BigImageDataset(TestInstances, (m,n,p*2), 0, data_dir=data_dir, gt=False)
    test_loader = data.DataLoader(test_data, batch_size=4, shuffle=False)
    nx = 0
    fnames = os.listdir(data_dir)
    fnames.sort()

    net_start = time.time()
    net_time = 0
    for i,(_,_,D) in tqdm(enumerate(test_loader)):
        for jj in range(len(D)):
            inputs = to_var(D[jj])

            net_start = time.time()
            out_S = model(inputs[None, None])
            net_time += (time.time()-net_start)
            [Sp, Dg]=convert.torch2np([out_S, D[jj]],[form_out, form_out])

            savemat(os.path.join(save_mat_dir, f'{nx}.mat'),{'D':Dg, 'Sp':Sp})

            nx += 1
    net_time /= TestInstances

print(f'ResNet average time per instance: {net_time}')