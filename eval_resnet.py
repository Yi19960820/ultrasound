# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 15:10:02 2018

@author: Yi Zhang
"""

import numpy as np
import sys
import torch
import matplotlib.pyplot as plt
from scipy.io import savemat
sys.path.append('../')
from CORONA.classes.Player import Player
from CORONA.classes.Dataset import Converter
from CORONA.network.ResNet3dC import ResNet3dC
from DataSet import BigImageDataset
import torch.utils.data as data
from main_resnet import to_var
#from tools.mat2gif import mat2gif

"""Settings"""
"""========================================================================="""
#Model file
# 10 epochs on sim
mfile = '/results/Res3dC_nocon_sim_Res3dC_Model_Tr1200_epoch10_lr2.00e-03.pkl'

"""Network Settings: Remember to change the parameters when you change model!"""
gpu=False #if gpu=True, the ResNet will use more parameters
#Directory of input data and its size
data_dir='/data/toy/'
m,n,time=39,39,101 #size of data
#Save gif
saveGif=True
save_gif_dir='/results/gifs'
cmap='hot'
note='abs'
#Save matrix
saveMat=True
save_mat_dir='/results/mats'
"""========================================================================="""

#Converter
form_in={'pre':'concat','shape':[-1,1,m,n,time*2]}
form_out={'pre':'concat','shape':[m,n,time]}
convert=Converter()

#Load the model
#device='cuda:0' if torch.cuda.is_available() else 'cpu'
device='cpu'
if mfile[-3:]=='pkl':
    model=ResNet3dC(gpu)
    state_dict=torch.load(mfile,map_location=device)
    model.load_state_dict(state_dict)
else:
    model=torch.load(mfile)

model.eval()
floss = torch.nn.MSELoss()

#Processing
with torch.no_grad():
    loss_mean = 0
    test_data = BigImageDataset(1000, (m,n,time*2), 2, data_dir=data_dir)
    test_loader = data.DataLoader(test_data, batch_size=4, num_workers=2, shuffle=False)
    for _,(_,S,D) in enumerate(test_loader):
        for jj in range(4):
            inputs = to_var(D[jj])
            targets = to_var(S[jj])

            out_S = model(inputs[None, None])
            loss = floss(out_S.squeeze(), targets).item()
            loss_mean += loss
            [predmv,Sum]=convert.torch2np([out_S, D[jj]],[form_out, form_out])
            #Save gif
            # if saveGif:
            #     mat2gif([Sum,predmv,Bubbles],save_gif_dir,
            #             note=note,cmap=cmap,tit=['Input','Prediction','Ground Truth'])
            #Save matrix
            if saveMat:
                savemat(save_mat_dir,{'D':D[jj],'S':S[jj],'Sp':predmv})

loss_mean /= 1000
print(f'Mean loss: {loss_mean}')