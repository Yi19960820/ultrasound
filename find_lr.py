# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 2020

@author: Sam Ehrenstein
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch_lr_finder import LRFinder

import torch.utils.data as data

import sys
# sys.path.append('../')
import os

from DataSet import BigImageDataset
from CORONA.classes.Dataset import Converter

import numpy as np
import time
import datetime
import pickle
from tqdm import tqdm
import yaml

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

if __name__=='__main__':
    """Settings"""
    """========================================================================="""
    #Name and choice of training set
    prefix='sim' #invivo,sim_pm,sim
    #Load model
    mfile='/results/multi_rank_1_7_sim_Res3dC_Model_Tr6000_epoch30_lr2.00e-03.pkl'

    """Network Settings: Remember to change the parameters when you change model!"""
    gpu=True #if gpu=True, the ResNet will use more parameters

    #seed
    # seed=1237
    # torch.manual_seed(seed)
    #parameters for training
    BatchSize      = 40
    ValBatchSize   = 40
    num_epochs     = 30
    frame=10
    #directory of datasets
    d_invivo='/data/Invivo/' 
    d_simpm='/data/Sim_PM/'

    # Load settings from config file
    cfg_file = sys.argv[1]
    cfg = yaml.safe_load(open(cfg_file))
    ProjectName=cfg['ProjectName']
    d_sim = cfg['datadir']
    loadmodel = cfg['loadmodel']
    if loadmodel=='False':
        loadmodel=False
    lr_list = [cfg['lr']]
    if loadmodel:
        mfile = cfg['mfile']
    TrainInstances = cfg['ntrain']
    ValInstances   = cfg['nval']
    out_dir = f'/results/{ProjectName}'
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    if 'batchsize' in cfg.keys():
        BatchSize = cfg['batchsize']
    if 'epochs' in cfg.keys():
        num_epochs = cfg['epochs']
    m = cfg['m']
    n = cfg['n']
    p = cfg['nframes']
    if cfg['custom']:
        from ResNet3dC import ResNet3dC
    else:
        from CORONA.network.ResNet3dC import ResNet3dC
    
    if 'weight_decay' in cfg.keys():
        wd = cfg['weight_decay']
    else:
        wd = 0

    if 'stop_early' in cfg.keys():
        stop_early = cfg['stop_early']
    else:
        stop_early = False
    
    if 'low_lr' in cfg.keys():
        low_lr = 1e-4
        high_lr = 5e-2
    else:
        quit()  # no bounds provided, so running this is user error
    """========================================================================="""

    #Dataset, converter and player
    data_dir={'invivo':d_invivo,'sim_pm':d_simpm,'sim':d_sim}[prefix]
    conter=Converter()
    formshow={'pre':'concat','shape':(m,n,p)}
    formlist=[]
    for i in range(6):
        formlist.append(formshow)
    minloss=np.inf

    print('Project Name: %s\n'%ProjectName)
    #Loading data
    print('Loading phase...')
    print('----------------')
    print(f'Data directory: {d_sim}')
    shape_dset=(m,n,p*2)    # The last dimension is 2*the number of frames (for real and imaginary)
    #training

    #Construct network
    print('Configuring network...')
    if not loadmodel:
        net=ResNet3dC(gpu)
    else:
        if mfile[-3:]=='pkl':
            net=ResNet3dC(gpu)
            state_dict=torch.load(mfile, map_location='cuda:0')
            net.load_state_dict(state_dict)
        else:
            net=torch.load(mfile)

    if torch.cuda.is_available():
        net=net.cuda()
    print('Configured.')

    train_dataset=BigImageDataset(round(TrainInstances),shape_dset,
                            train=0,data_dir=data_dir)
    train_loader=data.DataLoader(train_dataset,batch_size=BatchSize,shuffle=True)
    floss=nn.MSELoss()
    optimizer=torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=wd)
    print('Finished loading.\n')
    lr_finder = LRFinder(net, optimizer, floss, device='cuda:0')
    lr_finder.range_test(train_loader, start_lr=low_lr, end_lr=high_lr, num_iter=5, diverge_th=100)
    np.savez_compressed("/results/%s/lrfinderiter_%d_min_%.2e_max_%.2e.npz"\
                    %(ProjectName, 5, low_lr, high_lr), **lr_finder.history)