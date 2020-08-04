# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 21:48:02 2018
Modified on Fri Jun 12 2020 by Sam Ehrenstein

@author: Yi Zhang
@author: Sam Ehrenstein
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

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
    """========================================================================="""

    #Dataset, converter and player
    data_dir={'invivo':d_invivo,'sim_pm':d_simpm,'sim':d_sim}[prefix]
    conter=Converter()
    formshow={'pre':'concat','shape':(m,n,p)}
    formlist=[]
    for i in range(6):
        formlist.append(formshow)
    minloss=np.inf
    #Logs
    log=open('/results/%s/%s_Res3dC_Log_Tr%s_epoch%s_lr%.2e.txt'\
        %(ProjectName,prefix,TrainInstances,num_epochs,lr_list[0]),'a')

    print('Project Name: %s\n'%ProjectName)
    log.write('Project Name: %s\n\n'%ProjectName)
    #Loading data
    print('Loading phase...')
    print('----------------')
    log.write('Loading phase...\n')
    log.write('----------------\n')
    print(f'Data directory: {d_sim}')
    log.write(f'Data directory: {d_sim}')
    shape_dset=(m,n,p*2)    # The last dimension is 2*the number of frames (for real and imaginary)
    #training
    train_dataset=BigImageDataset(round(TrainInstances),shape_dset,
                            train=0,data_dir=data_dir)
    train_loader=data.DataLoader(train_dataset,batch_size=BatchSize,shuffle=True)
    #validation
    val_dataset=BigImageDataset(round(ValInstances),shape_dset,
                            train=1,data_dir=data_dir, train_size=TrainInstances)
    val_loader=data.DataLoader(val_dataset,batch_size=ValBatchSize,shuffle=True)
    print('Finished loading.\n')
    log.write('Finished loading.\n\n')

    # Training
    for learning_rate in lr_list:
        #Construct network
        print('Configuring network...')
        log.write('Configuring network...\n')
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

        #Loss and optimizer
        floss=nn.MSELoss()
        optimizer=torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=wd)

        #Array for recording data
        outputs_S = to_var(torch.zeros([1,1,m,m,p*2]))
        lossmean_vec=np.zeros((num_epochs,))
        lossmean_val_vec=np.zeros((num_epochs,))
        
        #Training
        print('Training the model over %d samples, with learning rate %.6f and weight decay %.6f\n'\
            %(TrainInstances,learning_rate, wd))
        log.write('Training the model over %d samples, with learning rate %.6f and weight decay %.6f\n'\
            %(TrainInstances,learning_rate, wd))
        print(f'Load model: {loadmodel}')
        log.write(f'Load model: {loadmodel}')
        if loadmodel:
            print(f'Loading from {mfile}')
            log.write(f'Loading from {mfile}')
        print(f'Train batch size: {BatchSize}')
        log.write(f'Train batch size: {BatchSize}')
        # Run over each epoch
        for epoch in range(num_epochs):
            #print time
            ts=time.time()
            st=datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
            print('\n'+st)
            log.write('\n'+st+'\n')
            
            loss_val_mean=0
            loss_mean=0
            # Train
            print('Loading and calculating training batches...')
            log.write('Loading and calculating training batches...\n')
            starttime=time.time()
            pbar = tqdm(range(int(TrainInstances/BatchSize)))
            for _,(_,S,D) in enumerate(train_loader):
                # set the gradients to zero at the beginning of each epoch
                optimizer.zero_grad()  
                batch_loss = 0
                for ii in range(BatchSize):
                    inputs=to_var(D[ii])   # "ii"th picture
                    targets_S=to_var(S[ii])

                    outputs_S=net(inputs[None,None])  # Forward
                    loss=floss(outputs_S.squeeze(), targets_S)  # Current loss
                    batch_loss += loss.item()
                    loss_mean+=loss.item()
                    loss.backward()
                optimizer.step()
                pbar.set_description("Batch loss: %2.9f" % (batch_loss/BatchSize))
                pbar.update()
            pbar.close()
            loss_mean=loss_mean/TrainInstances
            endtime=time.time()
            print('Training time is %f'%(endtime-starttime))
            log.write('Training time is %f\n'%(endtime-starttime))
            
            # Validation 
            print('Loading and calculating validation batches...')
            log.write('Loading and calculating validation batches...\n')
            starttime=time.time()
            with torch.no_grad():
                for _,(_,Sv,Dv) in enumerate(val_loader): 
                    for jj in range(ValBatchSize):
                        inputsv=to_var(Dv[jj])   # "jj"th picture
                        targets_Sv=to_var(Sv[jj])
        
                        outputs_Sv=net(inputsv[None,None])  # Forward
                        loss_val=floss(outputs_Sv.squeeze(),targets_Sv)  # Current loss
                        loss_val_mean+=loss_val.item()
            loss_val_mean=loss_val_mean/ValInstances
            endtime=time.time()
            print('Test time is %f'%(endtime-starttime))
            log.write('Test time is %f\n'%(endtime-starttime))

            torch.save(net.state_dict(), 
                    "/results/%s/%s_Res3dC_Model_Tr%s_epoch%s_lr%.2e.pkl"\
                    %(ProjectName,prefix,TrainInstances,num_epochs,learning_rate))
            minloss=min(loss_val_mean,minloss)

            if loss_val_mean<minloss:
                print('Best saved at [epoch%d/%d]'%(epoch+1,num_epochs))
                log.write('Best saved at [epoch%d/%d]\n'\
                    %(epoch+1,num_epochs))
                torch.save(net.state_dict(), 
                        "/results/%s/%s_Res3dC_Best_Model_Tr%s_epoch%s_lr%.2e.pkl"\
                        %(ProjectName,prefix,TrainInstances,num_epochs,learning_rate))
                minloss=min(loss_val_mean,minloss)
        
            # Print loss
            if (epoch + 1)%1==0:    # % 10
                print('Epoch [%d/%d], Lossmean: %.5e, Validation lossmean: %.5e'\
                    %(epoch+1,num_epochs,loss_mean,loss_val_mean))
                log.write('Epoch [%d/%d], Lossmean: %.5e, Validation lossmean: %.5e\n'\
                    %(epoch+1,num_epochs,loss_mean,loss_val_mean))
            if epoch > 0:
                # Improvement ratio
                print('Epoch [%d/%d], Train imprv. factor: %.5e, Val imprv. factor: %.5e'\
                    %(epoch+1,num_epochs,lossmean_vec[epoch-1]/loss_mean,lossmean_val_vec[epoch-1]/loss_val_mean))
                log.write('Epoch [%d/%d], Train imprv. factor: %.6f, Val imprv. factor: %.6f'\
                    %(epoch+1,num_epochs,loss_mean/lossmean_vec[epoch-1],loss_val_mean/lossmean_val_vec[epoch-1]))

            if loss.item() > 100:
                print('hitbadrut')
                log.write('hitbadrut\n')
                break
            
            lossmean_vec[epoch]=loss_mean
            lossmean_val_vec[epoch]=loss_val_mean

            np.savez('/results/%s/%s_Res3dC_LossData_Tr%s_epoch%s_lr%.2e'\
                %(ProjectName,prefix,TrainInstances,num_epochs,learning_rate),
                lossmean_vec,lossmean_val_vec)
            if epoch >= 2 and stop_early and (lossmean_val_vec[epoch] > 1.5*lossmean_val_vec[epoch-1]):
                print(f'Stopping early at epoch {epoch+1}')
                log.write(f'Stopping early at epoch {epoch+1}')
                break

        """Save logs, prediction, loss figure, loss data, model and settings """
        np.savez('/results/%s/%s_Res3dC_LossData_Tr%s_epoch%s_lr%.2e'\
                %(ProjectName,prefix,TrainInstances,num_epochs,learning_rate),
                lossmean_vec,lossmean_val_vec)
        
        #Save settings of training
        params={'ProjectName':ProjectName,
                'prefix':prefix,
                'mfile':mfile if loadmodel else None,
                'gpu':gpu,
                'data_dir':data_dir,
                'shape':shape_dset,
                'lr_list':lr_list,
                'TrainInstances':TrainInstances,
                'ValInstances':ValInstances,
                'BatchSize':BatchSize,
                'ValBatchSize':ValBatchSize,
                'num_epochs':num_epochs,
                'frame':frame}
        file=open('/results/%s/%s_Res3dC_Settings_Tr%s_epoch%s_lr%.2e.txt'\
                %(ProjectName,prefix,TrainInstances,num_epochs,learning_rate),'w')
        file.write('Training Settings:\n\t')
        for k,v in params.items():
            file.write(k+'='+str(v)+'\n')
            file.write('\t')
        file.close()
        
        #Print min loss
        print('\nmin Loss=%.3e'%minloss)
        log.write('\nmin Loss=%.3e'%minloss)
        log.close()
