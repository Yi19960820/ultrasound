# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 21:48:02 2018
Modified on Fri Jun 12 2020 by Sam Ehrenstein

@author: Yi Zhang
@author: Sam Ehrenstein
"""

from roi_loss import ROILoss
import torch
import torch.nn as nn
from torch.autograd import Variable

import torch.utils.data as data

import sys
# sys.path.append('../')
import os

from DataSet import BigImageDataset
from CORONA.classes.Dataset import Converter
from DR2Net import RealDR2Net, DR2Net
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
    mfile='/results/multi_rank_1_7_sim_DR2_Model_Tr6000_epoch30_lr2.00e-03.pkl'

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

    real=False
    if 'real' in cfg.keys():
        real=cfg['real']
    mixed_loss = False
    if 'mixed_loss' in cfg.keys():
        mixed_loss = cfg['mixed_loss']
    chk = cfg['checkpoint_every']

    # if cfg['custom']:
    #     from ResNet3dC import ResNet3dC
    # else:
    #     from CORONA.network.ResNet3dC import ResNet3dC
    
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
    if real:
        formshow={'pre':None,'shape':(m,n,p)}
    else:
        formshow={'pre':'concat','shape':(m,n,p)}
    formlist=[]
    for i in range(6):
        formlist.append(formshow)
    minloss=np.inf
    #Logs
    log=open('/results/%s/%s_DR2_Log_Tr%s_epoch%s_lr%.2e.txt'\
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
    if real:
        nf=p
    else:
        nf=p*2
    shape_dset=(m,n,nf)    # The last dimension is 2*the number of frames (for real and imaginary)
    #training

    #Construct network
    print('Configuring network...')
    log.write('Configuring network...\n')
    if not loadmodel:
        if real:
            net=RealDR2Net(gpu)
        else:
            net=DR2Net(gpu)
    else:
        if mfile[-3:]=='pkl':
            if real:
                net=RealDR2Net(gpu)
            else:
                net=DR2Net(gpu)
            state_dict=torch.load(mfile, map_location='cuda:0')
            net.load_state_dict(state_dict)
        else:
            net=torch.load(mfile)

    if torch.cuda.is_available():
        net=net.cuda()
    print('Configured.')
    log.write('Configured.\n')

    train_dataset=BigImageDataset(round(TrainInstances),shape_dset,
                            train=0,data_dir=data_dir, real=real, mask=mixed_loss)
    train_loader=data.DataLoader(train_dataset,batch_size=BatchSize,shuffle=True)
    #validation
    val_dataset=BigImageDataset(round(ValInstances),shape_dset,
                            train=1,data_dir=data_dir, train_size=TrainInstances,real=real)
    val_loader=data.DataLoader(val_dataset,batch_size=ValBatchSize,shuffle=True)
    print('Finished loading.\n')
    log.write('Finished loading.\n\n')

    # Training
    for learning_rate in lr_list:

        #Loss and optimizer
        floss = ROILoss
        optimizer=torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=wd)

        #Array for recording data
        outputs_S = to_var(torch.zeros([1,1,m,m,nf]))
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
            for _,(_,S,D,mask) in enumerate(train_loader):
                # set the gradients to zero at the beginning of each epoch
                optimizer.zero_grad()  
                batch_loss = 0
                for ii in range(BatchSize):
                    inputs=to_var(D[ii])   # "ii"th picture
                    targets_S=to_var(S[ii])

                    outputs_S=net(inputs[None,None])  # Forward
                    loss=floss(outputs_S.squeeze(), targets_S, to_var(mask[ii]))  # Current loss
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
                for _,(_,Sv,Dv, Mv) in enumerate(val_loader): 
                    for jj in range(ValBatchSize):
                        inputsv=to_var(Dv[jj])   # "jj"th picture
                        targets_Sv=to_var(Sv[jj])
        
                        outputs_Sv=net(inputsv[None,None])  # Forward
                        loss_val=floss(outputs_Sv.squeeze(),targets_Sv, to_var(Mv[jj]))  # Current loss
                        loss_val_mean+=loss_val.item()
            loss_val_mean=loss_val_mean/ValInstances
            endtime=time.time()
            print('Test time is %f'%(endtime-starttime))
            log.write('Test time is %f\n'%(endtime-starttime))
 
            # Save checkpoint every chk epochs
            if (epoch % chk) ==0:
                torch.save(net.state_dict(), 
                    "/results/%s/%s_DR2_Model_Tr%s_epoch%s_%s_lr%.2e.pkl"\
                    %(ProjectName,prefix,TrainInstances,epoch,num_epochs,learning_rate))

            if loss_val_mean<minloss:
                print('Best saved at [epoch%d/%d]'%(epoch+1,num_epochs))
                log.write('Best saved at [epoch%d/%d]\n'\
                    %(epoch+1,num_epochs))
                torch.save(net.state_dict(), 
                        "/results/%s/%s_DR2_Best_Model_Tr%s_epoch%s_lr%.2e.pkl"\
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
                print('Epoch [%d/%d], Train imprv. factor: %.6f, Val imprv. factor: %.6f'\
                    %(epoch+1,num_epochs,lossmean_vec[epoch-1]/loss_mean,lossmean_val_vec[epoch-1]/loss_val_mean))
                log.write('Epoch [%d/%d], Train imprv. factor: %.6f, Val imprv. factor: %.6f\n'\
                    %(epoch+1,num_epochs,lossmean_vec[epoch-1]/loss_mean,lossmean_val_vec[epoch-1]/loss_val_mean))

            if loss.item() > 1e5:
                print('hitbadrut')
                log.write('hitbadrut\n')
                break
            
            lossmean_vec[epoch]=loss_mean
            lossmean_val_vec[epoch]=loss_val_mean

            np.savez('/results/%s/%s_DR2_LossData_Tr%s_epoch%s_lr%.2e'\
                %(ProjectName,prefix,TrainInstances,num_epochs,learning_rate),
                lossmean_vec,lossmean_val_vec)
            if epoch >= 2 and stop_early and (lossmean_val_vec[epoch] > 1.001*lossmean_val_vec[epoch-1]):
                print(f'Stopping early at epoch {epoch+1}')
                log.write(f'Stopping early at epoch {epoch+1}')
                break

        """Save logs, prediction, loss figure, loss data, model and settings """
        np.savez('/results/%s/%s_DR2_LossData_Tr%s_epoch%s_lr%.2e'\
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
        file=open('/results/%s/%s_DR2_Settings_Tr%s_epoch%s_lr%.2e.txt'\
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
