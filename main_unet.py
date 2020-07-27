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
from UNet import UNet
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
    
    """Network Settings: Remember to change the parameters when you change model!"""
    gpu=True #if gpu=True, the ResNet will use more parameters

    # #seed
    # seed=1237
    # torch.manual_seed(seed)
    #parameters for training
    BatchSize      = 40
    ValBatchSize   = 40
    num_epochs     = 30
    m, n, p = (48,48,26)
    frame=10
    #directory of datasets
    d_invivo='/data/Invivo/' 
    d_simpm='/data/Sim_PM/'

    # Load settings from config file
    cfg = yaml.load(open('/data/resnet.yaml'))
    d_sim = cfg['datadir']
    loadmodel = cfg['loadmodel']
    if loadmodel=='False':
        loadmodel=False
    lr_list = [cfg['lr']]
    if loadmodel:
        mfile = cfg['mfile']
    print(loadmodel)
    TrainInstances = cfg['ntrain'] # Size of training dataset
    ValInstances   = cfg['nval']
    ProjectName = cfg['ProjectName']
    out_dir = f'/results/{ProjectName}'
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    if 'weight_decay' in cfg.keys():
        wd = cfg['weight_decay']
    else:
        wd = 0
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
    log=open('%s/%s_UNet_Log_Tr%s_epoch%s_lr%.2e.txt'\
        %(out_dir,prefix,TrainInstances,num_epochs,lr_list[0]),'a')

    print('Project Name: %s\n'%ProjectName)
    log.write('Project Name: %s\n\n'%ProjectName)
    #Loading data
    print('Loading phase...')
    print('----------------')
    log.write('Loading phase...\n')
    log.write('----------------\n')
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
            net=UNet(gpu)
        else:
            if mfile[-3:]=='pkl':
                net=UNet(gpu)
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
        print('Training the model over %d samples, with learning rate %.6f\n'\
            %(TrainInstances,learning_rate))
        log.write('Training the model over %d samples, with learning rate %.6f\n\n'\
            %(TrainInstances,learning_rate))
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
            for i,(_,S,D) in enumerate(train_loader):
                # set the gradients to zero at the beginning of each epoch
                optimizer.zero_grad()  
                for ii in range(BatchSize):
                    inputs=to_var(D[ii])   # "ii"th picture
                    targets_S=to_var(S[ii])

                    outputs_S=net(inputs[None,None])  # Forward
                    loss=floss(outputs_S.squeeze(), targets_S)  # Current loss
                    batch_loss = loss.item()
                    loss_mean+=batch_loss
                    loss.backward()
                optimizer.step()
                pbar.set_description("Batch loss: %2.9f" % batch_loss)
                pbar.update()
            pbar.close()
            loss_mean=loss_mean/len(train_loader)
            endtime=time.time()
            print('Training time is %f'%(endtime-starttime))
            log.write('Training time is %f\n'%(endtime-starttime))
            
            # Validation 
            print('Loading and calculating validation batches...')
            log.write('Loading and calculating validation batches...\n')
            starttime=time.time()
            with torch.no_grad():
                for _,(_,Sv,Dv) in enumerate(val_loader): 
                    for jj in range(BatchSize):
                        inputsv=to_var(Dv[jj])   # "jj"th picture
                        targets_Sv=to_var(Sv[jj])
        
                        outputs_Sv=net(inputsv[None,None])  # Forward
                        loss_val=floss(outputs_Sv.squeeze(),targets_Sv)  # Current loss
                        loss_val_mean+=loss_val.item()
            loss_val_mean=loss_val_mean/len(val_loader)
            endtime=time.time()
            print('Test time is %f'%(endtime-starttime))
            log.write('Test time is %f\n'%(endtime-starttime))

            #Save model in each epoch
            if True or loss_val_mean<minloss:
                print('saved at [epoch%d/%d]'%(epoch+1,num_epochs))
                log.write('saved at [epoch%d/%d]\n'\
                    %(epoch+1,num_epochs))
                torch.save(net.state_dict(), 
                        "%s/%s_UNet_Model_Tr%s_epoch%s_lr%.2e.pkl"\
                        %(out_dir,prefix,TrainInstances,num_epochs,learning_rate))
                minloss=min(loss_val_mean,minloss)
        
            # Print loss
            if (epoch + 1)%1==0:    # % 10
                print('Epoch [%d/%d], Lossmean: %.5e, Validation lossmean: %.5e'\
                    %(epoch+1,num_epochs,loss_mean,loss_val_mean))
                log.write('Epoch [%d/%d], Lossmean: %.5e, Validation lossmean: %.5e\n'\
                    %(epoch+1,num_epochs,loss_mean,loss_val_mean))

                if loss.item() > 100:
                    print('hitbadrut')
                    log.write('hitbadrut\n')
                    break
            
            lossmean_vec[epoch]=loss_mean
            lossmean_val_vec[epoch]=loss_val_mean

            np.savez('%s/%s_UNet_LossData_Tr%s_epoch%s_lr%.2e'\
                %(out_dir,prefix,TrainInstances,num_epochs,learning_rate),
                lossmean_vec,lossmean_val_vec)

        """Save logs, prediction, loss figure, loss data, model and settings """
        np.savez('%s/%s_UNet_LossData_Tr%s_epoch%s_lr%.2e'\
                %(out_dir,prefix,TrainInstances,num_epochs,learning_rate),
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
        file=open('%s/%s_UNet_Settings_Tr%s_epoch%s_lr%.2e.txt'\
                %(out_dir,prefix,TrainInstances,num_epochs,learning_rate),'w')
        file.write('Training Settings:\n\t')
        for k,v in params.items():
            file.write(k+'='+str(v)+'\n')
            file.write('\t')
        file.close()
        
        #Print min loss
        print('\nmin Loss=%.3e'%minloss)
        log.write('\nmin Loss=%.3e'%minloss)
        log.close()
