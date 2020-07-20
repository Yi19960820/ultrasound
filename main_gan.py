# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 2020

@author: Sam Ehrenstein
"""

import matplotlib 
matplotlib.use('Agg')

import torch
import torch.nn as nn
from torch.autograd import Variable

import torch.utils.data as data

import sys
# sys.path.append('../')
import os

from DataSet import BigImageDataset
from UGAN import UGenerator, UDiscriminator
from CORONA.classes.Dataset import Converter
from CORONA.classes.Player import Player

import numpy as np
import matplotlib.pyplot as plt
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
    ProjectName='gan'
    prefix='sim' #invivo,sim_pm,sim
    #Load model

    """Network Settings: Remember to change the parameters when you change model!"""
    gpu=True #if gpu=True, the ResNet will use more parameters
    #Whether to plot predictions during training and frequency
    plot=True
    plotT=1
    if not plot:
        plt.ioff()
    #seed
    seed=1234
    torch.manual_seed(seed)
    #parameters for training
    TrainInstances = 6000 # Size of training dataset
    ValInstances   = 800
    BatchSize      = 40
    ValBatchSize   = 40
    num_epochs     = 30
    m, n, p = (39,39,20)
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
        gfile = cfg['gfile']
        dfile = cfg['dfile']
    print(loadmodel)
    """========================================================================="""

    #Dataset, converter and player
    data_dir={'invivo':d_invivo,'sim_pm':d_simpm,'sim':d_sim}[prefix]
    conter=Converter()
    player=Player()
    formshow={'pre':'concat','shape':(m,n,p)}
    formlist=[]
    for i in range(6):
        formlist.append(formshow)
    minloss=np.inf
    #Logs
    log=open('/results/%s_%s_GAN_Log_Tr%s_epoch%s_lr%.2e.txt'\
        %(ProjectName,prefix,TrainInstances,num_epochs,lr_list[0]),'a')

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
            generator = UGenerator(gpu)
            discriminator = UDiscriminator((m, n, p), gpu)
        else:
            if gfile[-3:]=='pkl':
                generator = UGenerator(gpu)
                discriminator = UDiscriminator((m, n, p), gpu)
                g_state_dict=torch.load(gfile, map_location='cuda:0')
                d_state_dict=torch.load(dfile, map_location='cuda:0')
                generator.load_state_dict(g_state_dict)
                discriminator.load_state_dict(d_state_dict)
            else:
                generator = torch.load(gfile)
                discriminator = torch.load(dfile)

        if torch.cuda.is_available():
            generator = generator.cuda()
            discriminator = discriminator.cuda()

        #Loss and optimizer
        adv_loss=nn.BCELoss()
        if torch.cuda.is_available():
            adv_loss = adv_loss.cuda()
        
        g_optimizer=torch.optim.Adam(generator.parameters(), lr=learning_rate)
        d_optimizer=torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

        #Array for recording data
        outputs_S = to_var(torch.zeros([1,1,m,m,p*2]))
        g_lossmean_vec=np.zeros((num_epochs,))
        g_lossmean_val_vec=np.zeros((num_epochs,))
        d_lossmean_vec=np.zeros((num_epochs,))
        d_lossmean_val_vec=np.zeros((num_epochs,))

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
            
            gen_loss_mean = 0
            disc_loss_mean = 0
            # Train
            print('Loading and calculating training batches...')
            log.write('Loading and calculating training batches...\n')
            starttime=time.time()
            torch.autograd.set_detect_anomaly(True)
            for _,(_,S,D) in tqdm(enumerate(train_loader)):
                valid = to_var(torch.ones(BatchSize, 1))
                valid.requires_grad = False
                fake = to_var(torch.zeros(BatchSize, 1))
                fake.requires_grad = False
                Sg = to_var(S)
                Dg = to_var(D)

                # Train discriminator on real batch
                d_optimizer.zero_grad()
                real_out = discriminator(Sg[None].transpose(0,1))
                real_loss = adv_loss(real_out, valid)
                
                # Train discriminator on fake batch from generator
                fake_batch = generator(Dg[None].transpose(0,1))
                fake_out = discriminator(fake_batch)
                fake_loss = adv_loss(fake_out, fake)
                
                # Combine losses and step optimizer
                loss_d = real_loss + fake_loss
                loss_d.backward()
                d_optimizer.step()

                # Now train the generator to fool the discriminator
                g_optimizer.zero_grad()
                gen_loss = adv_loss(fake_batch, valid)
                gen_loss.backward()
                g_optimizer.step()

                gen_loss_mean += gen_loss.item()
                disc_loss_mean += loss_d.item()

            gen_loss_mean /= TrainInstances
            disc_loss_mean /= TrainInstances
            endtime=time.time()
            g_lossmean_vec[epoch] = gen_loss_mean
            d_lossmean_vec[epoch] = disc_loss_mean
            print('Training time is %f'%(endtime-starttime))
            log.write('Training time is %f\n'%(endtime-starttime))
            
            #Save model in each epoch
            print('saved at [epoch%d/%d]'%(epoch+1,num_epochs))
            log.write('saved at [epoch%d/%d]\n'\
                %(epoch+1,num_epochs))
            torch.save(generator.state_dict(), 
                    "/results/%s_%s_Gen_Model_Tr%s_epoch%s_lr%.2e.pkl"\
                    %(ProjectName,prefix,TrainInstances,num_epochs,learning_rate))
            torch.save(discriminator.state_dict(), 
                    "/results/%s_%s_Dis_Model_Tr%s_epoch%s_lr%.2e.pkl"\
                    %(ProjectName,prefix,TrainInstances,num_epochs,learning_rate))
        
            # Print loss
            if (epoch + 1)%1==0:    # % 10
                print('Epoch [%d/%d], G lossmean: %.5e, D lossmean: %.5e'\
                    %(epoch+1,num_epochs,gen_loss_mean,disc_loss_mean))

                if gen_loss.item() > 100:
                    print('hitbadrut')
                    log.write('hitbadrut\n')
                    break

            np.savez('/results/%s_%s_GAN_LossData_Tr%s_epoch%s_lr%.2e'\
                %(ProjectName,prefix,TrainInstances,num_epochs,learning_rate),
                g_lossmean_vec, d_lossmean_vec)

        """Save logs, prediction, loss figure, loss data, model and settings """
        #Save settings of training
        params={'ProjectName':ProjectName,
                'prefix':prefix,
                'gfile':gfile if loadmodel else None,
                'dfile':dfile if loadmodel else None,
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
        file=open('/results/%s_%s_GAN_Settings_Tr%s_epoch%s_lr%.2e.txt'\
                %(ProjectName,prefix,TrainInstances,num_epochs,learning_rate),'w')
        file.write('Training Settings:\n\t')
        for k,v in params.items():
            file.write(k+'='+str(v)+'\n')
            file.write('\t')
        file.close()