from CORONA.Res3dC.DataSet_3dC import preprocess, ImageDataset
import torch
from scipy.io import loadmat
import h5py
import numpy as np
import os

def preprocess_real(L, S, D):
    A=max(np.max(np.abs(L)),np.max(np.abs(S)),np.max(np.abs(D)))   
    if A==0:
        A=1
    L=np.abs(L)/A
    S=np.abs(S)/A
    D=np.abs(D)/A
    return L,S,D

class BigImageDataset(torch.utils.data.Dataset):
    DATA_DIR='/data/toy-real/'

    def __init__(self, NumInstances, shape, train, transform=None, data_dir=None, train_size=3200, val_size=800, gt=True, real=False):
        data_dir = self.DATA_DIR if data_dir is None else data_dir
        self.shape=shape
        self.fnames = os.listdir(data_dir)
        self.fnames.sort()

        if real:
            pp = preprocess
        else:
            pp = preprocess_real

        # dummy image loader
        images_L = torch.zeros(tuple([NumInstances])+self.shape)
        images_S = torch.zeros(tuple([NumInstances])+self.shape)
        images_D = torch.zeros(tuple([NumInstances])+self.shape)

        #   --  TRAIN  --  RAT 1
        if train is 0:
            self.fnames = self.fnames[:NumInstances]            
        #   --  VALIDATION -- RAT 2, 100 frames 
        elif train is 1:
            IndParam = train_size
            self.fnames = self.fnames[IndParam:NumInstances+IndParam]
        elif train is 2:
            IndParam = val_size+train_size
            self.fnames = self.fnames[IndParam:NumInstances+IndParam]
        
        for n in range(NumInstances):
            if np.mod(n, 50) == 0: print('loading train set %s' % (n))
            if gt:
                D = np.load(os.path.join(data_dir, self.fnames[n]))['D']
                S = np.load(os.path.join(data_dir, self.fnames[n]))['S']
                L = np.load(os.path.join(data_dir, self.fnames[n]))['L']
            else:
                D = np.load(os.path.join(data_dir, self.fnames[n]))['patch']
                L  = np.zeros_like(D)
                S = np.zeros_like(D)
            L,S,D = pp(L,S,D)
            try:
                images_L[n] = torch.from_numpy(L.reshape(self.shape)).float()
                images_S[n] = torch.from_numpy(S.reshape(self.shape)).float()
                images_D[n] = torch.from_numpy(D.reshape(self.shape)).float()
            except ValueError:
                print(n)
        
        self.transform = transform

        self.images_L = images_L
        self.images_S = images_S
        self.images_D = images_D
    
    def __getitem__(self, idx):
        # Do something here that will load the actual data from the list of datasets.
        # We want to use this lazy loading so we don't need 25 GB of RAM.

        # data = np.load(os.path.join(self.DATA_DIR, self.fnames[idx]))
        # L = data['L']
        # S = data['S']
        # D = L + S
        # L, S, D = preprocess(L, S, D)
        # L = torch.from_numpy(L.reshape(self.shape)).float()
        # S = torch.from_numpy(S.reshape(self.shape)).float()
        # D = torch.from_numpy(D.reshape(self.shape)).float()

        L = self.images_L[idx]
        S = self.images_S[idx]
        D = self.images_D[idx]

        return L, S, D
    
    def __len__(self):
        # return len(self.fnames)
        return len(self.images_L)
