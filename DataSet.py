from CORONA.Res3dC.DataSet_3dC import preprocess, ImageDataset
import torch
from scipy.io import loadmat
import h5py
import numpy as np
import os

class BigImageDataset(torch.utils.data.Dataset):
    DATA_DIR='/data/toy-real/'

    def __init__(self, NumInstances, shape, train, transform=None, data_dir=None):
        data_dir = self.DATA_DIR if data_dir is None else data_dir
        self.shape=shape
        self.fnames = os.listdir(data_dir)
        self.fnames.sort()

        # dummy image loader
        images_L = torch.zeros(tuple([NumInstances])+self.shape)
        images_S = torch.zeros(tuple([NumInstances])+self.shape)
        images_D = torch.zeros(tuple([NumInstances])+self.shape)

        #   --  TRAIN  --  RAT 1
        if train is 0:
            self.fnames = self.fnames[:NumInstances]            
            for n in range(NumInstances):
                if np.mod(n, 50) == 0: print('loading train set %s' % (n))
                L = np.load(os.path.join(data_dir, self.fnames[n]))['L']
                S = np.load(os.path.join(data_dir, self.fnames[n]))['S']
                D = L + S
                L,S,D=preprocess(L,S,D)
                
                images_L[n] = torch.from_numpy(L.reshape(self.shape)).float()
                images_S[n] = torch.from_numpy(S.reshape(self.shape)).float()
                images_D[n] = torch.from_numpy(D.reshape(self.shape)).float()
        #   --  VALIDATION -- RAT 2, 100 frames 
        if train is 1:
            IndParam = 2400
            self.fnames = self.fnames[IndParam:NumInstances+IndParam]
            for n in range(NumInstances):
                if np.mod(n, 50) == 0: print('loading train set %s' % (n))
                L = np.load(os.path.join(data_dir, self.fnames[n]))['L']
                S = np.load(os.path.join(data_dir, self.fnames[n]))['S']
                D = L + S
                L,S,D=preprocess(L,S,D)
                
                images_L[n] = torch.from_numpy(L.reshape(self.shape)).float()
                images_S[n] = torch.from_numpy(S.reshape(self.shape)).float()
                images_D[n] = torch.from_numpy(D.reshape(self.shape)).float()
        if train is 2:
            IndParam = 3000
            self.fnames = self.fnames[IndParam:NumInstances+IndParam]
            for n in range(NumInstances):
                if np.mod(n, 50) == 0: print('loading train set %s' % (n))
                L = np.load(os.path.join(data_dir, self.fnames[n]))['L']
                S = np.load(os.path.join(data_dir, self.fnames[n]))['S']
                D = L + S
                L,S,D=preprocess(L,S,D)
                
                images_L[n] = torch.from_numpy(L.reshape(self.shape)).float()
                images_S[n] = torch.from_numpy(S.reshape(self.shape)).float()
                images_D[n] = torch.from_numpy(D.reshape(self.shape)).float()
        
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
