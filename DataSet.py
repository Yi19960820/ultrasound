from CORONA.Res3dC.DataSet_3dC import preprocess, ImageDataset
import torch
from scipy.io import loadmat
import h5py
import numpy as np
import os

class BigImageDataset(torch.utils.data.Dataset):
    DATA_DIR='/data/toy/'

    def __init__(self, NumInstances, shape, train, transform=None, data_dir=None):
        data_dir = self.DATA_DIR if data_dir is None else data_dir
        self.shape=shape
        self.fnames = os.listdir(data_dir)
        self.fnames.sort()

        # dummy image loader
        # images_L = torch.zeros(tuple([NumInstances])+self.shape)
        # images_S = torch.zeros(tuple([NumInstances])+self.shape)
        # images_D = torch.zeros(tuple([NumInstances])+self.shape)
        
        #   --  TRAIN  --  RAT 1
        if train is 0:
            self.fnames = self.fnames[:NumInstances]            
            # for n in range(NumInstances):
            #     if np.mod(n, 600) == 0: print('loading train set %s' % (n))
            #     L,S,D=preprocess(L,S,D)
                
            #     images_L[n] = torch.from_numpy(L.reshape(self.shape))
            #     images_S[n] = torch.from_numpy(S.reshape(self.shape))
            #     images_D[n] = torch.from_numpy(D.reshape(self.shape))
        #   --  VALIDATION -- RAT 2, 100 frames 
        if train is 1:
            IndParam = 1323
            self.fnames = self.fnames[NumInstances:NumInstances+IndParam]
            # for n in range(IndParam, IndParam + NumInstances):
            #     if np.mod(n - IndParam, 200) == 0: print('loading validation set %s' % (n - IndParam))
            #     L=loadmat(data_dir + '/fista/val/L_fista%s.mat' % (n))['patch_180']
            #     S=loadmat(data_dir + '/fista/val/S_fista%s.mat' % (n))['patch_180']
            #     D=loadmat(data_dir + '/D_data/val/D%s.mat' % (n))['patch_180']
            #     L,S,D=preprocess(L,S,D)
                
            #     images_L[n-IndParam] = torch.from_numpy(L.reshape(self.shape))
            #     images_S[n-IndParam] = torch.from_numpy(S.reshape(self.shape))
            #     images_D[n-IndParam] = torch.from_numpy(D.reshape(self.shape))
        
        self.transform = transform

        # self.images_L = images_L
        # self.images_S = images_S
        # self.images_D = images_D
    
    def __getitem__(self, idx):
        # Do something here that will load the actual data from the list of datasets.
        # We want to use this lazy loading so we don't need 25 GB of RAM.

        # data_file = self.fnames[idx]
        # hf = h5py.File(self.fname, 'r')
        # data_arr = hf.get(data_file).value
        # L = data_arr[0]+1j*data_arr[1]
        # S = data_arr[2]+1j*data_arr[3]
        data = np.load(os.path.join(self.DATA_DIR, self.fnames[idx]))
        L = data['td']
        S = data['blood']
        D = L + S
        L, S, D = preprocess(L, S, D)
        L = torch.from_numpy(L.reshape(self.shape))
        S = torch.from_numpy(S.reshape(self.shape))
        D = torch.from_numpy(D.reshape(self.shape))
        return L, S, D
    
    def __len__(self):
        return len(self.fnames)
