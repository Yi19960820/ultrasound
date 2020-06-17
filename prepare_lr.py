import torch
from scipy.io import loadmat
import h5py
import numpy as np
import os
import random

LR_DIR = '/data/low-rank/'
OUT_DIR = '/data/toy/'
tissue_names = os.listdir(LR_DIR)
sim_data = h5py.File(os.path.join('/data/sim-data/','sim0614.h5'), 'r')
sd_names = list(sim_data.keys())
random.shuffle(sd_names)

for i in range(min(len(tissue_names), len(sd_names))):
    tb = sim_data.get(sd_names[i]).value
    blood = tb[2]+1j*tb[3]
    td = loadmat(os.path.join(LR_DIR, tissue_names[i]))['acc3']
    np.savez_compressed(os.path.join(OUT_DIR, f'_{i}'), blood, td)