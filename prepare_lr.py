import torch
from scipy.io import loadmat
import h5py
import numpy as np
import os
import random
import tqdm

LR_DIR = '/data/low-rank/'
SD_DIR = '/data/sim-data/'
OUT_DIR = '/data/toy/'
tissue_names = os.listdir(LR_DIR)
sd_names = os.listdir(SD_DIR)
random.shuffle(sd_names)

for i in tqdm.tqdm(range(min(len(tissue_names), len(sd_names)))):
    blood = loadmat(os.path.join(SD_DIR, sd_names[i]))['bloodData']
    tissue = loadmat(os.path.join(LR_DIR, tissue_names[i]))['acc3']
    np.savez_compressed(os.path.join(OUT_DIR, f'{i}'), L=tissue, S=blood)