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
TISSUE_BOOST = 2
tissue_names = os.listdir(LR_DIR)
sd_names = os.listdir(SD_DIR)
random.shuffle(tissue_names)
random.shuffle(sd_names)

for i in tqdm.tqdm(range(min(len(tissue_names), len(sd_names)))):
    for x in (1,2):
        for z in (1,2):
            blood = loadmat(os.path.join(SD_DIR, sd_names[i]))['bloodData']
            blood = blood[int(39*(z-1)):int(39*z), int(39*(x-1)):int(39*x)]*TISSUE_BOOST
            tissue = loadmat(os.path.join(LR_DIR, tissue_names[i]))['acc3']
            tissue = tissue[int(39*(z-1)):int(39*z), int(39*(x-1)):int(39*x)]
            np.savez_compressed(os.path.join(OUT_DIR, f'{i}_x{x}_z{z}'), L=tissue, S=blood)