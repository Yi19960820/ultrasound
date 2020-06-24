import torch
from scipy.io import loadmat
import h5py
import numpy as np
import os
import random
import tqdm

def find_2nd(string, substring):
   return string.find(substring, string.find(substring) + 1)

LR_DIR = '/data/low-rank/'
SD_DIR = '/data/sim-data/'
OUT_DIR = '/data/toy-widths/'
TISSUE_BOOST = 2
tissue_names = os.listdir(LR_DIR)
sd_names = os.listdir(SD_DIR)
random.shuffle(tissue_names)
random.shuffle(sd_names)

n = 0
for i in range(min(len(tissue_names), len(sd_names))):
    if n>24:
        break
    for x in (1,2):
        for z in (1,2):
            blood = loadmat(os.path.join(SD_DIR, sd_names[i]))['bloodData']
            blood = blood[int(39*(z-1)):int(39*z), int(39*(x-1)):int(39*x)]*TISSUE_BOOST
            tissue = loadmat(os.path.join(LR_DIR, tissue_names[i]))['acc3']
            tissue = tissue[int(39*(z-1)):int(39*z), int(39*(x-1)):int(39*x)]

            bw_start = sd_names[i].find('vesselwidth')+12
            print(sd_names[i])
            print(sd_names[i][bw_start:])
            bw_end = find_2nd(sd_names[i][bw_start:], '_')+bw_start
            width_str = sd_names[i][bw_start:bw_end]
            print(width_str)
            width = float(width_str[:width_str.find('_')])
            width_unit = width_str[width_str.find('_')+1:]
            if width_unit=='mm':
                width *= 0.001
            elif width_unit=='mum':
                width *= 1e-6

            np.savez_compressed(os.path.join(OUT_DIR, f'{i}_x{x}_z{z}'), L=tissue, S=blood, width=width)
    n += 1