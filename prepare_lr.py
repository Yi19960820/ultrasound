import torch
from scipy.io import loadmat
import h5py
import numpy as np
import os
import random
import tqdm
from scipy.linalg import svd

def find_2nd(string, substring):
   return string.find(substring, string.find(substring) + 1)

LR_DIR = '/data/low-rank/'
SD_DIR = '/data/sim-data-better/'
OUT_DIR = '/data/toy-real/'
BLOOD_BOOST = 2
NSV = 4
sd_names = os.listdir(SD_DIR)
random.shuffle(sd_names)

for i in tqdm.tqdm(range(len(sd_names))):
    mats = loadmat(os.path.join(SD_DIR, sd_names[i]))
    blood = mats['blood']*BLOOD_BOOST
    tissue = mats['L']
    angle = mats['a']
    width = mats['b']

    n1, n2, n3 = tissue.shape
    caso = tissue.reshape((n1*n2, n3))
    U, s, Vh = svd(caso, full_matrices=False)
    caso_red = U[:,:NSV]@np.diag(s[:NSV])@(Vh[:,:NSV].T)
    tissue = caso_red.reshape((n1, n2, n3))
    for x in (1,2):
        for z in (1,2):
            
            blood = blood[int(39*(z-1)):int(39*z), int(39*(x-1)):int(39*x)]
            tissue = tissue[int(39*(z-1)):int(39*z), int(39*(x-1)):int(39*x)]

            # bw_start = sd_names[i].find('vesselwidth')+12
            # bw_end = find_2nd(sd_names[i][bw_start:], '_')+bw_start
            # width_str = sd_names[i][bw_start:bw_end]
            # width = float(width_str[:width_str.find('_')])
            # width_unit = width_str[width_str.find('_')+1:]
            # if width_unit=='mm':
            #     width *= 0.001
            # elif width_unit=='mum':
            #     width *= 1e-6

            np.savez_compressed(os.path.join(OUT_DIR, f'{i}_x{x}_z{z}'), \
                L=tissue, S=blood, width=width, angle=angle, nsv=NSV, x=x, z=z)