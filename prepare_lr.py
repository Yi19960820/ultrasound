import torch
from scipy.io import loadmat
import h5py
import numpy as np
import os
import random
import tqdm
from scipy.linalg import svd
import yaml

def add_padding(arr, length):
    n1, n2, n3 = arr.shape
    padding = np.ones((n1, n2, length), dtype=arr.dtype)*np.mean(arr)
    return np.concatenate((padding, arr), axis=2)

def find_2nd(string, substring):
    '''
    Finds the start index of the second instance of `substring` in `string`.
    '''
    return string.find(substring, string.find(substring) + 1)

def create_quads(blood, tissue, x, z):
    '''
    Splits blood and tissue into quadrants. x,z are in [1,2] that define the quadrant.

    Arguments:
        blood : the blood array, at least 2D
        tissue : the tissue array, at least 2D
        x, int : the x quadrant index. 1=left, 2=right.
        z, int : the z quadrant index. 1=top, 2=bottom.
    
    Returns:
        array : the blood quadrant
        array : the tissue quandrant
    '''
    blood_quad = blood[int(39*(z-1)):int(39*z), int(39*(x-1)):int(39*x)]
    tissue_quad = tissue[int(39*(z-1)):int(39*z), int(39*(x-1)):int(39*x)]
    return blood_quad, tissue_quad

def create_random_quads(blood, tissue, x, z, maxdist, shape2d):
    '''
    Like create_quads, but with a random offset from the side that is between 0 and maxdist.

    Arguments:
        blood : the blood array, at least 2D
        tissue : the tissue array, at least 2D
        x, int : the x quadrant index. 1=left, 2=right.
        z, int : the z quadrant index. 1=top, 2=bottom.
        maxdist, int : the maximum distance, in array entries, to offset from the two sides the quadrant would align with if not\
for the offset. For example, the x=1, z=1 quadrant would be offset from the top and left by up to maxdist units.
        shape2d, (int, int) : the first two dimensions of blood and tissue.
    
    Returns:
        array : the blood quadrant
        array : the tissue quandrant
    '''
    n1, n2 = shape2d
    s1, s2, _ = blood.shape
    if x==1:
        xl = random.randint(0, maxdist)
        xr = xl+n2
    else:
        xr = s2-random.randint(0, maxdist)
        xl = xr-n2
    
    if z==1:
        zl = random.randint(0, maxdist)
        zr = zl+n1
    else:
        zr = s1-random.randint(0, maxdist)
        zl = zr-n1
    
    blood_quad = blood[zl:zr, xl:xr]
    tissue_quad = tissue[zl:zr, xl:xr]
    return blood_quad, tissue_quad

cfg = yaml.safe_load(open('/data/prepare.yaml'))
OUT_DIR = cfg['datadir']
SD_DIR = cfg['sd_dir']
merge = cfg['merge']
if 'nsamples' in cfg.keys():
    nsamples = cfg['nsamples']
else:
    nsamples = np.inf
if merge:
    TISSUE_DIR = cfg['tissue_dir']
if not os.path.isdir(OUT_DIR):
    os.mkdir(OUT_DIR)
m = cfg['m']
n = cfg['n']
NFRAMES = cfg['nframes']
NSV = cfg['nsv']
TB = cfg['tb']
noise = cfg['noise']
if noise:
    snr = cfg['snr']
sd_names = os.listdir(SD_DIR)
if merge:
    l_names = os.listdir(TISSUE_DIR)
if 'padding' in cfg.keys():
    padding = cfg['padding']
else:
    padding = 0
random.shuffle(sd_names)
if merge:
    random.shuffle(l_names)

for i in tqdm.tqdm(range(len(sd_names))):
    if i >= nsamples:
        break

    mats = loadmat(os.path.join(SD_DIR, sd_names[i]))
    coeff = random.choice([1,2,3,4,5,6,7,8,9,10])
    blood = mats['blood'][30:110,30:110,1:NFRAMES+1]*TB/coeff     # start from second frame because the first is weird sometimes
    if merge:
        tmats = loadmat(os.path.join(TISSUE_DIR, l_names[i]))
        tissue = tmats['L'][30:110,30:110,1:NFRAMES+1]
    else:
        tissue = mats['L'][30:110,30:110,1:NFRAMES+1]
    angle = mats['a']
    width = mats['b']

    n1, n2, n3 = tissue.shape
    caso = tissue.reshape((n1*n2, n3))
    U, s, Vh = svd(caso, full_matrices=False)
    for x in (1,2):
        for z in (1,2):
            rank = random.randint(1,NSV)
            caso_red = U[:,:rank]@np.diag(s[:rank])@(Vh[:,:rank].T)
            tissue_red = caso_red.reshape((n1, n2, n3))            
            blood_quad, tissue_quad = create_random_quads(blood, tissue_red, x, z, 10, (m, n))
            quad = blood_quad+tissue_quad

            # # Preprocess with SVT
            # quad_caso = quad.reshape(m*n, NFRAMES)
            # U, s, Vh = svd(quad_caso, full_matrices=False)
            # quad_caso_red = U[:,3:]@np.diag(s[3:])@(Vh[:,3:].T)
            # quad = quad_caso_red.reshape(m, n, NFRAMES)

            # Add Gaussian noise
            if noise:
                snr_raw = 10**(snr/10)   # SNR not in dB, defining as ratio of powers
                signal_power = np.mean(np.abs(tissue_quad)**2)  # power is square of RMS amplitude
                noise_power = signal_power/snr_raw
                noise_rms = np.sqrt(noise_power)
                radius = np.random.randn(*tissue_quad.shape)*noise_rms
                angle = np.random.rand(*tissue_quad.shape)*2*np.pi
                noise_quad = radius*(np.cos(angle)+np.sin(angle)*1j)
                quad = quad + noise_quad
            
            if padding > 0:
                tissue_quad = add_padding(tissue_quad, padding)
                blood_quad = add_padding(blood_quad, padding)
                quad = add_padding(quad, padding)

            np.savez_compressed(os.path.join(OUT_DIR, f'{i}_x{x}_z{z}'), L=tissue_quad, S=blood_quad, \
                D=quad, width=width, angle=angle, nsv=rank, x=x, z=z, coeff=coeff, padded=(padding>0))