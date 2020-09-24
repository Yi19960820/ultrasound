import numpy as np
from scipy.io import loadmat
import os
import h5py
import random
import tqdm
import yaml
from scipy.linalg import svd
from scipy.ndimage import zoom
from PIL import Image

cfg = yaml.safe_load(open('/data/prepare.yaml'))
DATA_FILE = '/data/dopperiq.mat'
OUT_DIR = cfg['datadir']
if not os.path.isdir(OUT_DIR):
    os.mkdir(OUT_DIR)
m = cfg['m']
n = cfg['n']
p = cfg['nframes']
nsamples = cfg['nsamples']
rank = cfg['threshrank']

# data = loadmat(DATA_FILE)
datafile = h5py.File(DATA_FILE, 'r')
data = datafile['iq'].value.view(np.complex128)
data = np.moveaxis(data, range(4), (3,2,1,0))
n1, n2, n3, n4 = data.shape
data_res = data.reshape(n1, n2, n3*n4)
caso = data_res.reshape((n1*n2, n3*n4))
U, s, Vh = svd(caso, full_matrices=False)
caso_red = U[:,rank:]@np.diag(s[rank:])@(Vh[:,rank:].T)
data_res = caso_red.reshape((n1, n2, n3*n4))

# Invivo pixel size is lambda/2 in z and lambda in x, vs. lambda/4 in both in sim
# But invivo lambda is 1/3 of sim lambda
# z_width = int(m*1.5)
# x_width = int(n*0.75)
z_width = m
x_width = n

for i in tqdm.tqdm(range(nsamples)):
    z = random.randint(0, n1-z_width)
    x = random.randint(0, n2-x_width)
    patch = data_res[z:z+z_width, x:x+x_width, 0:p]
    resized_r = np.zeros((m,n,p))
    resized_i = np.zeros((m,n,p))
    for k in range(p):
        resized_r[:,:,k] = np.rot90(patch.real[:,:,k])
        resized_i[:,:,k] = np.rot90(patch.imag[:,:,k])
        # resized_r[:,:,k] = np.array(Image.fromarray(patch.real[:,:,k]).resize((m,n), Image.BICUBIC))
        # resized_i[:,:,k] = np.array(Image.fromarray(patch.imag[:,:,k]).resize((m,n), Image.BICUBIC))
    patch = resized_r+1j*resized_i
    np.savez_compressed(os.path.join(OUT_DIR, f'{i}.npz'), patch=patch, z=z, x=x)