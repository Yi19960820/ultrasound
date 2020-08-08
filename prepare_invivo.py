import numpy as np
from scipy.io import loadmat
import os
import h5py
import random
import tqdm
import yaml
from scipy.linalg import svd
from scipy.ndimage import zoom

cfg = yaml.safe_load(open('/data/prepare.yaml'))
DATA_FILE = '/data/dopperiq.mat'
OUT_DIR = cfg['datadir']
m, n, p = 40,40,13
nsamples = 300
rank = 7

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

for i in tqdm.tqdm(range(nsamples)):
    z = random.randint(0, n1-20)
    x = random.randint(0, n2-20)
    patch = data_res[z:z+20, x:x+20, 0:p]
    zoomed_real = np.zeros((60, 60, p))
    zoomed_imag = np.zeros((60,60,p))
    for k in range(p):
        zoomed_real[:,:,k] = zoom(patch.real[:,:,k], 3)
        zoomed_imag[:,:,k] = zoom(patch.imag[:,:,k], 3)
    patch = zoomed_real+1j*zoomed_imag
    cx = patch.shape[1]//2
    cz = patch.shape[0]//2
    patch = patch[cz-(m//2):cz+(m//2),cx-(n//2):cx+(n//2)]
    np.savez_compressed(os.path.join(OUT_DIR, f'{i}.npz'), patch=patch, z=z, x=x)