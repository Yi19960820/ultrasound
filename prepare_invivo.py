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
m, n, p = 40,40,13
nsamples = 300
rank = 3

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
    z = random.randint(0, n1-m*3)
    x = random.randint(0, n2-n*3)
    patch = data_res[z:z+m*3, x:x+n*3, 0:p]
    resized_r = np.zeros((m,n,p))
    resized_i = np.zeros((m,n,p))
    for k in range(p):
        resized_r[:,:,k] = np.array(Image.fromarray(patch.real[:,:,k]).resize((m,n)), Image.BICUBIC)
        resized_i[:,:,k] = np.array(Image.fromarray(patch.imag[:,:,k]).resize((m,n)), Image.BICUBIC)
    patch = resized_r+1j*resized_i
    np.savez_compressed(os.path.join(OUT_DIR, f'{i}.npz'), patch=patch, z=z, x=x)