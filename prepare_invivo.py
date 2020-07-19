import numpy as np
from scipy.io import loadmat
import os
import h5py
import random
import tqdm

DATA_FILE = '/data/dopperiq.mat'
OUT_DIR = '/data/invivo-samples/'
m, n, p = 39,39,20
nsamples = 300

# data = loadmat(DATA_FILE)
datafile = h5py.File(DATA_FILE, 'r')
data = datafile['iq'].value.view(np.complex128)
data = np.moveaxis(data, range(4), (3,2,1,0))
n1, n2, n3, n4 = data.shape
data_res = data.reshape(n1, n2, n3*n4)

for i in tqdm.tqdm(range(nsamples)):
    z = random.randint(0, n1-m)
    x = random.randint(0, n2-n)
    patch = data_res[z:z+m, x:x+n, 0:p]
    np.savez_compressed(os.path.join(OUT_DIR, f'{i}.npz'), patch=patch, z=z, x=x)