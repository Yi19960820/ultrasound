import numpy as np
from scipy.io import loadmat
import os
import random
import tqdm

DATA_FILE = '/data/dopperiq.mat'
OUT_DIR = '/data/invivo-samples/'
m, n, p = 39,39,20
nsamples = 300

data = loadmat(DATA_FILE)
n1, n2, n3, n4 = data['iq'].shape
data_res = data['iq'].reshape(n1, n2, n3*n4)

for i in tqdm.tqdm(range(nsamples)):
    z = random.randint(0, n1-m)
    x = random.randint(0, n2-n)
    patch = data_res[z:z+m, x:x+n, 0:p]
    np.savez_compressed(os.path.join(OUT_DIR, f'{i}.npz'), patch=patch, z=z, x=x)