import numpy as np
from scipy.io import loadmat, savemat
import os
import random
import tqdm
from scipy.linalg import svd
import yaml

def svt(quad, pre_svd):
    m, n, p = quad.shape
    quad_caso = quad.reshape(m*n, p)
    Up, sp, Vhp = svd(quad_caso, full_matrices=False)
    quad_caso_red = Up[:,pre_svd:]@np.diag(sp[pre_svd:])@(Vhp[:,pre_svd:].T)
    return quad_caso_red.reshape(m, n, p)

if __name__=='__main__':
    cfg = yaml.safe_load(open('/data/prepare.yaml'))
    OUT_DIR = cfg['datadir']
    SD_DIR = cfg['sd_dir']
    if not os.path.isdir(OUT_DIR):
        os.mkdir(OUT_DIR)

    pre_svd = cfg['pre_svd']
    sd_names = os.listdir(SD_DIR)

    thresh_S = None
    if 'thresh_S' in cfg.keys():
        thresh_S = cfg['thresh_S']

    # print(OUT_DIR)

    for i in tqdm.trange(len(sd_names)):
        mats = dict(np.load(os.path.join(SD_DIR, sd_names[i])))
        D = mats.pop('D')
        S = mats.pop('S')
        quad = svt(D, pre_svd)
        if 'mask' in mats.keys():
            del mats['mask']
        if thresh_S:
            idx = np.abs(S) < thresh_S*np.max(np.abs(S))
            S[idx] = 0
            mask = (~idx).astype(S.dtype)
        else:
            mask = np.ones_like(S)
        
        np.savez_compressed(os.path.join(OUT_DIR, sd_names[i]), S=S, D=quad, mask=mask, **mats)