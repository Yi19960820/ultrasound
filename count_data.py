import numpy as np
from tqdm import tqdm
import os

if __name__=='__main__':
    datadir = '/data/toy-real-ranked/'
    counts = np.zeros(4)
    ratios = np.zeros(4)
    for fname in tqdm(os.listdir(datadir)):
        data = np.load(os.path.join(datadir, fname))
        rank = data['nsv']
        ratio = data['coeff']
        counts[rank-1] += 1
        ratios[int(ratio*2)-2] += 1
    print(counts)
    print(ratios)