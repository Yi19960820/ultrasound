import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

def log_rms(mat):
    meansquare = np.sum(np.abs(mat)**2, axis=2, dtype=float)
    logplot = 10*np.log10(meansquare/np.amax(meansquare))
    return logplot

if __name__=='__main__':
    outputs = loadmat('/home/sam/Documents/mats/0.mat')
    D = outputs['D']
    Sp = outputs['Sp']
    S = outputs['S']

    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    print(S.dtype)
    plt.set_cmap('hot')
    ax1.imshow(log_rms(D))
    ax2.imshow(log_rms(S))
    ax3.imshow(log_rms(Sp))
    plt.show()
