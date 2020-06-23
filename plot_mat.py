import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

def log_rms(mat):
    meansquare = np.sum(np.abs(mat)**2, axis=2, dtype=float)
    logplot = 10*np.log10(meansquare/np.amax(meansquare))
    return logplot

def plot_patches():
    outputs = loadmat('/home/sam/Documents/mats/9.mat')
    D = outputs['D']
    Sp = outputs['Sp']
    S = outputs['S']

    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(7,3))
    print(S.dtype)
    plt.set_cmap('hot')
    ax1.imshow(log_rms(D))
    ax1.set_title('Input')
    ax2.imshow(log_rms(S))
    ax2.set_title('Ground truth S')
    ax3.imshow(log_rms(Sp))
    ax3.set_title('Reconstructed S')
    plt.show()

def plot_loss():
    losses1 = np.load('/home/sam/Documents/Res3dC_nocon_sim_Res3dC_LossData_Tr2400_epoch20_lr2.00e-03.npz')
    losses2 = np.load('/home/sam/Documents/working-20/Res3dC_nocon_sim_Res3dC_LossData_Tr2400_epoch20_lr2.00e-03.npz')
    lossmeans = np.hstack([losses2['arr_0'], losses1['arr_0']])
    lossmeans_val = np.hstack([losses2['arr_1'], losses1['arr_1']])
    plt.semilogy(range(1,41), lossmeans, '-s', label='Train loss', markersize=3)
    plt.semilogy(range(1,41), lossmeans_val, '-x', label='Validation loss', markersize=3)
    plt.title('Losses over 40 epochs of training base ResNet')
    plt.xlabel('Epoch')
    plt.ylabel('MSE loss')
    plt.legend()
    plt.show()

plot_patches()
# plot_loss()