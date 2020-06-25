import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.linalg as la

def log_rms(mat):
    meansquare = np.sum(np.abs(mat)**2, axis=2, dtype=float)
    # logplot = 10*np.log10(meansquare/np.amax(meansquare))
    logplot = meansquare/np.amax(meansquare)
    return logplot

def plot_patches():
    outputs = loadmat('/home/sam/Documents/mats/6.mat')
    D = outputs['D']
    Sp = outputs['Sp']
    S = outputs['S']

    fig, ax = plt.subplots(2,3, figsize=(9,6))
    plt.set_cmap('hot')

    svals, Drec = svt(D, 4)

    ax[0][0].imshow(log_rms(D))
    ax[0][0].set_title('Input')

    ax[0][1].imshow(log_rms(S))
    ax[0][1].set_title('Ground truth S')

    ax[0][2].imshow(log_rms(Sp))
    ax[0][2].set_title('Reconstructed S')

    ax[1][0].semilogy(range(1, len(svals)+1), svals)
    ax[1][0].set_title('Singular values')

    ax[1][1].imshow(log_rms(Drec))
    ax[1][1].set_title('SVT')

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

def svt(D,e1, e2=None):
    n1, n2, n3 = D.shape
    e1 -= 1     # change 1-indexed e.val number to 0-indexed array index
    caso = D.reshape((n1*n2, n3))
    U, S, Vh = la.svd(caso, full_matrices=False)
    print(S.shape)
    if e2 is None:
        e2 = n3
    casorec = U[:,e1:e2]@np.diag(S[e1:e2])@(Vh[:,e1:e2].T)
    Drec = casorec.reshape(D.shape)
    return S, Drec

if __name__=='__main__':
    plot_patches()
    # plot_loss()