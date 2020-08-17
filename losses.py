import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import re

def scrape_log(fname):
    with open(fname, 'r') as log:
        tr_loss = []
        val_loss = []
        for line in log:
            stripped = line.strip()
            if 'Lossmean' in stripped:
                losses = re.findall(r'-?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?', stripped)
                tr_loss.append(float(losses[2]))
                val_loss.append(float(losses[3]))
    return tr_loss, val_loss

if __name__ == "__main__":
    # data1 = np.load(os.path.abspath('../rps-results/sim_Res3dC_LossData_Tr4000_epoch50_lr1.00e-03.npz'))
    # train = data1['arr_0']
    # val = data1['arr_1']
    train = []
    val = []
    for i in range(1, len(sys.argv)):
        train1, val1 = scrape_log(sys.argv[i])
        train.extend(train1)
        val.extend(val1)
    train = np.array(train)
    val = np.array(val)
    # np.savez_compressed(os.path.join(os.path.abspath('../rn-rps-real'), f'lossdata_{len(train)}.npz'), train=train, val=val)
    plt.semilogy(range(len(train)), train, label='Train loss')
    plt.semilogy(range(len(val)), val, label='Val loss')
    plt.title('ResNet train and val losses')
    plt.xlabel('Epoch')
    plt.ylabel('MSE lossmean')
    plt.legend()
    plt.show()