import numpy as np
import matplotlib.pyplot as plt
import os

data1 = np.load(os.path.abspath('../sim_Res3dC_LossData_Tr4000_epoch30_lr5.00e-03.npz'))
train = data1['arr_0']
val = data1['arr_1']
plt.semilogy(range(len(train)), train, label='Train loss')
plt.semilogy(range(len(val)), val, label='Val loss')
plt.title('ResNet train and val losses')
plt.xlabel('Epoch')
plt.ylabel('MSE lossmean')
plt.legend()
plt.show()