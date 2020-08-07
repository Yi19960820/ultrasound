import numpy as np
from scipy.io import loadmat
import sys
import os
import yaml

if __name__=='__main__':
    cfg = yaml.safe_load(open('/data/prepare.yaml'))
    data_dir = cfg['datadir']
    out_dir = cfg['outdir']
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    snr = cfg['snr']
    files = os.listdir(data_dir)
    for f in files:
        sample = np.load(os.path.join(data_dir, f))
        tissue_quad = sample['L']
        blood_quad = sample['S']
        quad = sample['D']
        width = sample['width']
        angle = sample['angle']
        rank = sample['rank']
        x = sample['x']
        z = sample['z']
        coeff = sample['coeff']

        snr_raw = 10**(snr/10)   # SNR not in dB, defining as ratio of powers
        signal_power = np.mean(np.abs(tissue_quad)**2)  # power is square of RMS amplitude
        noise_power = signal_power/snr_raw
        noise_rms = np.sqrt(noise_power)
        radius = np.random.randn(*tissue_quad.shape)*noise_rms
        angle = np.random.rand(*tissue_quad.shape)*2*np.pi
        noise_quad = radius*(np.cos(angle)+np.sin(angle)*1j)
        quad = quad + noise_quad

        np.savez_compressed(os.path.join(out_dir, f), L=tissue_quad, S=blood_quad, \
            D=quad, width=width, angle=angle, nsv=rank, x=x, z=z, coeff=coeff, padded=False)