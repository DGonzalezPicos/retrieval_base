import numpy as np
import matplotlib.pyplot as plt
import pathlib

from retrieval_base.auxiliary_functions import get_path

path = pathlib.Path(get_path())
target = 'gl15A'


files = sorted((path / target / 'data').glob('*48*npy'))

print(f' Number of files: {len(files)}')
n_orders = 3

fig, ax = plt.subplots(n_orders, 1, figsize=(10, 10))

for i, file in enumerate(files):
    wave, flux, err, transm = np.load(file)
    print(f' Loading {file.name}')
    for j in range(n_orders):
        
        snr_ij = np.nanmedian(flux[j] / err[j])

        ax[j].plot(wave[j], flux[j], label=f'SNR {snr_ij:.1f}', alpha=0.7)
        
        if i == len(files) - 1:
            ax[j].legend()
        
        
plt.show()
    