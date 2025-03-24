
import matplotlib.pyplot as plt
import pathlib
import numpy as np
from retrieval_base.slab_grid import SlabGrid
from retrieval_base.auxiliary_functions import get_path

path = pathlib.Path(get_path())
grating = 'g395h'
T_ex_range = np.array([600.0, 800.0, 1200.0])
log_N_mol = np.array([18.0])
N_mol_range = 10**log_N_mol
species = ['12CO']

# init dictionary with keys from species
slabs = []

wave_steps = [1e-6, 5e-6, 1e-5]
for wave_step in wave_steps:
    slab = SlabGrid(species=species[0], grating=grating, path=path)
    slab.wave_step = wave_step
    m = slab.get_flux(T_ex_range[0], N_mol_range[0])
    slabs.append(m)



fig, ax = plt.subplots(2,1,figsize=(14,6),sharex=True, gridspec_kw={'height_ratios': [3,1]})

for i, m in enumerate(slabs):
    ax[0].plot(m.wave, m.flux, label=f'wave_step = {wave_steps[i]:.1e}', alpha=0.8)
    if i > 0:
        flux_ref = np.interp(m.wave, slabs[0].wave, slabs[0].flux)
        residuals = (slabs[i].flux - flux_ref) / flux_ref
        residuals[:10] = np.nan
        residuals[-10:] = np.nan
        ax[1].plot(m.wave, residuals, label=f'wave_step = {wave_steps[i]:.1e}', alpha=0.8)

ax[0].legend()
ax[0].set_ylabel('Flux')
ax[0].set_xlabel('Wavelength (um)')
ax[0].set_title('Slab model')
ax[1].legend()
ax[1].set_ylabel('Residuals')
ax[1].set_xlabel('Wavelength (um)')
ax[1].set_title('Residuals')

plt.show()
