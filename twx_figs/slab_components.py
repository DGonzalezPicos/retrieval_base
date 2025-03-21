import matplotlib.pyplot as plt
import pathlib
import numpy as np
from retrieval_base.slab_grid import SlabGrid
from retrieval_base.auxiliary_functions import get_path

path = pathlib.Path(get_path())
grating = 'g395h'
T_ex_range = np.array([200.0, 600.0, 1000.0])
log_N_mol = np.array([14, 17, 18])
N_mol_range = 10**log_N_mol
species = ['12CO', '13CO']

# init dictionary with keys from species
slabs = {}

fig, ax = plt.subplots(2,1,figsize=(14,6), dpi=150, sharex=True, gridspec_kw={'hspace': 0.05},
                       constrained_layout=True)

for i, s in enumerate(species):
    slabs[s] = SlabGrid(species=s, grating=grating, path=path)
    N_scale = 1.0 if s == '12CO' else (1/68.0)
        
    slabs[s].get_grid(T_ex_range, N_mol_range * N_scale, cache=True)
    slabs[s].load_interpolator(del_flux_grid=True) # False to plot the grid
# m = slab.get_flux(T_ex, N_mol)


    for (T, N_mol) in zip(T_ex_range, N_mol_range * N_scale):
        flux_new = slabs[s].interpolate(T, N_mol)
        ax[i].plot(slabs[s].wave_grid,flux_new / np.nanmax(flux_new), label=f'T = {T:.0f} K, N = {N_mol:.0e} cm$^{-2}$', alpha=0.8)
        
ax[0].legend()
ax[0].set_ylabel('Flux normalized to max')
ax[0].grid(True)
ax[0].text(0.02, 0.92, r'$\rm ^{12}CO$', transform=ax[0].transAxes, ha='left', va='top', fontsize=20, weight='bold')

ax[1].legend()
ax[1].set_ylabel('Flux normalized to max')
ax[1].grid(True)
ax[1].text(0.02, 0.92, r'$\rm ^{13}CO$', transform=ax[1].transAxes, ha='left', va='top', fontsize=20, weight='bold')

xlim = (4.3, 5.3)
ax[1].set_xlim(xlim)
ax[1].set_xlabel('Wavelength (um)')

print(f'--> Done')
# plt.show()
fig_path = path / 'twx_figs'
assert fig_path.exists(), f'--> {fig_path} does not exist'
fig_name = fig_path / 'slab_components.pdf'
fig.savefig(fig_name)
print(f'--> Saved {fig_name}')
plt.close()