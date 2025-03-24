import matplotlib.pyplot as plt
import pathlib
import numpy as np
from retrieval_base.slab_grid import SlabGrid
from retrieval_base.auxiliary_functions import get_path

path = pathlib.Path(get_path())
grating = 'g395h'
T_ex_range = np.array([600.0, 800.0, 1200.0])
log_N_mol = np.array([16, 17, 18])
N_mol_range = 10**log_N_mol
species = ['12CO', '13CO', 'H2O']
labels = ['$^{12}CO$', '$^{13}CO$', 'H$_2$O']
# init dictionary with keys from species
slabs = {}

fig, ax = plt.subplots(len(species),1,figsize=(14,3*len(species)), dpi=150, sharex=True, gridspec_kw={'hspace': 0.05},
                       constrained_layout=True)

for i, s in enumerate(species):
    slabs[s] = SlabGrid(species=s, grating=grating, path=path)
    N_scale = (1/68.0) if s == '13CO' else 1.0
    # slabs[s].wave_range = (4.5, 5.3) if s == 'H2O' else (4.2, 5.3)
        
    slabs[s].get_grid(T_ex_range, N_mol_range * N_scale, cache=(s == '12CO'))
    slabs[s].load_interpolator(del_flux_grid=True) # False to plot the grid
# m = slab.get_flux(T_ex, N_mol)


    for (T, N_mol) in zip(T_ex_range, N_mol_range * N_scale):
        flux_new = slabs[s].interpolate(T, N_mol)
        ax[i].plot(slabs[s].wave_grid,flux_new / np.nanmax(flux_new), label=f'T = {T:.0f} K, N = {N_mol:.0e} cm$^{-2}$, (F_max={np.nanmax(flux_new):.1e})', alpha=0.8)
        
    ax[i].set_ylabel(f'Flux normalized')
    ax[i].grid(True)
    ax[i].text(0.02, 0.92, labels[i], transform=ax[i].transAxes, ha='left', va='top', fontsize=20, weight='bold')


xlim = (4.3, 5.3)
ax[1].set_xlim(xlim)
ax[1].set_xlabel('Wavelength (um)')
ax[0].legend()
ax[1].legend()
print(f'--> Done')
# plt.show()
fig_path = path / 'twx_figs'
assert fig_path.exists(), f'--> {fig_path} does not exist'
fig_name = fig_path / 'slab_components.pdf'
fig.savefig(fig_name)
print(f'--> Saved {fig_name}')
plt.close()