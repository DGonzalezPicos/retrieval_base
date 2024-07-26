import numpy as np
import matplotlib.pyplot as plt
import pathlib

from retrieval_base.PT_profile import PT_profile_SPHINX
from retrieval_base.sphinx import SPHINX
import config_freechem as conf
import retrieval_base.auxiliary_functions as af

p = np.logspace(-5, 2, 40)

sp = SPHINX(path=af.get_path()+'SPHINX')

sp.load_interpolator(species=conf.chem_kwargs['species'])
# sp.load_interpolator(species={})

free_params = {'Teff': conf.free_params['Teff'][0],
                'logg': conf.free_params['log_g'][0],
                'Z': conf.free_params['Z'][0],
                'C_O': conf.free_params['C_O'][0]}

n = 5

# uniform prior
params = {}

fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
for i in range(n):
    cube = np.random.rand(len(free_params))
    for j, key in enumerate(free_params.keys()):
        params[key] = free_params[key][0] + cube[j] * (free_params[key][1] - free_params[key][0])
        
    t = sp.temp_interpolator(params['Teff'], params['logg'], params['Z'], params['C_O'])
    nans = np.isnan(t)
    if np.all(nans):
        print(f'AFTER All nans for {params}')
        continue
    
    ax[0].plot(t, sp.pressure_full,alpha=0.8)
    # ax[1].plot()
    # for s in sp.species:
    for s in ['CO']:
        ax[1].plot(sp.vmr_interpolator[s](params['Teff'], params['logg'], params['Z'], params['C_O']), sp.pressure_full, label=s, alpha=0.8)
        
    
ax[0].set(yscale='log', ylabel='Temperature [K]', xlabel='Pressure [bar]', ylim=(np.max(sp.pressure_full), np.min(sp.pressure_full)))
ax[1].set(xscale='log', xlim=(1e-10, 1e-1))
plt.show()
    
               