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

# print(stop)
# get all atms files
files = sorted((sp.path / 'ATMS').glob('*atms.txt'))

def get_attrs(f):
    Teff = float(f.name.split('_')[1])
    logg = float(f.name.split('_')[3])
    Z_sign = f.name.split('_')[5][0]
    Z_abs = float(f.name.split('_')[5][1:])
    Z = Z_abs * (-1)**(Z_sign == '-')
    
    C_O = float(f.name.split('_')[7])
    return Teff, logg, Z, C_O

# create interpolator for 4D grid 
Teffs, loggs, Zs, C_Os = [], [], [], []
temps = []
points = []
for f in files:
    Teff, logg, Z, C_O = get_attrs(f)
    # print(f'Checking {f.name:90}', end='\r')
    sp.set_attrs(Teff=Teff, logg=logg, Z=Z, C_O=C_O)
    sp.get_file(kind='atms', update=True)
    sp.load_PT()
    nans = np.isnan(sp.temperature)
    if np.all(nans):
        print(f'All nans for {f.name}')
        continue
    
    points.append((Teff, logg, Z, C_O))
    Teffs.append(Teff)
    loggs.append(logg)
    Zs.append(Z)
    C_Os.append(C_O)
    n_layers = len(sp.temperature)
    # print(f'N layers: {n_layers}')
    # print(f' Mean temperature {np.mean(sp.temperature)}')
    temps.append(sp.interp_makima(p, sp.pressure, sp.temperature, smooth_nans=True))
    
# create interpolator
from scipy.interpolate import LinearNDInterpolator

points = np.array(points)
temps = np.array(temps)
interpolator = LinearNDInterpolator(points, temps)


# test interpolator
free_params = {'Teff':conf.free_params['Teff'][0],
                'logg':conf.free_params['log_g'][0],
                'Z':conf.free_params['Z'][0],
                'C_O':conf.free_params['C_O'][0]
}

n = 10

for i in range(n):
    
    # uniform prior
    cube = np.random.rand(len(free_params))
    for j, key in enumerate(free_params.keys()):
        cube[j] = free_params[key][0] + cube[j] * (free_params[key][1] - free_params[key][0])
        
    Teff, logg, Z, C_O = cube
    # params_dict = dict(zip(free_params.keys(), cube))
    temp = interpolator((Teff, logg, Z, C_O))
    nans = np.isnan(temp)
    
    print(f'Teff: {Teff:2f}, logg: {logg:2f}, Z: {Z:2f}, C_O: {C_O:2f}')
    if np.sum(nans) > 0:
        print(f'WARNING: Iter {i} --> Nans in temp: {np.sum(nans)}')
        print(f'--> Teff: {Teff}, logg: {logg}, Z: {Z}, C_O: {C_O}')
        break
    