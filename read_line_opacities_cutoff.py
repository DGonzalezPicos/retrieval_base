from petitRADTRANS import Radtrans
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import os
import sys
from pympler import asizeof


line_species = ['CO_high_Sam', 'H2O_pokazatel_main_iso', 'HF_high',
                'K_high', 'Ca_high', 'Ti_high', 'Fe_high',
                'Sc_high', 'Na_allard_high']
                
atm = Radtrans(
                line_species=line_species,
                rayleigh_species=['H2', 'He'],
                continuum_opacities=['H2-H2', 'H2-He'],
                cloud_species=None,
                wlen_bords_micron=[1.72, 3.29],
                mode='lbl',
                lbl_opacity_sampling=10,
                )


# attributes of interest: [custom_line_TP_grid, custom_line_paths]
T_min = 1200.0
T_max = 3600.0

def apply_PT_cutoff(atm, T_min, T_max, P_min=1e-4, P_max=1e2):
   
    # convert P from bar to cgs
    P_min_cgs = P_min * 1e6
    P_max_cgs = P_max * 1e6
    
    for i, species in enumerate(atm.line_species):
        if atm.custom_grid[species]:
            new_custom_line_TP_grid = [] # old has shape (152, 2) for each PT pair
            new_custom_line_paths = []
            new_line_grid_kappas_custom_PT = []
            for j, (T_j, P_j) in enumerate(atm.custom_line_TP_grid[species]):
                # print(f' T_j = {T_j}, P_j = {P_j}')
                if T_j >= T_min and T_j <= T_max and P_j >= P_min_cgs and P_j <= P_max_cgs:
                    new_custom_line_TP_grid.append([T_j, P_j])
                    new_custom_line_paths.append(atm.custom_line_paths[species][j])
                    new_line_grid_kappas_custom_PT.append(atm.line_grid_kappas_custom_PT[species][:,:,j])
                
            print(f' Number of PT pairs {species}:\n -> before = {len(atm.custom_line_TP_grid[species])} \n -> after = {len(new_custom_line_TP_grid)}')
            # save new values
            atm.custom_line_TP_grid[species] = np.array(new_custom_line_TP_grid)
            atm.custom_line_paths[species] = np.array(new_custom_line_paths)
            atm.line_grid_kappas_custom_PT[species] = np.moveaxis(np.array(new_line_grid_kappas_custom_PT), 0, 2)
            
            
            Ts = np.unique(np.array(new_custom_line_TP_grid)[:,0])
            atm.custom_diffTs[species] = len(Ts)
            
            Ps = np.unique(np.array(new_custom_line_TP_grid)[:,1])
            atm.custom_diffPs[species] = len(Ps)
    return atm
            
new_atm = apply_PT_cutoff(copy.deepcopy(atm), T_min, T_max)

def get_flux(atm):
    p = np.logspace(-5, 2, 30)
    atm.setup_opa_structure(p)

    vmr = dict(zip(line_species, [1e-3]*len(line_species)))
    mmw = 2.3 * np.ones_like(p)
    mf = {k: vmr[k] * mmw for k in line_species}
    mf['MMW'] = mmw

    mf['H2'] = 0.74 * np.ones_like(p)
    mf['He'] = 0.24 * np.ones_like(p)
    mf['H'] = 0.02 * np.ones_like(p)

    log_g = 4.0
    t = np.linspace(1000.0, 2800.0, len(p))
    
    start = time.time()

    atm.calc_flux(t, mf, 10.0**log_g, mmw=mf['MMW'])
    flux = atm.flux
    wave = 2.998e10 / atm.freq
    wave *= 1e7 # [cm] -> nm
    end = time.time()
    print(f' time = {end-start:.2e} s')
    print(f' flux.shape = {flux.shape}')
    # display memory in GB of atm object
    get_memory(atm)
    
    return wave, flux

def get_memory(obj, min_size_mb=1.0):
    memory_dict = {k:asizeof.asizeof(v) for k,v in obj.__dict__.items() if not k.startswith('__')}
    # sort by memory usage
    memory_dict = dict(sorted(memory_dict.items(), key=lambda item: item[1], reverse=True))
    print('** Memory usage **')
    for k, v in memory_dict.items():
        v_MB = v / (1024 ** 2)
        if v_MB < min_size_mb:
            continue
        print(f'{k}: {v_MB:.2f} MB')

wave, flux = get_flux(atm)

print()
new_wave, new_flux = get_flux(new_atm)

mad = np.mean(np.abs(flux - new_flux))
print(f' MAD = {mad:.2e}')
assert mad < 1e-8, f' MAD = {mad:.2e}'

plot = False
if plot:
    fig, ax = plt.subplots(1,1, figsize=(14,4))

    ax.plot(wave, flux, 'k')
    ax.plot(new_wave, new_flux, 'r')

    ax.set_xlabel('Wavelength / nm')
    plt.show()