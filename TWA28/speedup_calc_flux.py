import numpy as np
import pathlib
import os
import json
import time

import seaborn as sns
# set palette from sns for matplotlib
sns.set_palette('deep')


import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from retrieval_base.spectrum_jwst import SpectrumJWST
from retrieval_base.pRT_model import pRT_model
import retrieval_base.auxiliary_functions as af

target = 'TWA28'
cwd = os.getcwd()
if target not in cwd:
    os.chdir(f'/home/dario/phd/retrieval_base/{target}')
    print(f'Changed directory to {os.getcwd()}')


gratings = [
        # 'g140h-f100lp', 
        'g235h-f170lp', 
        # 'g395h-f290lp',
        ]
assert len(gratings) == 1, f'Only single-grating implemented, got {len(gratings)}'

files = [f'jwst/TWA28_{g}.fits' for g in gratings]
spec = SpectrumJWST(Nedge=40).load_gratings(files)
spec.reshape(spec.n_orders, 1)



opacity_params = {
    'log_12CO': ([(-14,-2), r'$\log\ \mathrm{^{12}CO}$'], 'CO_high_Sam'),
    'log_13CO': ([(-14,-2), r'$\log\ \mathrm{^{13}CO}$'], 'CO_36_high_Sam'),

    'log_H2O': ([(-14,-2), r'$\log\ \mathrm{H_2O}$'], 'H2O_pokazatel_main_iso'),
    'log_Na': ([(-14,-2), r'$\log\ \mathrm{Na}$'], 'Na_allard_high'),
    # 'log_K': ([(-14,-2), r'$\log\ \mathrm{K}$'], 'K_high'),
    'log_Ca': ([(-14,-2), r'$\log\ \mathrm{Ca}$'], 'Ca_high'),
    'log_VO': ([(-14,-2), r'$\log\ \mathrm{VO}$'], 'VO_HyVO_main_iso'), # DGP (2024-07-16): 3.4 um bump?
    'log_TiO': ([(-14,-2), r'$\log\ \mathrm{TiO}$'], 'TiO_48_Exomol_McKemmish'),
    'log_SiO': ([(-14,-2), r'$\log\ \mathrm{SiO}$'], 'SiO_SiOUVenIR_main_iso'),
    'log_C2H2': ([(-14,-2), r'$\log\ \mathrm{C_2H_2}$'], 'C2H2_main_iso'),
}

line_species = [v[-1] for k,v in opacity_params.items()]

cache = True
lbl = 12

def load_pRT_atm(grating, lbl, cache=True):
    pRT_file = f'./lbl_test/pRT_atm_{grating}_lbl{lbl}.pkl'
    if not pathlib.Path(pRT_file).exists() or not cache:
        print(f'{pRT_file} not found... generating')
        # return None
        pRT_atm = pRT_model(
            line_species=line_species, 
            d_spec=spec, 
            mode='lbl', 
            lbl_opacity_sampling=lbl,
            rayleigh_species=['H2', 'He'], 
            continuum_opacities=['H2-H2', 'H2-He'], 
            log_P_range=log_P_range,
            n_atm_layers=n_atm_layers,
            rv_range=rv_range,
            T_cutoff=(1200.0, 3200.0),
            P_cutoff=(1e-4, 1e1),
            )
        # pickle save
        af.pickle_save(pRT_file, pRT_atm)
    else: 
        pRT_atm = af.pickle_load(pRT_file)
        print(f'Loaded {pRT_file}')
    return pRT_atm

    

bestfit_run = 'lbl15_G2_4'
chem = af.pickle_load(f'retrieval_outputs/{bestfit_run}/test_data/bestfit_Chem.pkl')
PT  = af.pickle_load(f'retrieval_outputs/{bestfit_run}/test_data/bestfit_PT.pkl')
bestfit = json.load(open(f'retrieval_outputs/{bestfit_run}/test_data/bestfit.json'))
bestfit['params']['gratings'] =[gratings[0][:5] for _ in range(2)]
print(f' bestfit params gratings ', bestfit['params']['gratings'])
wave = np.squeeze(spec.wave)

log_P_range = (-5,2)
n_atm_layers = len(bestfit['temperature'])
rv_range = (-20,20.)

# fig, ax = 
path = af.get_path(return_pathlib=True)
pdf_name = path / f'TWA28/lbl_test/speed_calcflux_{gratings[0]}lbl_{lbl}.pdf'

flux_list = []
    
speeds = ['norm', 'fast']
for i, speed in enumerate(speeds):
    # pRT_file = f'./lbl_test/pRT_atm_{gratings[0]}_lbl{lbl}.pkl'

    cache_i = True 
    if not cache and i ==0:
        cache_i = False
    pRT_atm = load_pRT_atm(gratings[0], lbl, cache=cache_i)
    
    for i in range(3):
        start = time.time()
        m_spec = pRT_atm(mass_fractions=chem.mass_fractions,
                        temperature=PT.temperature,
                        params=bestfit['params'],
                        calc_flux_fast= (speed == 'fast'),
                        )
        end = time.time()

        print(f'{i+1}/2 --> Speed={speed}, time={1000*(end-start):.1f} ms\n')
    # print(m_spec.flux.shape)
    flux_list.append(np.atleast_2d(np.squeeze(m_spec.flux)))
        
lw = 1.0
wave = np.atleast_2d(wave)

with PdfPages(pdf_name) as pdf:
    
    for i in range(len(wave)):
        
        fig, ax = plt.subplots(2,1, figsize=(12, 6), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        
        for j in range(2):
            ax[0].plot(wave[i], flux_list[j][i,], label='Fast' if speeds[j]=='fast' else 'Norm',
                       alpha=0.9-0.1*j, lw=lw)
            
            if j > 0:
                res = flux_list[j][i,] - flux_list[0][i,]
                # make relative residuals
                # res /= flux_list[0][i,]
                res = np.divide(res, flux_list[0][i,], out=np.zeros_like(res), where=flux_list[0][i,]!=0)
                res *= 100
                ax[1].plot(wave[i], res, color=ax[0].get_lines()[-1].get_color(),
                            alpha=1.0-0.1*j, lw=lw)
            
            
        ax[0].legend()
        ax[1].axhline(0, color='k', ls='-', lw=0.7)
        ax[1].set(xlabel='Wavelength / nm', ylabel='Rel. Residuals / %')
        ax[0].set(ylabel='Flux', xlim=(np.nanmin(wave[i]), np.nanmax(wave[i])))
        # make ylim residuals ax[1] symmetric
        ylim = ax[1].get_ylim()
        ax[1].set_ylim(-np.nanmax(np.abs(ylim)), np.nanmax(np.abs(ylim)))
        pdf.savefig(fig)
        plt.close(fig)
        
    print(f'--> Saved {pdf_name}')