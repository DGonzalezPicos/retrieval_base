import argparse
import pathlib
# import subprocess as sp
import os
import time 

import numpy as np
from sed_fit import SED
from retrieval_base.auxiliary_functions import pickle_load

# Instantiate the parser
parser = argparse.ArgumentParser()
parser.add_argument('--pre_processing', '-p', action='store_true', default=False)
parser.add_argument('--prior_check', '-c', action='store_true', default=False)
parser.add_argument('--retrieval', '-r', action='store_true', default=False)
parser.add_argument('--evaluation', '-e', action='store_true', default=False)
parser.add_argument('--tmp_path', '-t', type=str, default=None)
args = parser.parse_args()

path_suffix = 'dario/phd' if 'dario' in os.environ['HOME'] else 'dgonzalezpi'
base_path = pathlib.Path(f'/home/{path_suffix}')    
target = 'TWA28'

run = 'with_spitzer_9'
sed_file = base_path / f'retrieval_base/{target}/SED_runs/{run}/sed.pkl'

if args.prior_check:
    grisms = [
                # 'g140h-f100lp', 
                'g235h-f170lp', 
                'g395h-f290lp',
                ]

    Nedge = 40
    start = time.time()
    sed = SED(grisms, run=run).load_spec(Nedge=Nedge)
    
    # wmin = 0.95 * np.nanmin(sed.spec.wave) # [nm]
    wmin = 2200.0 # [nm]
    wmax = 32.0 * 1e3 # [um] -> [nm]
    
    bt_model_file = sed.run_path / f'BTSETTL_{wmin:.0f}_{wmax:.0f}.nc'
    sed.init_BTSettl(wmin=wmin,
                    #  wmax=1.05 * np.nanmax(sed.spec.wave)*1e-3,
                    # wmax=33.0, # Spitzer
                    wmax=wmax,
                    # create_grid=args.pre_processing, # RECOMPUTE GRID
                    file=bt_model_file,
                     wave_unit='nm')
    sed.mask_wave(wmin=wmin)

    end = time.time()
    print(f' Time to load data and model: {end - start:.2f} s')


    wave = np.squeeze(sed.spec.wave)
    nans = np.isnan(wave) | np.isnan(sed.spec.flux)
    wave[nans] = np.nan
    sed.resample(wave_step=100)
    # print(stop)
    sed.spec.scatter_overlapping_points(plot=False)
    sed.spec.apply_error_scaling()
    
    sed.load_spitzer(wmax=wmax*1e-3, sigma_clip=3.0, sigma_width=5)

    parallax_mas = 16.88 # Gaia DR3
    d_pc = 1e3 / parallax_mas # ~ 59.17 pc
    # d_pc = 59.17 

    free_params = {
                    'teff': (2200, 2700),
                   'logg': (3.0, 4.5), 
                   'R_p': (2.2, 3.5), 
                    'T_d': (100, 900),
                    'R_d': (1, 100),
                    # 'T_d':(100,105),
                    # 'R_d':(0.01, 0.02),
                    }
    
    constant_params = {
        'd_pc': d_pc, 
        'resolution': 2700., 
        # 'teff': 2382, # Cooper+24, Gaia DR3
        }
    sed.set_params(free_params, constant_params)
    sed.prior_check(n=2, random=False, 
                    # inset_xlim=(25e3, 30e3),
                    xscale='log',
                    yscale='log',
                    )
    sed.save(sed_file)
    
if args.retrieval:
    
    sed = pickle_load(sed_file)
    sed.PMN_run()
    sed.PMN_eval()