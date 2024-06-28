import argparse
import pathlib
# import subprocess as sp
import os
import time 
from memory_profiler import profile

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
parser.add_argument('--memory_profiler', '-m', action='store_true', default=False)
args = parser.parse_args()

cwd = str(pathlib.Path(__file__).parent.absolute())
if 'dgonzalezpi' in cwd:
    print('Running on Snellius.. disabling interactive plotting')
    import matplotlib
    # disable interactive plotting
    matplotlib.use('Agg')
    # path = pathlib.Path('/home/dgonzalezpi/retrieval_base/')

path_suffix = 'dario/phd' if 'dario' in os.environ['HOME'] else 'dgonzalezpi'
base_path = pathlib.Path(f'/home/{path_suffix}')    
target = 'TWA28'

run = 'with_spitzer_22'
sed_file = base_path / f'retrieval_base/{target}/SED_runs/{run}/sed.pkl'

if args.prior_check:
    grisms = [
                # 'g140h-f100lp', 
                'g235h-f170lp', 
                'g395h-f290lp',
                ]
    # @profile
    def main():
        
        if sed_file.exists():
            sed = pickle_load(sed_file)
        else:
            Nedge = 40
            start = time.time()
            sed = SED(grisms, run=run).load_spec(Nedge=Nedge)
            
            # wmin = 0.95 * np.nanmin(sed.spec.wave) # [nm]
            wmin = 2600.0 # [nm]
            wmax = 12.0 * 1e3 # [um] -> [nm]
            
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
            sed.spec.apply_error_scaling(default=100.0)
            
            sed.load_spitzer(wmax=wmax*1e-3, sigma_clip=3.0, sigma_width=5)

            parallax_mas = 16.88 # Gaia DR3
            d_pc = 1e3 / parallax_mas # ~ 59.17 pc
            # d_pc = 59.17 

            free_params = {
                            'teff': (2100, 2900),
                        'logg': (3.0, 4.5), 
                        'R_p': (2.2, 3.5), 
                            'T_d': (50, 900),
                            # 'R_d': (1, 1000),
                            'log_R_d':(0, 4),
                            # 'T_d':(100,105),
                            # 'R_d':(0.01, 0.02),
                            # 'a_j': (0.60, 1.20),
                            # 'a_j':(0.1,0.2),
                            # 'a_h': (0.60, 1.20),
                            # 'a_hk': (0.60, 1.20),
                            }
            
            constant_params = {
                'd_pc': d_pc, 
                'resolution': 2700., 
                # 'teff': 2382, # Cooper+24, Gaia DR3
                }
            sed.set_params(free_params, constant_params)
            sed.set_PMN_hyperparameters(
            n_live_points=200,
            evidence_tolerance=0.5,
            n_iter_before_update=300,
            sampling_efficiency=0.05,
            )
            sed.save(sed_file)
            
        sed.prior_check(n=2, random=False, 
                        # inset_xlim=(25e3, 30e3),
                        xscale='log',
                        yscale='log',
                        )
        
        
        
    main()
    
if args.memory_profiler:
    sed = pickle_load(sed_file)
    sed.list_memory_allocation()
    
if args.retrieval:
    sed = pickle_load(sed_file)
    sed.PMN_run()
if args.evaluation:
    sed = pickle_load(sed_file)
    sed.update_path(path=base_path)
    sed.PMN_eval()