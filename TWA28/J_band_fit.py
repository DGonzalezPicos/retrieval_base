import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os


from retrieval_base.retrieval import pre_processing, prior_check, Retrieval
from retrieval_base.spectrum_jwst import SpectrumJWST
from retrieval_base.pRT_model import pRT_model
import retrieval_base.auxiliary_functions as af
from retrieval_base.parameters import Parameters
from retrieval_base.config import Config
import config_jwst_J as conf


config_file = 'config_jwst.txt'
target = 'TWA28'
run = 'J_1'

cwd = os.getcwd()
if target not in cwd:
    print(f'Changing directory to {target}')
    os.chdir(target)

conf_data = conf.config_data['NIRSpec']
for key in ['data', 'plots', 'output']:
    pathlib.Path(f'{conf.prefix}{key}/').mkdir(parents=True, exist_ok=True)
gratings = [
            'g140h-f100lp', 
            # 'g235h-f170lp', 
            # 'g395h-f290lp',
            ]
files = [f'jwst/TWA28_{g}.fits' for g in gratings]
spec = SpectrumJWST(Nedge=40).load_gratings(files)
spec.reshape(spec.n_orders, 1)
# spec.fix_wave_nans() # experimental...
spec.sigma_clip_reshaped(use_flux=False, 
                            sigma=1.0, 
                            width=31, 
                            max_iter=5,
                            fun='median', 
                            debug=True)
plt.show()
spec.plot_orders(fig_name=f'{conf.prefix}plots/spec_to_fit.pdf')

## Create pRT_atm object
pRT_file =pathlib.Path(f'{conf.prefix}data/pRT_atm_{spec.w_set}.pkl')
if not pRT_file.exists():
    print(f'--> Creating {pRT_file}')

    pRT_atm = pRT_model(
        line_species=conf.line_species, 
        d_spec=spec, 
        mode='lbl', 
        lbl_opacity_sampling=conf_data['lbl_opacity_sampling'], 
        cloud_species=conf.cloud_species, 
        # rayleigh_species=['H2', 'He'], 
        # continuum_opacities=['H2-H2', 'H2-He'], 
        rayleigh_species=conf.rayleigh_species,
        continuum_opacities=conf.continuum_opacities,
        log_P_range=conf_data.get('log_P_range'), 
        n_atm_layers=conf_data.get('n_atm_layers'), 
        rv_range=conf.free_params['rv'][0], 
        )
    # check parent directory
    # pRT_file.parent.mkdir(parents=True, exist_ok=True)
    af.pickle_save(pRT_file, pRT_atm)
    print(f'   --> Saved {pRT_file}')