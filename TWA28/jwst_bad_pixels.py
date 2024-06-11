import pathlib
import os

cwd = os.getcwd()
if 'TWA28' not in cwd:
    os.chdir('TWA28')
    print(f'--> Changed to {os.getcwd()}')
    
from retrieval_base.retrieval import pre_processing, prior_check, Retrieval
from retrieval_base.spectrum_jwst import SpectrumJWST
import retrieval_base.auxiliary_functions as af
from retrieval_base.parameters import Parameters
from retrieval_base.config import Config
import config_jwst as conf

config_file = 'config_jwst.txt'
target = 'TWA28'
run = None
# run = 'jwst_KLM_N10_veiling3'

conf_data = conf.config_data['NIRSpec']


grisms = [
            # 'g140h-f100lp', 
            'g235h-f170lp', 
            'g395h-f290lp',
            ]
files = [f'jwst/TWA28_{g}.fits' for g in grisms]

spec = SpectrumJWST(Nedge=40).load_grisms(files)
spec.reshape(spec.n_orders, 1)
# spec.fix_wave_nans() # experimental...
spec.sigma_clip_reshaped(use_flux=False, 
                            sigma=3, 
                            width=31, 
                            max_iter=5,
                            fun='median', 
                            debug=True)

# spec.sigma_clip_reshaped(use_flux=False, 
#                             sigma=3, 
#                             width=300, 
#                             max_iter=3,
#                             fun='median', 
#                             debug=True)
spec.plot_orders(fig_name=f'{conf.prefix}plots/spec_to_fit.pdf', lw=0.8)
spec.prepare_for_covariance()