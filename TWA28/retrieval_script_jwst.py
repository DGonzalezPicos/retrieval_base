import argparse
import pathlib

from retrieval_base.retrieval import pre_processing, prior_check, Retrieval
from retrieval_base.spectrum_jwst import SpectrumJWST
from retrieval_base.pRT_model import pRT_model
import retrieval_base.auxiliary_functions as af
from retrieval_base.parameters import Parameters

import config_jwst as conf

conf_data = conf.config_data['G395H_F290LP']
# create output directory

for key in ['data', 'plots']:
    pathlib.Path(f'{conf.prefix}{key}/').mkdir(parents=True, exist_ok=True)
    # print(f'--> Created {conf.prefix}{key}/')
    
    
cwd = str(pathlib.Path(__file__).parent.absolute())
if 'dgonzalezpi' in cwd:
    print('Running on Snellius.. disabling interactive plotting')
    import matplotlib
    # disable interactive plotting
    matplotlib.use('Agg')
    # path = pathlib.Path('/home/dgonzalezpi/retfish/')

# Instantiate the parser
parser = argparse.ArgumentParser()
parser.add_argument('--pre_processing', '-p', action='store_true', default=False)
parser.add_argument('--prior_check', '-c', action='store_true', default=False)
parser.add_argument('--retrieval', '-r', action='store_true', default=False)
parser.add_argument('--evaluation', '-e', action='store_true', default=False)
args = parser.parse_args()

if args.pre_processing:
    
    ## Pre-processing data
    spec = SpectrumJWST(file='jwst/TWA28_g395h-f290lp.fits')
    spec.split_grism(4155., keep=1)
    # spec.sigma_clip(sigma=3, width=5, max_iter=5, fun='median')
    spec.sigma_clip(spec.err, sigma=3, width=50, max_iter=5, fun='median')
    spec.reshape(1,1)
    spec.prepare_for_covariance()

    af.pickle_save(f'{conf.prefix}data/d_spec_{spec.w_set}.pkl', spec)
    print(f'--> Saved {f"{conf.prefix}data/d_spec_{spec.w_set}.pkl"}')


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

if args.prior_check:
    print('--> Running prior predictive check..')
    figs_path = pathlib.Path(f'{conf.prefix}plots/')
    figs_path.mkdir(parents=True, exist_ok=True)
    
    prior_check(conf=conf, fig_name=figs_path / 'prior_predictive_check.pdf')

if args.retrieval:
    ret = Retrieval(
        conf=conf, 
        evaluation=args.evaluation
        )
    ret.PMN_run()

if args.evaluation:
    ret = Retrieval(
        conf=conf, 
        evaluation=args.evaluation
        )
    ret.PMN_callback_func(
        n_samples=None, 
        n_live=None, 
        n_params=None, 
        live_points=None, 
        posterior=None, 
        stats=None,
        max_ln_L=None, 
        ln_Z=None, 
        ln_Z_err=None, 
        nullcontext=None
        )
