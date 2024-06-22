import argparse
import pathlib
import subprocess as sp

from retrieval_base.retrieval import pre_processing, prior_check, Retrieval
from retrieval_base.spectrum_jwst import SpectrumJWST
from retrieval_base.pRT_model import pRT_model
import retrieval_base.auxiliary_functions as af
from retrieval_base.parameters import Parameters
from retrieval_base.config import Config
import config_jwst_J as conf

config_file = 'config_jwst.txt'
target = 'TWA28'
run = None
# run = 'jwst_KLM_N10_veiling3'

conf_data = conf.config_data['NIRSpec']
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
    path = pathlib.Path('/home/dgonzalezpi/retrieval_base/')
else:
    path = pathlib.Path('/home/dario/phd/retrieval_base')
    

# Instantiate the parser
parser = argparse.ArgumentParser()
parser.add_argument('--pre_processing', '-p', action='store_true', default=False)
parser.add_argument('--prior_check', '-c', action='store_true', default=False)
parser.add_argument('--retrieval', '-r', action='store_true', default=False)
parser.add_argument('--evaluation', '-e', action='store_true', default=False)
args = parser.parse_args()

if args.pre_processing:
    sp.call(['python', f'{path}/{target}/config_jwst.py'])
    ## Pre-processing data
    
    grisms = [
            'g140h-f100lp', 
            # 'g235h-f170lp', 
            # 'g395h-f290lp',
            ]
    files = [f'jwst/TWA28_{g}.fits' for g in grisms]

    spec = SpectrumJWST(Nedge=40).load_grisms(files)
    spec.reshape(spec.n_orders, 1)
    # spec.fix_wave_nans() # experimental...
    spec.sigma_clip_reshaped(use_flux=False, 
                                # sigma=3, # KM bands
                                sigma=1.0, # J band
                                width=31, 
                                max_iter=5,
                                fun='median', 
                                debug=False)
    spec.plot_orders(fig_name=f'{conf.prefix}plots/spec_to_fit.pdf')
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
        print(f'   --> Saved {pRT_file}')

if args.prior_check:
    print('--> Running prior predictive check..')
    figs_path = pathlib.Path(f'{conf.prefix}plots/')
    figs_path.mkdir(parents=True, exist_ok=True)
    
    random = True
    random_label = '_random' if random else ''
    prior_check(conf=conf, n=5, 
                random=random, 
                get_contr=True,
                fig_name=figs_path / f'prior_predictive_check{random_label}.pdf')

if args.retrieval:
    ret = Retrieval(
        conf=conf, 
        evaluation=args.evaluation
        )
    ret.PMN_run()

if args.evaluation:
    if run is not None: # override config file
        conf = Config(path=path, target=target, run=run)
        conf(config_file)        
    
    ret = Retrieval(
        conf=conf, 
        evaluation=args.evaluation
        )
    ret.plot_ccf = False
    
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
