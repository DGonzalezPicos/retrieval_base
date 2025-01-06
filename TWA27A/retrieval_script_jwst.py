import argparse
import pathlib
import subprocess as sp

import shutil
import os

from retrieval_base.retrieval import prior_check, Retrieval
from retrieval_base.spectrum_jwst import SpectrumJWST
from retrieval_base.pRT_model import pRT_model
import retrieval_base.auxiliary_functions as af
from retrieval_base.parameters import Parameters
from retrieval_base.config import Config
import config_jwst as conf

config_file = 'config_jwst.txt'
target = 'TWA27A'

conf_data = conf.config_data['NIRSpec']
# create output directory

for key in ['data', 'plots', 'output']:
    pathlib.Path(f'{conf.prefix}{key}/').mkdir(parents=True, exist_ok=True)
    # print(f'--> Created {conf.prefix}{key}/')
    
    
path = af.get_path(return_pathlib=True)

# Instantiate the parser
parser = argparse.ArgumentParser()
parser.add_argument('--pre_processing', '-p', action='store_true', default=False)
parser.add_argument('--prior_check', '-c', action='store_true', default=False)
parser.add_argument('--retrieval', '-r', action='store_true', default=False)
parser.add_argument('--evaluation', '-e', action='store_true', default=False)
# parser.add_argument('--time_profiler', '-t', action='store_true', default=False)
parser.add_argument('--memory_profiler', '-m', action='store_true', default=False)
parser.add_argument('--copy_to_snellius', '-copy_to_snellius', action='store_true', default=False)
parser.add_argument('--download', '-d', action='store_true', default=False)
parser.add_argument('--run', '-run', type=str, default=None)
parser.add_argument('--ccf', '-ccf', action='store_true', default=False)
args = parser.parse_args()
run = args.run
if args.pre_processing:
    sp.call(['python', f'{path}/{target}/config_jwst.py'])
    ## Pre-processing data
    
    # gratings = [
    #         # 'g140h-f100lp', 
    #         'g235h-f170lp', 
    #         'g395h-f290lp',
    #         ]
    gratings_dict = {'g140h': 'g140h-f100lp',
                     'g235h': 'g235h-f170lp', 
                     'g395h': 'g395h-f290lp'}
    
    gratings_list = list(set(conf.constant_params['gratings']))
    gratings = [gratings_dict[g] for g in gratings_list]
    
    # each grating has two filters, make list [a,b] to [a,a,b,b]
    # gratings_list = [g.split('-')[0] for g in gratings for _ in range(2)]
    # print(f'--> Loading data for {gratings_list}')
    
    files = [f'jwst/{target}_{g}.fits' for g in gratings]
    Nedge = conf_data.get('Nedge', 40)
    spec = SpectrumJWST(Nedge=Nedge).load_gratings(files)
    print(f' Orders: {spec.n_orders}')
    spec.reshape(spec.n_orders*2, 1)
    # spec.fix_wave_nans() # experimental...
    sigma_clip_width = conf_data.get('sigma_clip_width', 30)
    if len(conf.mask_lines)>0:
        spec.mask_lines(conf.mask_lines)
        
    for i in range(2):
        spec.sigma_clip_reshaped(use_flux=False, 
                                    # sigma=3, # KM bands
                                    sigma=conf_data.get('sigma_clip', 2),
                                    width=sigma_clip_width * (i+1)**2,
                                    max_iter=5,
                                    fun='median', 
                                    fig_name=f'{conf.prefix}plots/sigma_clip_{i}.pdf')
    # spec.scatter_overlapping_points()
    # spec.apply_error_scaling()
    
    spec.plot_orders(fig_name=f'{conf.prefix}plots/spec_to_fit.pdf', grid=True)
    
    if conf.cov_mode == 'GP':
        spec.prepare_for_covariance()
        
    spec.gratings_list = conf.constant_params['gratings']

    af.pickle_save(f'{conf.prefix}data/d_spec_{spec.w_set}.pkl', spec)


    ## Create pRT_atm object
    pRT_file =pathlib.Path(f'{conf.prefix}data/pRT_atm_{spec.w_set}.pkl')
    if 'rv' in conf.free_params:
        rv_range = conf.free_params['rv'][0]
    elif 'rv' in conf.constant_params:
        rv = conf.constant_params['rv']
        rv_range = (rv-10.0, rv+10.0)
    else:
        raise ValueError('No rv_range defined in conf.free_params or conf.constant_params')
        
    if not pRT_file.exists():
        print(f'--> Creating {pRT_file}')
        lbl = conf_data['lbl_opacity_sampling']
        pRT_atm = pRT_model(
            line_species=conf.line_species, 
            d_spec=spec, 
            mode='lbl' if (lbl is not None) else 'c-k',
            lbl_opacity_sampling=lbl,
            cloud_species=conf.cloud_species, 
            # rayleigh_species=['H2', 'He'], 
            # continuum_opacities=['H2-H2', 'H2-He'], 
            rayleigh_species=conf.rayleigh_species,
            continuum_opacities=conf.continuum_opacities,
            log_P_range=conf_data.get('log_P_range'), 
            n_atm_layers=conf_data.get('n_atm_layers'), 
            rv_range=rv_range,
            disk_species=getattr(conf, 'disk_species', []),
            disk_kwargs=getattr(conf, 'disk_kwargs', {}),
            T_ex_range=getattr(conf, 'T_ex_range', None),
            N_mol_range=getattr(conf, 'N_mol_range', None),
            T_cutoff=conf_data.get('T_cutoff', None),
            P_cutoff=conf_data.get('P_cutoff', None),
            species_wave=getattr(conf, 'species_wave', {}),
            )
        # check parent directory
        # pRT_file.parent.mkdir(parents=True, exist_ok=True)
        af.pickle_save(pRT_file, pRT_atm)
        print(f'   --> Saved {pRT_file}')

if args.prior_check:
    print('--> Running prior predictive check..')
    figs_path = pathlib.Path(f'{conf.prefix}plots/')
    figs_path.mkdir(parents=True, exist_ok=True)
    
    random = False
    random_label = '_random' if random else ''
    disk = True
    disk_label = '_disk' if disk else ''
    ret = prior_check(conf=conf, n=3, 
                random=random, 
                get_contr=False,
                remove_disk=not disk,
                species_to_plot=['12CO', 'H2O', 'FeH','TiO','VO', 'C2H2', 'CrH','HCl','HF','Fe','Na','K','Ca','Ti'],
                fig_name=figs_path / f'prior_predictive_check{disk_label}{random_label}.pdf')
    
    if args.memory_profiler:
        print('--> Running memory profiler..')
        ret.list_memory_allocation(min_size_mb=0.1)
        # print(ret.w_set)
        for w_set in ret.d_spec.keys():
            ret.list_memory_allocation(obj=ret.pRT_atm[w_set], min_size_mb=0.1)
        
# if args.minimize:
#     print('--> Running minimization..')
#     ret = Retrieval(conf=conf, evaluation=False)
#     ret.gradient_based_optimization(method='L-BFGS-B', options={'maxiter': 100})

if args.copy_to_snellius:
    # copy this folder to snellius
    snellius_dir = f'/home/dgonzalezpi/retrieval_base/{target}/retrieval_outputs/{conf.run}'
    local_dir = str(path / target / 'retrieval_outputs' / conf.run)
    print(f' Copying {local_dir} to {snellius_dir}...')
    
    # if parent directory does not exist, create it on remote
    # sp.run(f'scp -r {local_dir} dgonzalezpi@snellius.surf.nl:{snellius_dir}', shell=True, check=True)
    # use rync -av --delete instead of scp -r
    rsync_command = f'rsync -av --delete {local_dir}/ dgonzalezpi@snellius.surf.nl:{snellius_dir}/'
    try:
        sp.run(rsync_command, shell=True, check=True)
        
    except:
        proxy_jump = 'ssh.strw.leidenuniv.nl'
       # Define SSH command with all options for skipping confirmation
        ssh_command = f"ssh -J picos@{proxy_jump} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o CheckHostIP=no"
        # Set the RSYNC_RSH environment variable to the SSH command
        os.environ["RSYNC_RSH"] = ssh_command

        # Now run rsync without needing the -e option
        print(f' Attempting to use RSYNC_RSH with command:\n{rsync_command}')
        
        sp.run(rsync_command, shell=True, check=True)
        print(f' Successful download for {target} {run}!\n')

    print(f' Succesful copy for {target}!\n')
    
if args.download:
    # download from snellius using scp -r
    run = conf.run
    snellius_dir = f'/home/dgonzalezpi/retrieval_base/{target}/retrieval_outputs/{run}/test_output'
    local_dir = str(path / target / f'retrieval_outputs/{run}/test_output')
    print(f' Downloading {snellius_dir} to {local_dir}...')
    
    cache = False
    test_output = path / target / f'retrieval_outputs/{run}/test_output'
    if test_output.exists() and not cache:
        shutil.rmtree(test_output)
    
    rsync_command = f'rsync -av --progress dgonzalezpi@snellius.surf.nl:{snellius_dir}/ {local_dir}/'

    try:
        # sp.run(f'scp -r dgonzalezpi@snellius.surf.nl:{snellius_dir} {local_dir}', shell=True, check=True)
        sp.run(rsync_command, shell=True, check=True)
        print(f' Succesful download for {target} {run}!\n')
    except:
        proxy_jump = 'ssh.strw.leidenuniv.nl'
       # Define SSH command with all options for skipping confirmation
        ssh_command = f"ssh -J picos@{proxy_jump} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o CheckHostIP=no"
        # Set the RSYNC_RSH environment variable to the SSH command
        os.environ["RSYNC_RSH"] = ssh_command

        # Now run rsync without needing the -e option
        rsync_command = f'rsync -av --progress dgonzalezpi@snellius.surf.nl:{snellius_dir}/ {local_dir}/'
        print(f' Attempting to use RSYNC_RSH with command:\n{rsync_command}')
        
        sp.run(rsync_command, shell=True, check=True)
        print(f' Successful download for {target} {run}!\n')
        
if args.retrieval:
    ret = Retrieval(
        conf=conf, 
        evaluation=args.evaluation,
        # tmp_path=args.tmp_path
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
    
if args.ccf:
    run = run or conf.run
    print(f' ** Running cross-correlation function for {target} {run}..')
    # command = f'python {path}/retrieval_base/cross_correlation.py -t {target} -r {run}'
    # print(f' ** Running command: {command}')
    sp.call(command, shell=True)
    print(f' ** Done with cross-correlation function for {target} {run}..')