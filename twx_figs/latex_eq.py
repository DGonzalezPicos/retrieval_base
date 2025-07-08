"""LaTeX equations for the paper"""

import numpy as np
import os
from retrieval_base.config import Config
from retrieval_base.retrieval import Retrieval
import retrieval_base.auxiliary_functions as af
import pathlib
import h5py

path = af.get_path(return_pathlib=True)
path_eq = pathlib.Path('/home/dario/phd/twa2x_paper/equations')
config_file = 'config_jwst.txt'
w_set='NIRSpec'

def save_posterior_h5(posterior_dict: dict, param_keys: list, filename: str):
    """Save posterior samples to HDF5 file for fast loading"""
    with h5py.File(filename, 'w') as f:
        # Save parameter names
        f.create_dataset('param_keys', data=[k.encode('utf-8') for k in param_keys])
        
        # Save posterior samples for each parameter
        for param, samples in posterior_dict.items():
            f.create_dataset(param, data=samples)
        
        print(f'Saved posterior to {filename}')

def load_posterior_h5(filename: str) -> tuple[dict, list]:
    """Load posterior samples from HDF5 file"""
    posterior_dict = {}
    
    with h5py.File(filename, 'r') as f:
        # Load parameter names
        param_keys = [k.decode('utf-8') for k in f['param_keys'][:]]
        
        # Load posterior samples
        for param in param_keys:
            posterior_dict[param] = f[param][:]
    
    print(f'Loaded posterior from {filename}')
    return posterior_dict, param_keys

def transform_parameters(posterior_dict):
    """Transform log parameters to linear scale where needed"""
    transformed_dict = posterior_dict.copy()
    
    # Convert log_R_d to R_d by taking 10^log_R_d
    if 'log_R_d' in posterior_dict:
        transformed_dict['R_d'] = 10**posterior_dict['log_R_d']
        # Remove the log version to avoid confusion
        if 'log_R_d' in transformed_dict:
            del transformed_dict['log_R_d']
            
    if 'log_R_jup' in posterior_dict:
        transformed_dict['R_jup'] = 10**posterior_dict['log_R_jup']
        # Remove the log version to avoid confusion
        if 'log_R_jup' in transformed_dict:
            del transformed_dict['log_R_jup']
            
    if 'log_T_ex' in posterior_dict:
        transformed_dict['T_ex'] = 10**posterior_dict['log_T_ex']
        # Remove the log version to avoid confusion
        if 'log_T_ex' in transformed_dict:
            del transformed_dict['log_T_ex']
        
    return transformed_dict

def load_data(path, target, run, cache=True):
    """Load posterior data from H5 cache or generate from retrieval"""
    cwd = os.getcwd()
    if target not in cwd:
        os.chdir(f'{path}/{target}')
        print(f'Changed directory to {target}')
    
    config_file = 'config_jwst.txt'    
    conf = Config(path=path, target=target, run=run)(config_file)
    
    posterior_file = f'{conf.prefix}data/posteriors.h5'
    
    if not cache or not os.path.exists(posterior_file):
        ret = Retrieval(conf=conf, evaluation=False)
        _, posterior = ret.PMN_analyze()
        
        # Create posterior dictionary
        posterior_dict = {}
        for i, param in enumerate(ret.Param.param_keys):
            posterior_dict[param] = posterior[:, i]
        
        # Save to HDF5 file
        save_posterior_h5(posterior_dict, ret.Param.param_keys, posterior_file)
    else:
        # Load from HDF5 file
        posterior_dict, param_keys = load_posterior_h5(posterior_file)
    
    # Transform parameters (e.g., log_R_d to R_d)
    posterior_dict = transform_parameters(posterior_dict)
    
    return posterior_dict

# Load best fit parameters of MultiNest using H5 cache
runs = dict(
    TWA27A='freeslab_lbl10_G1G2G3_1',
    TWA28='freeslab_lbl10_G1G2G3_1',
    )

targets = list(runs.keys())

# Load posteriors using H5 caching
posterior_list = []
for target in targets:
    run = runs[target]
    print(f'\nLoading data for {target}, run: {run}')
    posterior_dict = load_data(path, target, run, cache=True)
    posterior_list.append(posterior_dict)

# blackbody parameters
def get_blackbody_params(posterior_dict):
    """Extract blackbody parameters from posterior dictionary"""
    q = [0.16, 0.5, 0.84]
    
    # Parameters should already be transformed by transform_parameters()
    # but check if transformation is needed
    if 'T_d' not in posterior_dict and 'log_T_d' in posterior_dict:
        T_d_samples = 10.0**posterior_dict['log_T_d']
    else:
        T_d_samples = posterior_dict['T_d']
        
    if 'R_d' not in posterior_dict and 'log_R_d' in posterior_dict:
        R_d_samples = 10.0**posterior_dict['log_R_d']
    else:
        R_d_samples = posterior_dict['R_d']
    
    T_d = af.quantiles(T_d_samples, q=q)
    R_d = af.quantiles(R_d_samples, q=q)
    return T_d, R_d

def blackbody_eq(T_d, R_d):
    """write latex equation for blackbody with measurements and uncertainties"""
    eq = f'${T_d[0]:.2f}^{{+{T_d[2]-T_d[0]:.2f}}}_{{-{T_d[0]-T_d[1]:.2f}}}$ K, ${R_d[0]:.2f}^{{+{R_d[2]-R_d[0]:.2f}}}_{{-{R_d[0]-R_d[1]:.2f}}}$ R$_\\odot$'
    return eq

def blackbody_eq_str(T_d_list, R_d_list, eq_path=None):
    """Write LaTeX equation for blackbody parameters with measurements and uncertainties
    
    Parameters
    ----------
    T_d_list : list of arrays
        List of temperature arrays, each containing [median, lower, upper] quantiles
    R_d_list : list of arrays
        List of radius arrays, each containing [median, lower, upper] quantiles
    eq_path : pathlib.Path, optional
        Path to save the equation
        
    Returns
    -------
    str
        LaTeX formatted equation
    """
    eq = '\\begin{align*}\n'
    names = ['TWA 28', 'TWA 27A']
    
    for i, name in enumerate(names):
        if i > 0:
            eq += '\\\\\n'  # Line break between objects
            
        T_med, T_low, T_up = T_d_list[i]
        R_med, R_low, R_up = R_d_list[i]
        
        # Convert solar radii to Jupiter radii (R_☉ ≈ 9.955 R_Jup)
        R_conv = 1.0  # Assuming already in Jupiter radii
        R_med *= R_conv
        R_low *= R_conv
        R_up *= R_conv
        # use proper text format instead of math mode for name of target
        eq += f'\\text{{{name}}}: '
        eq += f'T_{{\mathrm{{d}}}} = {T_med:.0f}'
        eq += f'^{{+{T_up-T_med:.0f}}}'
        eq += f'_{{-{abs(T_med-T_low):.0f}}}'
        eq += f'\,{{\mathrm{{K}}}},\\,'
        
        eq += f'R_{{\mathrm{{d}}}} = {R_med:.1f}'
        eq += f'^{{+{R_up-R_med:.1f}}}'
        eq += f'_{{-{abs(R_med-R_low):.1f}}}'
        eq += f'\,{{\mathrm{{R}}}}_{{\\!\\mathrm{{Jup}}}}'
        
    eq += '\n\\end{align*}'
    
    if eq_path is not None:
        eq_path.parent.mkdir(parents=True, exist_ok=True)
        with open(eq_path, 'w') as f:
            f.write(eq)
        print(f'Saved equation to {eq_path}')
    
    return eq

T_d_list, R_d_list = zip(*[get_blackbody_params(posterior) for posterior in posterior_list])


print(blackbody_eq_str(T_d_list, R_d_list, eq_path=path_eq/'blackbody_eq.tex'))












