"""LaTeX equations for the paper"""

import numpy as np
from retrieval_base.config import Config
import retrieval_base.auxiliary_functions as af
import pathlib

path = af.get_path(return_pathlib=True)
path_eq = pathlib.Path('/home/dario/phd/twa2x_paper/equations')
config_file = 'config_jwst.txt'
w_set='NIRSpec'

# load best fit parameters of MultiNest

runs = dict(
    TWA27A='freeslab_lbl10_G1G2G3_0',
    TWA28='freeslab_lbl10_G1G2G3_0',
    )

targets = list(runs.keys())

target = 'TWA28'
run = runs[target]
posterior_list = []
for target in targets:
    run = runs[target]
    posterior = af.load_posterior(target, run)
    posterior_list.append(posterior)

# blackbody parameters
def get_blackbody_params(posterior):
    q = [0.16, 0.5, 0.84]
    keys = ['T_d', 'R_d']
    
    for key in keys:
        if key not in posterior:
            log_key = f'log_{key}'
            assert log_key in posterior
            posterior[key] = 10.0**posterior[log_key]
            
    T_d = np.quantile(posterior['T_d'], q)
    R_d = np.quantile(posterior['R_d'], q)
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












