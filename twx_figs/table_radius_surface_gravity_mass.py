"""
Latex table with radius, surface gravity, and mass for TWA27A and TWA28
from different retrieval runs.

date: 2025-07-02
"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid display issues

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
import os
import pathlib
from retrieval_base.retrieval import Retrieval
import retrieval_base.auxiliary_functions as af
from retrieval_base.config import Config
import h5py
from scipy.stats import gaussian_kde
import corner
import time
from scipy.ndimage import gaussian_filter
import matplotlib.patheffects as pe

def setup_paths():
    path = af.get_path(return_pathlib=True)
    path_tables = pathlib.Path('/home/dario/phd/twa2x_paper/tables')
    return path, path_tables

def define_runs():
    # only one run per target
    runs = {
        'TWA28': {'freeslab_lbl10_G1G2G3_1': r'0.97-5.30 $\mu$m', # G1+G2+G3
                    'freeslab_lbl10_G2G3_1': r'1.63-5.30 $\mu$m', # G2+G3
                    },
        'TWA27A': {'freeslab_lbl10_G1G2G3_1': r'0.97-5.30 $\mu$m', # G1+G2+G3
                    'freeslab_lbl10_G2G3_2': r'1.63-5.30 $\mu$m', # G2+G3
                    },
    }

    return runs

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

def load_data(path, target, run, cache=True):
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

def compute_quantiles(samples: np.ndarray, q: list = [0.16, 0.5, 0.84]) -> np.ndarray:
    """Compute quantiles for parameter samples"""
    return af.quantiles(samples, q=q)

def format_quantile_title(param_name: str, quantiles_dict: dict, param_labels: dict, colors: dict) -> str:
    """Format parameter title with median and 1-sigma errors for both targets"""
    param_label = param_labels.get(param_name, param_name)
    
    # Start with parameter name
    title_parts = [param_label]
    
    # Add values for each target
    for target in ['TWA27A', 'TWA28']:
        if target in quantiles_dict:
            quantiles = quantiles_dict[target]
            median = quantiles[1]
            lower = quantiles[1] - quantiles[0]
            upper = quantiles[2] - quantiles[1]
            
            # Format numbers appropriately
            if abs(median) > 100 or abs(median) < 0.01:
                value_str = f"{median:.2e}$^{{+{upper:.1e}}}_{{-{lower:.1e}}}$"
            else:
                value_str = f"{median:.2f}$^{{+{upper:.2f}}}_{{-{lower:.2f}}}$"
            
            # Simple text without color coding in LaTeX
            title_parts.append(f"{value_str}")
    
    return '\n'.join(title_parts)

def compute_mass(log_g, R_p, print_mass=False):
    g_cgs = 10**log_g
    r_cm = R_p * 7.1492e9
    mass_cgs = g_cgs * r_cm**2 / 6.67430e-8
    mass_mjup = mass_cgs / 1.898e30
   
    if print_mass:
        print(f' [Parameters.compute_mass]: mass = {mass_mjup:.2e} Mjup')
    return mass_mjup
def transform_parameters(posterior_dict):
    """Transform log parameters to linear scale where needed"""
    transformed_dict = posterior_dict.copy()
    
    # Convert log_R_d to R_d by taking 10^log_R_d
    if 'log_g' and 'R_p' in posterior_dict:
        transformed_dict['mass'] = compute_mass(posterior_dict['log_g'], posterior_dict['R_p'])
        
    return transformed_dict

def manjavacas_results():
    """
    Load Manjavacas et al. 2024 results
    """
    
    twa28 = {
        'ATMO':
            {
                'R_p': (2.90, 0.24),
                'log_g': (4.0, 0.5),
                'mass': (32.0, 22.2, 67.0),
            },
        'BT-Settl':
            {
                'R_p': (2.55, 0.19),
                'log_g': (4.0, 0.5),
                'mass': (26.0, 16.9, 53.5),
            },
    }
    twa27a = {
        'ATMO':
            {
                'R_p': (2.70, 0.21),
                'log_g': (4.0, 0.5),
                'mass': (28.1, -19.8, 61.8),
            },
        'BT-Settl':
            {
                'R_p': (2.70, 0.20),
                'log_g': (4.0, 0.5),
                'mass': (29.3, 19.8, 62.8),
            },
    }
    return twa28, twa27a

def venuti_results():
    """
    Load Venuti et al. 2019 results
    """
    solar_radius_cm = 6.957e10
    jupiter_radius_cm = 7.1492e9
    mass_sun_g = 1.989e33
    mass_jupiter_g = 1.898e30
    twa27a = {
        'R_p': 0.35 * solar_radius_cm / jupiter_radius_cm,
        'log_g': (3.75, 0.14),
        'mass': np.array([0.019, 0.006]) * mass_sun_g / mass_jupiter_g,
    }
    twa28 = {
        'R_p': 0.29 * solar_radius_cm / jupiter_radius_cm,
        'log_g': (4.1, 0.3),
        'mass': np.array([0.020, 0.005]) * mass_sun_g / mass_jupiter_g,
    }
    return twa28, twa27a

def format_value_with_uncertainty(value, uncertainty=None, uncertainty_lower=None, 
                                 uncertainty_upper=None, precision=2, use_scientific=False,
                                 force_symmetric=False):
    """
    Format a value with symmetric or asymmetric uncertainties for LaTeX
    
    Parameters:
    -----------
    value : float
        Central value
    uncertainty : float, optional
        Symmetric uncertainty
    uncertainty_lower : float, optional
        Lower uncertainty (for asymmetric errors)
    uncertainty_upper : float, optional
        Upper uncertainty (for asymmetric errors)
    precision : int
        Number of decimal places
    use_scientific : bool
        Force scientific notation for large/small values
    force_symmetric : bool
        Convert asymmetric uncertainties to symmetric using max
    
    Returns:
    --------
    str : LaTeX formatted string
    """
    if uncertainty is not None:
        # Symmetric uncertainty
        if use_scientific and (abs(value) > 100 or abs(value) < 0.01):
            return f"${value:.1e} \\pm {uncertainty:.1e}$"
        else:
            return f"${value:.{precision}f} \\pm {uncertainty:.{precision}f}$"
    elif uncertainty_lower is not None and uncertainty_upper is not None:
        if force_symmetric:
            # Convert to symmetric using the larger uncertainty
            symmetric_uncertainty = max(uncertainty_lower, uncertainty_upper)
            if use_scientific and (abs(value) > 100 or abs(value) < 0.01):
                return f"${value:.1e} \\pm {symmetric_uncertainty:.1e}$"
            else:
                return f"${value:.{precision}f} \\pm {symmetric_uncertainty:.{precision}f}$"
        else:
            # Asymmetric uncertainty
            if use_scientific and (abs(value) > 100 or abs(value) < 0.01):
                return f"${value:.1e}^{{+{uncertainty_upper:.1e}}}_{{-{uncertainty_lower:.1e}}}$"
            else:
                return f"${value:.{precision}f}^{{+{uncertainty_upper:.{precision}f}}}_{{-{uncertainty_lower:.{precision}f}}}$"
    else:
        # No uncertainty
        if use_scientific and (abs(value) > 100 or abs(value) < 0.01):
            return f"${value:.1e}$"
        else:
            return f"${value:.{precision}f}$"

def generate_latex_table(data_dict: dict, output_path: str = None):
    """
    Generate a publication-ready LaTeX table with radius, surface gravity, and mass
    for TWA 27A and TWA 28 from different studies.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing data from this work
    output_path : str, optional
        Path to save the LaTeX table file
    """
    
    # Get literature data
    twa28_manjavacas, twa27a_manjavacas = manjavacas_results()
    twa28_venuti, twa27a_venuti = venuti_results()
    
    latex_table = []
    latex_table.append("% Physical properties table for TWA 27A and TWA 28")
    latex_table.append("% Generated automatically by table_radius_surface_gravity_mass.py")
    latex_table.append("\\begin{table}")
    latex_table.append("\\centering")
    latex_table.append("\\caption{Physical properties of TWA 27A and TWA 28 from different studies.}")
    latex_table.append("\\label{tab:physical_properties}")
    latex_table.append("\\renewcommand{\\arraystretch}{1.3}")  # Increase row spacing
    latex_table.append("\\begin{tabular}{lccc}")
    latex_table.append("\\hline")
    latex_table.append("Target & Radius & $\\log g$ & Mass \\\\")
    latex_table.append(" & ($R_{\\mathrm{Jup}}$) & (cgs) & ($M_{\\mathrm{Jup}}$) \\\\")
    latex_table.append("\\hline")
    
    # Venuti et al. 2019 block
    latex_table.append("\\multicolumn{4}{c}{\\textit{Venuti et al.} (2019)} \\\\")
    # latex_table.append("\\vspace{0.5em}")  # Add spacing
    latex_table.append("\\hline")
    
    # TWA 27A - Venuti
    r_27a = twa27a_venuti['R_p']
    logg_27a = twa27a_venuti['log_g']
    mass_27a = twa27a_venuti['mass']
    
    r_27a_str = format_value_with_uncertainty(r_27a, precision=2)
    logg_27a_str = format_value_with_uncertainty(logg_27a[0], logg_27a[1], precision=2)
    mass_27a_str = format_value_with_uncertainty(mass_27a[0], mass_27a[1], precision=1)
    
    latex_table.append(f"TWA 27A & {r_27a_str} & {logg_27a_str} & {mass_27a_str} \\\\")
    
    # TWA 28 - Venuti
    r_28 = twa28_venuti['R_p']
    logg_28 = twa28_venuti['log_g']
    mass_28 = twa28_venuti['mass']
    
    r_28_str = format_value_with_uncertainty(r_28, precision=2)
    logg_28_str = format_value_with_uncertainty(logg_28[0], logg_28[1], precision=2)
    mass_28_str = format_value_with_uncertainty(mass_28[0], mass_28[1], precision=1)
    
    latex_table.append(f"TWA 28 & {r_28_str} & {logg_28_str} & {mass_28_str} \\\\")
    latex_table.append("\\hline")
    
    # Manjavacas et al. 2024 block
    latex_table.append("\\multicolumn{4}{c}{\\textit{Manjavacas et al.} (2024)} \\\\")
    # latex_table.append("\\vspace{0.5em}")  # Add spacing
    latex_table.append("\\hline")
    
    # ATMO model subblock
    latex_table.append("\\multicolumn{4}{l}{\\quad ATMO} \\\\")
    for target_name, target_data in [('TWA 27A', twa27a_manjavacas), ('TWA 28', twa28_manjavacas)]:
        data = target_data['ATMO']
        r_str = format_value_with_uncertainty(data['R_p'][0], data['R_p'][1], precision=2)
        logg_str = format_value_with_uncertainty(data['log_g'][0], data['log_g'][1], precision=1)
        
        # Handle asymmetric mass uncertainties (keep asymmetric for mass)
        mass_central = data['mass'][0]
        mass_lower = abs(data['mass'][1])  # Take absolute value for lower uncertainty
        mass_upper = data['mass'][2]
        mass_str = format_value_with_uncertainty(mass_central, 
                                               uncertainty_lower=mass_lower,
                                               uncertainty_upper=mass_upper, 
                                               precision=1)
        
        latex_table.append(f"{target_name} & {r_str} & {logg_str} & {mass_str} \\\\")
    
    # BT-Settl model subblock
    latex_table.append("\\multicolumn{4}{l}{\\quad BT-Settl} \\\\")
    for target_name, target_data in [('TWA 27A', twa27a_manjavacas), ('TWA 28', twa28_manjavacas)]:
        data = target_data['BT-Settl']
        r_str = format_value_with_uncertainty(data['R_p'][0], data['R_p'][1], precision=2)
        logg_str = format_value_with_uncertainty(data['log_g'][0], data['log_g'][1], precision=1)
        
        # Handle asymmetric mass uncertainties (keep asymmetric for mass)
        mass_central = data['mass'][0]
        mass_lower = abs(data['mass'][1])  # Take absolute value for lower uncertainty
        mass_upper = data['mass'][2]
        mass_str = format_value_with_uncertainty(mass_central,
                                               uncertainty_lower=mass_lower,
                                               uncertainty_upper=mass_upper,
                                               precision=1)
        
        latex_table.append(f"{target_name} & {r_str} & {logg_str} & {mass_str} \\\\")
    latex_table.append("\\hline")
    
    # This work block
    latex_table.append("\\multicolumn{4}{c}{This work} \\\\")
    # latex_table.append("\\vspace{0.5em}")  # Add spacing
    latex_table.append("\\hline")
    
    runs = define_runs()
    
    # Group runs by wavelength range
    wavelength_groups = {
        'freeslab_lbl10_G1G2G3_1': r'0.97-5.30 $\mu$m',
        'freeslab_lbl10_G2G3_1': r'1.63-5.30 $\mu$m',
        'freeslab_lbl10_G2G3_2': r'1.63-5.30 $\mu$m'
    }
    
    # Process each wavelength range as a subblock
    for wavelength_range in [r'0.97-5.30 $\mu$m', r'1.63-5.30 $\mu$m']:
        latex_table.append(f"\\multicolumn{{4}}{{l}}{{\\quad {wavelength_range}}} \\\\")
        
        for target in ['TWA27A', 'TWA28']:
            target_runs = runs[target]
            for run_key, run_label in target_runs.items():
                # Only process runs that match current wavelength range
                if wavelength_groups.get(run_key) == wavelength_range:
                    data_key = f"{target}_{run_key}"
                    if data_key in data_dict:
                        quantiles = data_dict[data_key]['quantiles']
                        
                        # Format radius with symmetric uncertainty
                        r_precision = 2
                        r_q = [round(r, r_precision) for r in quantiles['R_p']]
                        r_str = format_value_with_uncertainty(r_q[1], 
                                                            uncertainty_lower=r_q[1]-r_q[0],
                                                            uncertainty_upper=r_q[2]-r_q[1],
                                                            precision=r_precision,
                                                            force_symmetric=True)
                        
                        # Format log g with symmetric uncertainty
                        logg_q = quantiles['log_g']
                        logg_str = format_value_with_uncertainty(logg_q[1],
                                                               uncertainty_lower=logg_q[1]-logg_q[0],
                                                               uncertainty_upper=logg_q[2]-logg_q[1],
                                                               precision=2,
                                                               force_symmetric=True)
                        
                        # Format mass with asymmetric uncertainty (keep original)
                        mass_q = quantiles['mass']
                        mass_str = format_value_with_uncertainty(mass_q[1],
                                                               uncertainty_lower=mass_q[1]-mass_q[0],
                                                               uncertainty_upper=mass_q[2]-mass_q[1],
                                                               precision=1)
                        
                        target_name = target.replace('TWA', 'TWA ')
                        latex_table.append(f"{target_name} & {r_str} & {logg_str} & {mass_str} \\\\")
    
    latex_table.append("\\hline")
    latex_table.append("\\end{tabular}")
    latex_table.append("\\tablefoot{")
    latex_table.append("Radius and surface gravity uncertainties are shown as symmetric (using the larger of the asymmetric uncertainties), ")
    latex_table.append("while mass uncertainties are shown asymmetrically where appropriate. ")
    latex_table.append("ATMO and BT-Settl refer to different atmospheric model grids from \\textit{Manjavacas et al.} (2024). ")
    latex_table.append("Wavelength ranges indicate the spectral coverage: 0.97--5.30~$\\mu$m includes all three ")
    latex_table.append("NIRSpec gratings (G140H, G235H, G395H), while 1.63--5.30~$\\mu$m uses only G235H and G395H. }")
    latex_table.append("\\end{table}")
    
    # Join all lines
    latex_content = '\n'.join(latex_table)
    
    # Print to console
    print("\nLaTeX Table:")
    print("=" * 60)
    print(latex_content)
    print("=" * 60)
    
    # Save to file if path provided
    if output_path:
        with open(output_path, 'w') as f:
            f.write(latex_content)
        print(f"\nTable saved to: {output_path}")
    
    return latex_content

def main():
    path, path_tables = setup_paths()
    runs = define_runs()
    
    # Define the parameters to show in the table (using available parameters)
    # Common parameters available in both datasets
    param_order = ['R_p', 'log_g', 'mass']
        
    # Load data for both targets and runs
    data_dict = {}
    
    for target in runs.keys():
        for run_key, run_label in runs[target].items():
            print(f'\nLoading data for {target}, run: {run_key} ({run_label})')
        
            try:
                posterior_dict = load_data(path, target, run_key, cache=True)
                
                # Filter to only include parameters we want to plot
                filtered_dict = {}
                for param in param_order:
                    if param in posterior_dict:
                        filtered_dict[param] = posterior_dict[param]
                    else:
                        print(f'Warning: Parameter {param} not found for {target}')
                
                # Get quantiles for each parameter
                quantiles = {}
                for param in param_order:
                    if param in filtered_dict:
                        quantiles[param] = compute_quantiles(filtered_dict[param])
                
                # Store with unique key
                data_dict[f"{target}_{run_key}"] = {
                    'data': filtered_dict,
                    'quantiles': quantiles
                }
                
                print(f'Loaded {len(filtered_dict)} parameters for {target} ({run_label})')
                
                # Print pretty table with median and 1-sigma errors
                print(f'{target} ({run_label}):')
                for param in param_order:
                    if param in quantiles:
                        q = quantiles[param]
                        # print(f'{param}: {q[1]:.2f}$^{{+{q[2]-q[1]:.2f}}}_{{-{q[1]-q[0]:.2f}}}$')
                
            except Exception as e:
                print(f'Error loading data for {target} {run_key}: {e}')
                continue
    
    if not data_dict:
        print('No data loaded successfully. Exiting.')
        return
    
    # Generate LaTeX table
    output_file = path_tables / 'table_physical_properties.tex'
    generate_latex_table(data_dict, str(output_file))
    
    print(f'\nLaTeX table generated and saved to: {output_file}')

if __name__ == '__main__':
    main()