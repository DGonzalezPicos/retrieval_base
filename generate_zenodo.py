#!/usr/bin/env python3
"""
Generate Zenodo datasets for publication: "Disentangling disc and atmospheric signatures 
of young brown dwarfs in JWST/NIRSpec spectra"

Authors: D. González Picos, S. de Regt, S. Gandhi, N. Grasser, and I.A.G. Snellen
Institution: Leiden Observatory, Leiden University, P.O. Box 9513, 2300 RA, Leiden, The Netherlands
Contact: picos@strw.leidenuniv.nl

This script extracts and packages the key data products from the retrieval analysis:
1. Observational spectra and best-fit models
2. Pressure-temperature profiles with uncertainty envelopes
3. Posterior distributions for all free parameters
4. Metadata and documentation

Usage:
    python generate_zenodo.py [--targets TWA27A,TWA28] [--no-cache]
"""

import os
import sys
import argparse
import pathlib
import numpy as np
import h5py
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings

# Add retrieval_base to path
sys.path.append('retrieval_base')

from retrieval_base.retrieval import Retrieval
import retrieval_base.auxiliary_functions as af
from retrieval_base.config import Config

# Publication metadata
PUBLICATION_INFO = {
    "title": "Disentangling disc and atmospheric signatures of young brown dwarfs in JWST/NIRSpec spectra",
    "authors": ["D. González Picos", "S. de Regt", "S. Gandhi", "N. Grasser", "I.A.G. Snellen"],
    "institution": "Leiden Observatory, Leiden University, P.O. Box 9513, 2300 RA, Leiden, The Netherlands",
    "contact": "picos@strw.leidenuniv.nl",
    "year": 2025,
    "journal": "Astronomy & Astrophysics",
    "description": "Atmospheric retrieval analysis of TWA 27A and TWA 28 brown dwarfs using JWST/NIRSpec data"
}

# Default configuration
DEFAULT_CONFIG = {
    "targets": ["TWA27A", "TWA28"],
    "runs": {
        "TWA27A": "freeslab_lbl10_G1G2G3_1",
        "TWA28": "freeslab_lbl10_G1G2G3_1"
    },
    "config_file": "config_jwst.txt",
    "w_set": "NIRSpec",
    "output_dir": "zenodo_twa"
}

def setup_paths():
    """Setup and validate paths"""
    path = pathlib.Path(af.get_path())
    zenodo_dir = path / DEFAULT_CONFIG["output_dir"]
    zenodo_dir.mkdir(exist_ok=True)
    return path, zenodo_dir

def create_metadata(target: str, run: str, data_type: str, **kwargs) -> Dict:
    """Create standardized metadata for all data files"""
    metadata = {
        "publication": PUBLICATION_INFO,
        "data_info": {
            "target": target,
            "run": run,
            "data_type": data_type,
            "created": datetime.now().isoformat(),
            "instrument": "JWST/NIRSpec",
            "wavelength_range": "0.97-5.27 μm",
            "gratings": ["G140H", "G235H", "G395H"]
        }
    }
    
    # Add any additional metadata
    metadata["data_info"].update(kwargs)
    
    return metadata

def check_directory(target: str, path: pathlib.Path) -> None:
    """Change to target directory if not already there"""
    cwd = os.getcwd()
    if target not in cwd:
        os.chdir(path / target)
        print(f'Changed directory to {target}')

def extract_spectral_data(path: pathlib.Path, target: str, run: str, 
                         zenodo_dir: pathlib.Path, cache: bool = True) -> None:
    """Extract observational spectra and best-fit models organized by grating"""
    print(f"\n=== Extracting spectral data for {target} ===")
    
    check_directory(target, path)
    
    # Load configuration
    conf = Config(path=path, target=target, run=run)(DEFAULT_CONFIG["config_file"])
    
    # Load spectral data
    print("Loading spectral data...")
    m_spec = af.pickle_load(f'{conf.prefix}data/bestfit_m_spec_NIRSpec.pkl')
    d_spec = af.pickle_load(f'{conf.prefix}data/d_spec_NIRSpec.pkl')
    cov = af.pickle_load(f'{conf.prefix}data/bestfit_Cov_NIRSpec.pkl')
    
    # Process data
    m_spec.flux = m_spec.flux.squeeze()
    m_spec.wave = d_spec.wave
    m_spec.flux_bb = m_spec.blackbody_disk(**m_spec.blackbody_disk_args).squeeze()
    
    # Calculate uncertainties
    err = np.array([cov[cov_i,0].get_err(mask=d_spec.mask_isfinite[cov_i]) 
                   for cov_i in range(len(cov))])
    d_spec.err = err
    d_spec.squeeze()
    
    # Apply flux unit factor
    flux_factor = conf.config_data['NIRSpec'].get('flux_unit_factor', 1.0)
    
    # Define grating wavelength ranges (in nm)
    grating_ranges = {
        'g140h': (900, 1900),
        'g235h': (1650, 3180), 
        'g395h': (2890, 5290)
    }
    
    # Function to flatten and combine orders for a grating
    def flatten_grating_data(wave_data, flux_data, mask_data, wave_range):
        """Flatten orders within a grating's wavelength range"""
        all_wave = []
        all_flux = []
        all_mask = []
        
        for order_idx in range(len(wave_data)):
            # Handle different data shapes - squeeze to get 1D arrays
            wave_order = wave_data[order_idx].squeeze()
            flux_order = flux_data[order_idx].squeeze()
            mask_order = mask_data[order_idx].squeeze()
            
            # Find points within grating range
            in_range = (wave_order >= wave_range[0]) & (wave_order <= wave_range[1])
            
            if np.any(in_range):
                all_wave.extend(wave_order[in_range])
                all_flux.extend(flux_order[in_range])
                all_mask.extend(mask_order[in_range])
        
        if all_wave:
            # Sort by wavelength
            sort_idx = np.argsort(all_wave)
            return (np.array(all_wave)[sort_idx], 
                   np.array(all_flux)[sort_idx], 
                   np.array(all_mask)[sort_idx])
        else:
            return np.array([]), np.array([]), np.array([])
    
    # Create output file
    output_file = zenodo_dir / f"{target}_spectral_data.h5"
    
    with h5py.File(output_file, 'w') as f:
        # Add metadata
        metadata = create_metadata(target, run, "spectral_data", 
                                 flux_unit_factor=flux_factor,
                                 n_orders=len(d_spec.wave),
                                 grating_organization=True)
        
        # Save metadata as JSON string
        f.attrs['metadata'] = json.dumps(metadata, indent=2)
        
        # Process each grating
        for grating, wave_range in grating_ranges.items():
            print(f"Processing grating {grating.upper()}...")
            
            # Flatten observational data
            obs_wave, obs_flux, obs_mask = flatten_grating_data(
                d_spec.wave, d_spec.flux, d_spec.mask_isfinite, wave_range)
            
            # Flatten model data
            model_wave, model_flux, model_mask = flatten_grating_data(
                m_spec.wave, m_spec.flux, d_spec.mask_isfinite, wave_range)
            
            # Flatten blackbody data
            _, bb_flux, _ = flatten_grating_data(
                m_spec.wave, m_spec.flux_bb, d_spec.mask_isfinite, wave_range)
            
            # Flatten error data
            _, obs_err, _ = flatten_grating_data(
                d_spec.wave, d_spec.err, d_spec.mask_isfinite, wave_range)
            
            if len(obs_wave) > 0:
                # Create grating group
                grating_grp = f.create_group(grating)
                
                # Observational data
                obs_grp = grating_grp.create_group('observational_data')
                obs_grp.create_dataset('wavelength', data=obs_wave, compression='gzip')
                obs_grp.create_dataset('flux', data=obs_flux, compression='gzip')
                obs_grp.create_dataset('flux_error', data=obs_err, compression='gzip')
                obs_grp.create_dataset('mask_isfinite', data=obs_mask, compression='gzip')
                
                # Model data
                model_grp = grating_grp.create_group('model_data')
                model_grp.create_dataset('wavelength', data=model_wave, compression='gzip')
                model_grp.create_dataset('flux_total', data=model_flux, compression='gzip')
                model_grp.create_dataset('flux_blackbody', data=bb_flux, compression='gzip')
                
                # Add units and descriptions
                obs_grp['wavelength'].attrs['units'] = 'nm'
                obs_grp['wavelength'].attrs['description'] = f'Wavelength grid for {grating.upper()}'
                obs_grp['flux'].attrs['units'] = 'erg s^-1 cm^-2 nm^-1'
                obs_grp['flux'].attrs['description'] = f'Observed flux for {grating.upper()}'
                obs_grp['flux_error'].attrs['units'] = 'erg s^-1 cm^-2 nm^-1'
                obs_grp['flux_error'].attrs['description'] = f'Flux uncertainties for {grating.upper()}'
                
                model_grp['wavelength'].attrs['units'] = 'nm'
                model_grp['wavelength'].attrs['description'] = f'Wavelength grid for {grating.upper()}'
                model_grp['flux_total'].attrs['units'] = 'erg s^-1 cm^-2 nm^-1'
                model_grp['flux_total'].attrs['description'] = f'Best-fit model flux for {grating.upper()}'
                model_grp['flux_blackbody'].attrs['units'] = 'erg s^-1 cm^-2 nm^-1'
                model_grp['flux_blackbody'].attrs['description'] = f'Blackbody disk component for {grating.upper()}'
                
                # Add grating-specific metadata
                grating_grp.attrs['wavelength_range_nm'] = wave_range
                grating_grp.attrs['n_points'] = len(obs_wave)
                
                print(f"  ✓ {grating.upper()}: {len(obs_wave)} data points, {wave_range[0]}-{wave_range[1]} nm")
        
        # Save additional information
        info_grp = f.create_group('info')
        info_grp.attrs['flux_unit_factor'] = flux_factor
        info_grp.attrs['n_orders'] = len(d_spec.wave)
        info_grp.attrs['grating_ranges'] = json.dumps(grating_ranges)
        
    print(f"✓ Spectral data saved to {output_file}")

def extract_pt_profiles(path: pathlib.Path, target: str, run: str, 
                       zenodo_dir: pathlib.Path, cache: bool = True) -> None:
    """Extract pressure-temperature profiles with uncertainty envelopes"""
    print(f"\n=== Extracting PT profiles for {target} ===")
    
    check_directory(target, path)
    
    # Check for cached data
    envelopes_dir = path / target / f'retrieval_outputs/{run}/test_data/envelopes'
    pt_envelopes_file = envelopes_dir / 'PT_envelopes.npy'
    logg_posterior_file = path / target / f'retrieval_outputs/{run}/test_data/log_g_posterior.npy'
    
    if cache and pt_envelopes_file.exists() and logg_posterior_file.exists():
        print("Loading cached PT data...")
        pt_envelopes_data = np.load(pt_envelopes_file)
        pressure = pt_envelopes_data[0]
        temperature_envelopes = pt_envelopes_data[1:-1]
        integrated_contribution = pt_envelopes_data[-1]
        logg_posterior = np.load(logg_posterior_file)
    else:
        print("Calculating PT envelopes...")
        conf = Config(path=path, target=target, run=run)(DEFAULT_CONFIG["config_file"])
        
        ret = Retrieval(conf=conf, evaluation=False)
        bestfit_params, posterior = ret.PMN_analyze()
        
        ret.evaluate_model(bestfit_params)
        ret.evaluation = True
        ret.PMN_lnL_func()
        ret.get_PT_mf_envelopes(posterior)
        
        # Extract data
        pressure = ret.PT.pressure
        temperature_envelopes = ret.PT.temperature_envelopes
        ret.copy_integrated_contribution_emission()
        integrated_contribution = ret.PT.int_contr_em['NIRSpec']
        
        logg_posterior = posterior[:, list(ret.Param.param_keys).index('log_g')]
        
        # Save cache
        envelopes_dir.mkdir(parents=True, exist_ok=True)
        np.save(pt_envelopes_file, np.vstack([pressure, temperature_envelopes, integrated_contribution]))
        np.save(logg_posterior_file, logg_posterior)
    
    # Create output file
    output_file = zenodo_dir / f"{target}_pt_profiles.h5"
    
    with h5py.File(output_file, 'w') as f:
        # Add metadata
        metadata = create_metadata(target, run, "pt_profiles",
                                 n_pressure_points=len(pressure),
                                 n_envelopes=temperature_envelopes.shape[0],
                                 logg_median=float(np.median(logg_posterior)))
        
        f.attrs['metadata'] = json.dumps(metadata, indent=2)
        
        # Save PT data
        pt_grp = f.create_group('pt_profiles')
        pt_grp.create_dataset('pressure', data=pressure, compression='gzip')
        pt_grp.create_dataset('temperature_envelopes', data=temperature_envelopes, compression='gzip')
        pt_grp.create_dataset('integrated_contribution', data=integrated_contribution, compression='gzip')
        
        # Add units and descriptions
        pt_grp['pressure'].attrs['units'] = 'bar'
        pt_grp['pressure'].attrs['description'] = 'Pressure grid'
        pt_grp['temperature_envelopes'].attrs['units'] = 'K'
        pt_grp['temperature_envelopes'].attrs['description'] = 'Temperature envelopes: 16%, 84%, 2.5%, 97.5%, 0.15%, 99.85%, median'
        pt_grp['integrated_contribution'].attrs['units'] = 'dimensionless'
        pt_grp['integrated_contribution'].attrs['description'] = 'Integrated contribution to emission'
        
        # Add summary statistics
        stats_grp = f.create_group('statistics')
        stats_grp.attrs['logg_median'] = float(np.median(logg_posterior))
        stats_grp.attrs['n_samples'] = len(logg_posterior)
        
    print(f"✓ PT profiles saved to {output_file}")

def extract_posteriors(path: pathlib.Path, target: str, run: str, 
                      zenodo_dir: pathlib.Path, cache: bool = True) -> None:
    """Extract posterior distributions for all free parameters"""
    print(f"\n=== Extracting posteriors for {target} ===")
    
    check_directory(target, path)
    
    # Load configuration
    conf = Config(path=path, target=target, run=run)(DEFAULT_CONFIG["config_file"])
    
    # Check for cached posteriors
    posterior_file = f'{conf.prefix}data/posteriors.h5'
    
    if cache and os.path.exists(posterior_file):
        print("Loading cached posterior data...")
        with h5py.File(posterior_file, 'r') as f:
            param_keys = [k.decode('utf-8') for k in f['param_keys'][:]]
            posterior_dict = {}
            for param in param_keys:
                posterior_dict[param] = f[param][:]
    else:
        print("Calculating posteriors...")
        ret = Retrieval(conf=conf, evaluation=False)
        _, posterior = ret.PMN_analyze()
        
        # Create posterior dictionary
        posterior_dict = {}
        param_keys = ret.Param.param_keys
        for i, param in enumerate(param_keys):
            posterior_dict[param] = posterior[:, i]
        
        # Save cache
        with h5py.File(posterior_file, 'w') as f:
            f.create_dataset('param_keys', data=[k.encode('utf-8') for k in param_keys])
            for param, samples in posterior_dict.items():
                f.create_dataset(param, data=samples)
    
    # Create output file
    output_file = zenodo_dir / f"{target}_posteriors.h5"
    
    with h5py.File(output_file, 'w') as f:
        # Add metadata
        metadata = create_metadata(target, run, "posteriors",
                                 n_parameters=len(param_keys),
                                 n_samples=len(posterior_dict[param_keys[0]]))
        
        f.attrs['metadata'] = json.dumps(metadata, indent=2)
        
        # Save parameter information
        params_grp = f.create_group('parameters')
        params_grp.create_dataset('param_keys', data=[k.encode('utf-8') for k in param_keys])
        params_grp.attrs['n_parameters'] = len(param_keys)
        
        # Save posterior samples
        posterior_grp = f.create_group('posterior_samples')
        for param, samples in posterior_dict.items():
            posterior_grp.create_dataset(param, data=samples, compression='gzip')
            posterior_grp[param].attrs['description'] = f'Posterior samples for {param}'
        
        # Add summary statistics
        stats_grp = f.create_group('statistics')
        for param, samples in posterior_dict.items():
            param_stats = stats_grp.create_group(param)
            param_stats.attrs['median'] = float(np.median(samples))
            param_stats.attrs['std'] = float(np.std(samples))
            param_stats.attrs['percentile_16'] = float(np.percentile(samples, 16))
            param_stats.attrs['percentile_84'] = float(np.percentile(samples, 84))
            param_stats.attrs['n_samples'] = len(samples)
    
    print(f"✓ Posteriors saved to {output_file}")

def create_usage_examples(zenodo_dir: pathlib.Path) -> None:
    """Create minimal Python scripts demonstrating data usage"""
    print(f"\n=== Creating usage examples ===")
    
    # Create examples and figures directories
    examples_dir = zenodo_dir / "examples"
    figures_dir = zenodo_dir / "figures"
    examples_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    
    # Create spectral data example
    spectral_example = examples_dir / "load_spectral_data.py"
    with open(spectral_example, 'w') as f:
        f.write('''#!/usr/bin/env python3
"""
Example script to load and plot spectral data from Zenodo dataset

Usage:
    python load_spectral_data.py
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import json

def load_spectral_data(filename):
    """Load spectral data from HDF5 file"""
    with h5py.File(filename, 'r') as f:
        # Load metadata
        metadata = json.loads(f.attrs['metadata'])
        print(f"Target: {metadata['data_info']['target']}")
        print(f"Run: {metadata['data_info']['run']}")
        
        # Load observational data
        obs_wave = f['observational_data/wavelength'][:]
        obs_flux = f['observational_data/flux'][:]
        obs_err = f['observational_data/flux_error'][:]
        
        # Load model data
        model_wave = f['model_data/wavelength'][:]
        model_flux = f['model_data/flux_total'][:]
        model_bb = f['model_data/flux_blackbody'][:]
        
        return {
            'metadata': metadata,
            'obs_wave': obs_wave,
            'obs_flux': obs_flux,
            'obs_err': obs_err,
            'model_wave': model_wave,
            'model_flux': model_flux,
            'model_bb': model_bb
        }

def plot_spectrum(data, order=0):
    """Plot observed and model spectra for a given order"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot spectra
    ax1.plot(data['obs_wave'][order], data['obs_flux'][order], 'k-', alpha=0.7, label='Observed')
    ax1.plot(data['model_wave'][order], data['model_flux'][order], 'r-', alpha=0.8, label='Model')
    ax1.plot(data['model_wave'][order], data['model_bb'][order], 'b--', alpha=0.6, label='Blackbody disk')
    
    ax1.set_ylabel('Flux (erg s⁻¹ cm⁻² nm⁻¹)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot residuals
    residuals = (data['obs_flux'][order] - data['model_flux'][order]) / data['obs_flux'][order]
    ax2.plot(data['obs_wave'][order], residuals, 'g-', alpha=0.7)
    ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('Relative residuals')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    target = "TWA28"  # Change to TWA27A for the other target
    filename = f"{target}_spectral_data.h5"
    
    print(f"Loading spectral data for {target}...")
    data = load_spectral_data(filename)
    
    print(f"Data shape: {data['obs_flux'].shape}")
    print(f"Wavelength range: {np.min(data['obs_wave']):.1f} - {np.max(data['obs_wave']):.1f} nm")
    
    # Plot first order
    plot_spectrum(data, order=0)
''')
    
    # Create PT profiles example
    pt_example = examples_dir / "load_pt_profiles.py"
    with open(pt_example, 'w') as f:
        f.write('''#!/usr/bin/env python3
"""
Example script to load and plot pressure-temperature profiles from Zenodo dataset

Usage:
    python load_pt_profiles.py
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import json

def load_pt_profiles(filename):
    """Load PT profile data from HDF5 file"""
    with h5py.File(filename, 'r') as f:
        # Load metadata
        metadata = json.loads(f.attrs['metadata'])
        print(f"Target: {metadata['data_info']['target']}")
        print(f"Run: {metadata['data_info']['run']}")
        
        # Load PT data
        pressure = f['pt_profiles/pressure'][:]
        temperature_envelopes = f['pt_profiles/temperature_envelopes'][:]
        integrated_contribution = f['pt_profiles/integrated_contribution'][:]
        
        # Load surface gravity posterior
        logg_posterior = f['surface_gravity/posterior_samples'][:]
        
        return {
            'metadata': metadata,
            'pressure': pressure,
            'temperature_envelopes': temperature_envelopes,
            'integrated_contribution': integrated_contribution,
            'logg_posterior': logg_posterior
        }

def plot_pt_profile(data):
    """Plot pressure-temperature profile with uncertainty envelopes"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    pressure = data['pressure']
    temp_envelopes = data['temperature_envelopes']
    
    # Plot temperature envelopes
    # Envelope order: 16%, 84%, 2.5%, 97.5%, 0.15%, 99.85%, median
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    alphas = [0.3, 0.2, 0.1]
    
    for i in range(3):
        ax1.fill_betweenx(pressure, temp_envelopes[i], temp_envelopes[-(i+1)], 
                         color=colors[i], alpha=alphas[i], 
                         label=f'{[68, 95, 99.7][i]}% confidence')
    
    # Plot median
    ax1.plot(temp_envelopes[3], pressure, 'k-', linewidth=2, label='Median')
    
    ax1.set_xlabel('Temperature (K)')
    ax1.set_ylabel('Pressure (bar)')
    ax1.set_yscale('log')
    ax1.invert_yaxis()
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Pressure-Temperature Profile')
    
    # Plot contribution function
    ax2.plot(data['integrated_contribution'], pressure, 'r-', linewidth=2)
    ax2.set_xlabel('Integrated Contribution Function')
    ax2.set_ylabel('Pressure (bar)')
    ax2.set_yscale('log')
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Contribution Function')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    target = "TWA28"  # Change to TWA27A for the other target
    filename = f"{target}_pt_profiles.h5"
    
    print(f"Loading PT profiles for {target}...")
    data = load_pt_profiles(filename)
    
    print(f"Pressure range: {np.min(data['pressure']):.2e} - {np.max(data['pressure']):.2e} bar")
    print(f"Temperature range: {np.min(data['temperature_envelopes']):.1f} - {np.max(data['temperature_envelopes']):.1f} K")
    print(f"Log g posterior: {np.median(data['logg_posterior']):.2f} ± {np.std(data['logg_posterior']):.2f}")
    
    # Plot PT profile
    plot_pt_profile(data)
''')
    
    # Create posteriors example
    posterior_example = examples_dir / "load_posteriors.py"
    with open(posterior_example, 'w') as f:
        f.write('''#!/usr/bin/env python3
"""
Example script to load and analyze posterior distributions from Zenodo dataset

Usage:
    python load_posteriors.py
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import json

def load_posteriors(filename):
    """Load posterior data from HDF5 file"""
    with h5py.File(filename, 'r') as f:
        # Load metadata
        metadata = json.loads(f.attrs['metadata'])
        print(f"Target: {metadata['data_info']['target']}")
        print(f"Run: {metadata['data_info']['run']}")
        
        # Load parameter names
        param_keys = [k.decode('utf-8') for k in f['parameters/param_keys'][:]]
        
        # Load posterior samples
        posterior_dict = {}
        for param in param_keys:
            posterior_dict[param] = f[f'posterior_samples/{param}'][:]
        
        return {
            'metadata': metadata,
            'param_keys': param_keys,
            'posterior_dict': posterior_dict
        }

def plot_corner_subset(data, params_to_plot=None):
    """Plot corner plot for a subset of parameters"""
    if params_to_plot is None:
        # Default parameters to plot
        params_to_plot = ['R_p', 'T_0', 'log_g', 'log_12CO/13CO']
    
    # Filter available parameters
    available_params = [p for p in params_to_plot if p in data['param_keys']]
    n_params = len(available_params)
    
    fig, axes = plt.subplots(n_params, n_params, figsize=(12, 12))
    
    for i, param_y in enumerate(available_params):
        for j, param_x in enumerate(available_params):
            ax = axes[i, j] if n_params > 1 else axes
            
            if i == j:
                # Diagonal: histograms
                samples = data['posterior_dict'][param_y]
                ax.hist(samples, bins=50, alpha=0.7, density=True)
                ax.set_ylabel('Density')
                if i == n_params - 1:
                    ax.set_xlabel(param_x)
                    
            elif i > j:
                # Lower triangle: scatter plots
                x_samples = data['posterior_dict'][param_x]
                y_samples = data['posterior_dict'][param_y]
                ax.scatter(x_samples[::10], y_samples[::10], alpha=0.5, s=1)
                
                if i == n_params - 1:
                    ax.set_xlabel(param_x)
                if j == 0:
                    ax.set_ylabel(param_y)
            else:
                # Upper triangle: hide
                ax.set_visible(False)
    
    plt.tight_layout()
    plt.show()

def print_parameter_summary(data):
    """Print summary statistics for all parameters"""
    print("\\nParameter Summary:")
    print("-" * 60)
    print(f"{'Parameter':<20} {'Median':<12} {'16%':<10} {'84%':<10}")
    print("-" * 60)
    
    for param in data['param_keys']:
        samples = data['posterior_dict'][param]
        median = np.median(samples)
        p16 = np.percentile(samples, 16)
        p84 = np.percentile(samples, 84)
        
        print(f"{param:<20} {median:<12.3f} {p16:<10.3f} {p84:<10.3f}")

if __name__ == "__main__":
    # Example usage
    target = "TWA28"  # Change to TWA27A for the other target
    filename = f"{target}_posteriors.h5"
    
    print(f"Loading posteriors for {target}...")
    data = load_posteriors(filename)
    
    print(f"Number of parameters: {len(data['param_keys'])}")
    print(f"Number of samples: {len(data['posterior_dict'][data['param_keys'][0]])}")
    
    # Print parameter summary
    print_parameter_summary(data)
    
    # Plot corner plot for key parameters
    plot_corner_subset(data)
''')
    
    print(f"✓ Usage examples created in {examples_dir}")

def create_readme(zenodo_dir: pathlib.Path) -> None:
    """Create comprehensive README file"""
    print(f"\n=== Creating README ===")
    
    readme_file = zenodo_dir / "README.md"
    with open(readme_file, 'w') as f:
        f.write(f'''# Zenodo Dataset: {PUBLICATION_INFO["title"]}

**Authors:** {", ".join(PUBLICATION_INFO["authors"])}  
**Institution:** {PUBLICATION_INFO["institution"]}  
**Contact:** {PUBLICATION_INFO["contact"]}  
**Year:** {PUBLICATION_INFO["year"]}

## Description

This dataset contains the key data products from the atmospheric retrieval analysis of TWA 27A and TWA 28 brown dwarfs using JWST/NIRSpec observations. The data supports the results presented in our paper on disentangling disk and atmospheric signatures in young brown dwarf spectra.

## Data Products

### 1. Spectral Data (`*_spectral_data.h5`)
- **Observational data**: Wavelength, flux, and uncertainties for each spectral order
- **Best-fit models**: Total model flux and blackbody disk component
- **Metadata**: Instrument configuration, flux units, and processing information

### 2. Pressure-Temperature Profiles (`*_pt_profiles.h5`)
- **Temperature envelopes**: Confidence intervals (68%, 95%, 99.7%) and median profiles
- **Pressure grid**: Atmospheric pressure levels
- **Contribution function**: Integrated emission contribution
- **Surface gravity**: Posterior samples for log g

### 3. Posterior Distributions (`*_posteriors.h5`)
- **Parameter samples**: MCMC chains for all free parameters
- **Statistics**: Median, standard deviation, and percentiles
- **Metadata**: Parameter descriptions and units

## File Structure

```
zenodo_twa/
├── README.md                     # This file
├── TWA27A_spectral_data.h5      # TWA 27A spectral data
├── TWA27A_pt_profiles.h5        # TWA 27A PT profiles
├── TWA27A_posteriors.h5         # TWA 27A posteriors
├── TWA28_spectral_data.h5       # TWA 28 spectral data
├── TWA28_pt_profiles.h5         # TWA 28 PT profiles
├── TWA28_posteriors.h5          # TWA 28 posteriors
└── examples/                     # Usage examples
    ├── load_spectral_data.py
    ├── load_pt_profiles.py
    └── load_posteriors.py
```

## Usage

### Requirements
- Python 3.7+
- numpy
- h5py
- matplotlib
- json

### Quick Start

1. **Load spectral data:**
```python
import h5py
import numpy as np

with h5py.File('TWA28_spectral_data.h5', 'r') as f:
    wavelength = f['observational_data/wavelength'][:]
    flux = f['observational_data/flux'][:]
    model_flux = f['model_data/flux_total'][:]
```

2. **Load PT profiles:**
```python
with h5py.File('TWA28_pt_profiles.h5', 'r') as f:
    pressure = f['pt_profiles/pressure'][:]
    temperature = f['pt_profiles/temperature_envelopes'][:]
    contribution = f['pt_profiles/integrated_contribution'][:]
```

3. **Load posteriors:**
```python
with h5py.File('TWA28_posteriors.h5', 'r') as f:
    param_keys = [k.decode('utf-8') for k in f['parameters/param_keys'][:]]
    log_g_samples = f['posterior_samples/log_g'][:]
```

### Examples

See the `examples/` directory for complete working examples:
- `load_spectral_data.py`: Load and plot observed vs. model spectra
- `load_pt_profiles.py`: Load and plot pressure-temperature profiles
- `load_posteriors.py`: Load and analyze posterior distributions

## Data Format

All data files use HDF5 format with the following structure:

### Metadata
Each file contains a `metadata` attribute with:
- Publication information
- Target and run identifiers
- Data creation timestamp
- Instrument configuration
- Data-specific parameters

### Data Groups
- **observational_data/**: Raw observational data
- **model_data/**: Best-fit model results
- **pt_profiles/**: Pressure-temperature information
- **posterior_samples/**: MCMC parameter samples
- **statistics/**: Summary statistics

### Units
- Wavelength: nm
- Flux: erg s⁻¹ cm⁻² nm⁻¹
- Pressure: bar
- Temperature: K
- Surface gravity: log₁₀(cm s⁻²)

## Targets

### TWA 27A
- **Spectral type:** M9
- **Distance:** ~55 pc
- **Age:** ~10 Myr
- **Gratings:** G140H, G235H, G395H
- **Wavelength range:** 0.97-5.27 μm

### TWA 28
- **Spectral type:** M8.5
- **Distance:** ~55 pc
- **Age:** ~10 Myr
- **Gratings:** G140H, G235H, G395H
- **Wavelength range:** 0.97-5.27 μm

## Retrieval Method

The atmospheric retrieval analysis used:
- **Code:** retrieval_base (custom Python package)
- **Sampler:** PyMultiNest
- **Atmospheric model:** petitRADTRANS
- **Chemistry:** Free chemistry with slab disk model
- **Wavelength coverage:** 0.97-5.27 μm across three gratings

## Citation

If you use this dataset, please cite our paper:

```bibtex
@article{{gonzalez_picos_2024,
    title = {{{PUBLICATION_INFO["title"]}}},
    author = {{{" and ".join(PUBLICATION_INFO["authors"])}}},
    journal = {{{PUBLICATION_INFO["journal"]}}},
    year = {{{PUBLICATION_INFO["year"]}}},
    institution = {{{PUBLICATION_INFO["institution"]}}}
}}
```

## Contact

For questions about this dataset, please contact:
- **Primary contact:** {PUBLICATION_INFO["contact"]}
- **Institution:** {PUBLICATION_INFO["institution"]}

## License

This dataset is released under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
You are free to use, modify, and distribute this data with proper attribution.

## Acknowledgments

This work is based on observations made with the NASA/ESA/CSA James Webb Space Telescope, 
which is operated by the Space Telescope Science Institute (STScI) under NASA contract NAS5-03127.

We thank the JWST team for their excellent work in commissioning and operating the observatory.
''')
    
    print(f"✓ README created: {readme_file}")

def main():
    """Main function to generate all Zenodo datasets"""
    parser = argparse.ArgumentParser(description='Generate Zenodo datasets for publication')
    parser.add_argument('--targets', type=str, default='TWA27A,TWA28',
                       help='Comma-separated list of targets (default: TWA27A,TWA28)')
    parser.add_argument('--no-cache', action='store_true',
                       help='Regenerate all data without using cache')
    
    args = parser.parse_args()
    
    # Parse targets
    targets = [t.strip() for t in args.targets.split(',')]
    cache = not args.no_cache
    
    print(f"=== Generating Zenodo datasets ===")
    print(f"Targets: {targets}")
    print(f"Using cache: {cache}")
    
    # Setup paths
    path, zenodo_dir = setup_paths()
    
    # Process each target
    for target in targets:
        if target not in DEFAULT_CONFIG["runs"]:
            print(f"Warning: No run defined for target {target}, skipping...")
            continue
        
        run = DEFAULT_CONFIG["runs"][target]
        
        try:
            # Extract all data types
            extract_spectral_data(path, target, run, zenodo_dir, cache)
            extract_pt_profiles(path, target, run, zenodo_dir, cache)
            extract_posteriors(path, target, run, zenodo_dir, cache)
            
        except Exception as e:
            print(f"Error processing {target}: {e}")
            continue
    
    # Create usage examples and documentation
    create_usage_examples(zenodo_dir)
    create_readme(zenodo_dir)
    
    print(f"\n=== Dataset generation complete ===")
    print(f"Output directory: {zenodo_dir}")
    print(f"Files created:")
    for file in sorted(zenodo_dir.glob("*.h5")):
        print(f"  - {file.name}")
    print(f"  - README.md")
    print(f"  - examples/ (3 files)")

if __name__ == "__main__":
    main() 