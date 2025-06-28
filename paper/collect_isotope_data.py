#!/usr/bin/env python3
"""
Collect isotope ratio data for carbon and oxygen from all targets in the sample.

This script gathers:
- Isotope ratios (12C/13C and 16O/18O) with uncertainties for all targets
- Stellar parameters (Teff, metallicity) with uncertainties  
- Detection significance (sigma levels)
- GCE model predictions from Romano+2022
- Reference values (Sun, ISM, Crossfield+2019)

Data is saved to an HDF5 file with comprehensive metadata and references.

Author: Dario Gonzalez Picos
Date: December 2024
"""

import numpy as np
import h5py
import os
import pathlib
from datetime import datetime
import warnings

from retrieval_base.retrieval import Retrieval
from retrieval_base.config import Config
from retrieval_base.auxiliary_functions import (
    spirou_sample, read_spirou_sample_csv, find_run, load_romano_models
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def extract_isotope_data(target: str, isotope: str, base_path: str, 
                        main_label: str = 'CO', run: str = None) -> dict:
    """
    Extract isotope ratio data for a single target.
    
    Parameters
    ----------
    target : str
        Target name (e.g., 'gl699')
    isotope : str
        Isotope type ('carbon' or 'oxygen')
    base_path : str
        Base path to retrieval results
    main_label : str
        Molecule label ('CO' or 'H2O')
    run : str
        Specific run to use (if None, uses latest)
        
    Returns
    -------
    dict
        Dictionary containing isotope ratio data and metadata
    """
    
    # Change to target directory
    if target not in os.getcwd():
        os.chdir(base_path + target)
    
    outputs = pathlib.Path(base_path) / target / 'retrieval_outputs'
    
    # Find available runs
    dirs = [d for d in outputs.iterdir() if d.is_dir() and 'fc' in d.name and '_' not in d.name]
    if not dirs:
        print(f'No runs found for {target}')
        return None
        
    runs = [int(d.name.split('fc')[-1]) for d in dirs]
    
    if run is None:
        run = 'fc' + str(max(runs))
    else:
        run = 'fc' + str(run)
        if run not in [d.name for d in dirs]:
            print(f'Run {run} not found for {target}')
            return None
    
    # Check test_output folder
    test_output = outputs / run / 'test_output'
    if not test_output.exists() or len(list(test_output.iterdir())) == 0:
        print(f'No test output found for {target}')
        return None
    
    # Load detection significance (sigma)
    sigma = 10.0  # default for plotting as upper limit
    if main_label == 'CO':
        species_sigma = 'C18O' if isotope == 'oxygen' else '13CO'
        sigma_file = test_output / f'lnB_sigma_{species_sigma}.dat'
        if sigma_file.exists():
            try:
                lnB, sigma = np.loadtxt(sigma_file)
                sigma = 0.0 if np.isnan(sigma) else sigma
                sigma = 100.0 if sigma > 100.0 else sigma
            except:
                sigma = 10.0
    
    # Load or compute isotope posterior
    isotope_posterior_file = base_path + target + '/retrieval_outputs/' + run + f'/{main_label}_{isotope}_isotope_posterior.npy'
    
    if os.path.exists(isotope_posterior_file):
        isotope_posterior = np.load(isotope_posterior_file)
    else:
        # Compute isotope posterior
        try:
            config_file = 'config_freechem.txt'
            conf = Config(path=base_path, target=target, run=run)(config_file)
            
            ret = Retrieval(conf=conf, evaluation=False)
            bestfit_params, posterior = ret.PMN_analyze()
            param_keys = list(ret.Param.param_keys)
            
            if isotope == 'oxygen':
                key = 'log_H2O/H2O_181' if main_label == 'H2O' else 'log_12CO/C18O'
            elif isotope == 'carbon':
                key = 'log_12CO/13CO'
            
            if key not in param_keys:
                print(f'Parameter {key} not found for {target}')
                return None
                
            log_ratio_id = param_keys.index(key)
            isotope_posterior = 10.0**posterior[:, log_ratio_id]
            
            # Save for future use
            np.save(isotope_posterior_file, isotope_posterior)
            
        except Exception as e:
            print(f'Error processing {target}: {e}')
            return None
    
    # Compute quantiles
    q = [0.16, 0.5, 0.84]
    isotope_quantiles = np.quantile(isotope_posterior, q)
    
    return {
        'target': target,
        'isotope': isotope,
        'ratio_median': isotope_quantiles[1],
        'ratio_lower': isotope_quantiles[0],
        'ratio_upper': isotope_quantiles[2],
        'sigma': sigma,
        'run': run,
        'posterior_samples': isotope_posterior
    }


def collect_all_data() -> dict:
    """
    Collect isotope data for all targets in the sample.
    
    Returns
    -------
    dict
        Complete dataset with all targets and metadata
    """
    
    base_path = '/home/dario/phd/retrieval_base/'
    
    # Read sample information
    df = read_spirou_sample_csv()
    names = df['Star'].to_list()
    
    # Extract stellar parameters
    teff = dict(zip(names, [float(t.split('+-')[0]) for t in df['Teff (K)'].to_list()]))
    teff_err = dict(zip(names, [float(t.split('+-')[1]) for t in df['Teff (K)'].to_list()]))
    spt = dict(zip(names, [t.split('+-')[0] for t in df['SpT'].to_list()]))
    
    # Load metallicity data (Cristofari+2023, Table 3)
    metallicity_ref = 'C23'
    table_id = 3
    c23 = np.loadtxt(f'{base_path}paper/data/c23_table{table_id}_mh.txt', dtype=object)
    
    # Add GL 4063 manually from Table 3
    c23 = np.append(c23, [['Gl4063', '0.36', '0.1']], axis=0)
    
    c23_names = ['Gl ' + n[2:] for n in c23[:, 0]]
    metallicity = dict(zip(c23_names, c23[:, 1].astype(float)))
    metallicity_err = dict(zip(c23_names, c23[:, 2].astype(float)))
    
    # Initialize data storage
    data = {
        'targets': [],
        'carbon_data': [],
        'oxygen_data': [],
        'stellar_params': {},
        'metadata': {}
    }
    
    isotopes = ['carbon', 'oxygen']
    ignore_targets = []
    
    print("Collecting isotope data for all targets...")
    
    for name in names:
        target = name.replace('Gl ', 'gl')
        
        if target in ignore_targets:
            print(f'Skipping {target}')
            continue
            
        print(f'Processing {name} ({target})')
        
        # Store stellar parameters
        stellar_data = {
            'name': name,
            'target_id': target,
            'teff': teff.get(name, np.nan),
            'teff_err': teff_err.get(name, np.nan),
            'spectral_type': spt.get(name, ''),
            'metallicity': metallicity.get(name, np.nan),
            'metallicity_err': metallicity_err.get(name, np.nan)
        }
        
        # Extract isotope data
        target_data = {'stellar': stellar_data}
        
        for isotope in isotopes:
            isotope_result = extract_isotope_data(
                target, isotope, base_path, main_label='CO'
            )
            
            if isotope_result is not None:
                target_data[isotope] = isotope_result
                print(f'  {isotope}: {isotope_result["ratio_median"]:.1f} '
                      f'+{isotope_result["ratio_upper"]-isotope_result["ratio_median"]:.1f}'
                      f'-{isotope_result["ratio_median"]-isotope_result["ratio_lower"]:.1f} '
                      f'({isotope_result["sigma"]:.1f}σ)')
            else:
                print(f'  {isotope}: No data')
        
        if 'carbon' in target_data or 'oxygen' in target_data:
            data['targets'].append(target_data)
    
    # Load GCE models
    print("\nLoading GCE models...")
    gce_models = {}
    mass_ranges = ['1_8', '3_8']
    
    for mass_range in mass_ranges:
        Z, c12c13, o16o18 = load_romano_models(Z_min=-3.0, mass_range=mass_range)
        gce_models[mass_range] = {
            'metallicity': Z,
            'carbon_ratio': c12c13,
            'oxygen_ratio': o16o18,
            'description': f'Romano+2022 GCE model for {mass_range.replace("_", "-")} M☉ stellar mass range'
        }
    
    data['gce_models'] = gce_models
    
    # Reference values
    print("Adding reference values...")
    data['reference_values'] = {
        'sun': {
            'carbon': {'ratio': 93.5, 'error': 3.0, 'reference': 'Lodders 2003'},
            'oxygen': {'ratio': 529.7, 'error': 1.7, 'reference': 'McKeegan et al. 2011 (solar wind)'}
        },
        'ism': {
            'carbon': {'ratio': 68.0, 'error': 14.0, 'reference': 'Wilson et al. 1999'},
            'oxygen': {'ratio': 557.0, 'error': 30.0, 'reference': 'Wilson et al. 1999'}
        },
        'crossfield_2019': {
            'Gl 745 A': {
                'carbon': {'ratio': 296, 'error': 45, 'teff': 3454, 'teff_err': 31, 'metallicity': -0.43, 'metallicity_err': 0.05},
                'oxygen': {'ratio': 1220, 'error': 260, 'teff': 3454, 'teff_err': 31, 'metallicity': -0.43, 'metallicity_err': 0.05}
            },
            'Gl 745 B': {
                'carbon': {'ratio': 224, 'error': 26, 'teff': 3440, 'teff_err': 31, 'metallicity': -0.39, 'metallicity_err': 0.05},
                'oxygen': {'ratio': 1550, 'error': 360, 'teff': 3440, 'teff_err': 31, 'metallicity': -0.39, 'metallicity_err': 0.05}
            }
        }
    }
    
    # Metadata
    data['metadata'] = {
        'creation_date': datetime.now().isoformat(),
        'description': 'Carbon and oxygen isotope ratios for M dwarf sample from atmospheric retrieval',
        'isotope_labels': {'carbon': '¹²C/¹³C', 'oxygen': '¹⁶O/¹⁸O'},
        'molecule_source': 'CO lines (not H2O)',
        'metallicity_reference': f'Cristofari+2023 Table {table_id}',
        'sample_description': 'SPIRou M dwarf sample',
        'retrieval_method': 'petitRADTRANS + PyMultiNest',
        'units': {
            'isotope_ratios': 'dimensionless',
            'temperature': 'K',
            'metallicity': '[M/H] dex',
            'sigma': 'detection significance level'
        }
    }
    
    return data


def save_to_hdf5(data: dict, filename: str = 'isotope_data.h5') -> None:
    """
    Save collected data to HDF5 file with comprehensive metadata.
    
    Parameters
    ----------
    data : dict
        Complete dataset
    filename : str
        Output filename
    """
    
    print(f"\nSaving data to {filename}...")
    
    with h5py.File(filename, 'w') as f:
        
        # Create main groups
        targets_grp = f.create_group('targets')
        models_grp = f.create_group('gce_models')
        refs_grp = f.create_group('reference_values')
        
        # Save target data
        n_targets = len(data['targets'])
        
        # Create arrays for tabular data
        target_names = []
        target_ids = []
        teff_vals = []
        teff_errs = []
        spectral_types = []
        metallicity_vals = []
        metallicity_errs = []
        
        carbon_ratios = []
        carbon_lower = []
        carbon_upper = []
        carbon_sigma = []
        carbon_runs = []
        
        oxygen_ratios = []
        oxygen_lower = []
        oxygen_upper = []
        oxygen_sigma = []
        oxygen_runs = []
        
        # Store posterior samples separately
        carbon_posteriors = {}
        oxygen_posteriors = {}
        
        for i, target_data in enumerate(data['targets']):
            stellar = target_data['stellar']
            
            target_names.append(stellar['name'])
            target_ids.append(stellar['target_id'])
            teff_vals.append(stellar['teff'])
            teff_errs.append(stellar['teff_err'])
            spectral_types.append(stellar['spectral_type'])
            metallicity_vals.append(stellar['metallicity'])
            metallicity_errs.append(stellar['metallicity_err'])
            
            # Carbon data
            if 'carbon' in target_data:
                c_data = target_data['carbon']
                carbon_ratios.append(c_data['ratio_median'])
                carbon_lower.append(c_data['ratio_lower'])
                carbon_upper.append(c_data['ratio_upper'])
                carbon_sigma.append(c_data['sigma'])
                carbon_runs.append(c_data['run'])
                carbon_posteriors[stellar['target_id']] = c_data['posterior_samples']
            else:
                carbon_ratios.append(np.nan)
                carbon_lower.append(np.nan)
                carbon_upper.append(np.nan)
                carbon_sigma.append(np.nan)
                carbon_runs.append('')
            
            # Oxygen data
            if 'oxygen' in target_data:
                o_data = target_data['oxygen']
                oxygen_ratios.append(o_data['ratio_median'])
                oxygen_lower.append(o_data['ratio_lower'])
                oxygen_upper.append(o_data['ratio_upper'])
                oxygen_sigma.append(o_data['sigma'])
                oxygen_runs.append(o_data['run'])
                oxygen_posteriors[stellar['target_id']] = o_data['posterior_samples']
            else:
                oxygen_ratios.append(np.nan)
                oxygen_lower.append(np.nan)
                oxygen_upper.append(np.nan)
                oxygen_sigma.append(np.nan)
                oxygen_runs.append('')
        
        # Save tabular data
        targets_grp.create_dataset('target_names', data=np.array(target_names, dtype='S20'))
        targets_grp.create_dataset('target_ids', data=np.array(target_ids, dtype='S20'))
        targets_grp.create_dataset('teff', data=teff_vals)
        targets_grp.create_dataset('teff_err', data=teff_errs)
        targets_grp.create_dataset('spectral_types', data=np.array(spectral_types, dtype='S10'))
        targets_grp.create_dataset('metallicity', data=metallicity_vals)
        targets_grp.create_dataset('metallicity_err', data=metallicity_errs)
        
        # Carbon isotope data
        carbon_grp = targets_grp.create_group('carbon')
        carbon_grp.create_dataset('ratio_median', data=carbon_ratios)
        carbon_grp.create_dataset('ratio_lower', data=carbon_lower)
        carbon_grp.create_dataset('ratio_upper', data=carbon_upper)
        carbon_grp.create_dataset('sigma', data=carbon_sigma)
        carbon_grp.create_dataset('runs', data=np.array(carbon_runs, dtype='S10'))
        
        # Oxygen isotope data
        oxygen_grp = targets_grp.create_group('oxygen')
        oxygen_grp.create_dataset('ratio_median', data=oxygen_ratios)
        oxygen_grp.create_dataset('ratio_lower', data=oxygen_lower)
        oxygen_grp.create_dataset('ratio_upper', data=oxygen_upper)
        oxygen_grp.create_dataset('sigma', data=oxygen_sigma)
        oxygen_grp.create_dataset('runs', data=np.array(oxygen_runs, dtype='S10'))
        
        # Save posterior samples
        if carbon_posteriors:
            carbon_post_grp = carbon_grp.create_group('posteriors')
            for target_id, posterior in carbon_posteriors.items():
                carbon_post_grp.create_dataset(target_id, data=posterior)
        
        if oxygen_posteriors:
            oxygen_post_grp = oxygen_grp.create_group('posteriors')
            for target_id, posterior in oxygen_posteriors.items():
                oxygen_post_grp.create_dataset(target_id, data=posterior)
        
        # Save GCE models
        for mass_range, model_data in data['gce_models'].items():
            model_grp = models_grp.create_group(mass_range)
            model_grp.create_dataset('metallicity', data=model_data['metallicity'])
            model_grp.create_dataset('carbon_ratio', data=model_data['carbon_ratio'])
            model_grp.create_dataset('oxygen_ratio', data=model_data['oxygen_ratio'])
            model_grp.attrs['description'] = model_data['description']
        
        # Save reference values
        for ref_name, ref_data in data['reference_values'].items():
            ref_grp = refs_grp.create_group(ref_name)
            
            if ref_name == 'crossfield_2019':
                for target, target_ref in ref_data.items():
                    target_grp = ref_grp.create_group(target.replace(' ', '_'))
                    for isotope, iso_data in target_ref.items():
                        iso_grp = target_grp.create_group(isotope)
                        for key, value in iso_data.items():
                            iso_grp.create_dataset(key, data=value)
            else:
                for isotope, iso_data in ref_data.items():
                    iso_grp = ref_grp.create_group(isotope)
                    iso_grp.create_dataset('ratio', data=iso_data['ratio'])
                    iso_grp.create_dataset('error', data=iso_data['error'])
                    iso_grp.attrs['reference'] = iso_data['reference']
        
        # Save metadata as attributes
        for key, value in data['metadata'].items():
            if isinstance(value, dict):
                meta_grp = f.create_group(f'metadata/{key}')
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, str):
                        meta_grp.attrs[subkey] = subvalue
                    else:
                        meta_grp.create_dataset(subkey, data=subvalue)
            else:
                f.attrs[key] = value
    
    print(f"Data saved successfully to {filename}")
    print(f"Total targets: {len(data['targets'])}")
    print(f"Carbon detections: {len(carbon_posteriors)}")
    print(f"Oxygen detections: {len(oxygen_posteriors)}")


def main():
    """Main execution function."""
    
    # Collect all data
    data = collect_all_data()
    
    # Save to HDF5
    output_file = '/home/dario/phd/retrieval_base/paper/isotope_data.h5'
    save_to_hdf5(data, output_file)
    
    print(f"\nData collection complete!")
    print(f"Output file: {output_file}")


if __name__ == '__main__':
    main() 