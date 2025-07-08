"""Generate LaTeX table with isotope ratios for different targets
"""
import numpy as np
import os
import pathlib
from retrieval_base.retrieval import Retrieval
import retrieval_base.auxiliary_functions as af
from retrieval_base.config import Config
import h5py

def setup_paths():
    path = af.get_path(return_pathlib=True)
    path_tables = pathlib.Path('/home/dario/phd/twa2x_paper/tables')
    return path, path_tables

def define_runs():
    runs = {
        'TWA27A': [
            ('freeslab_lbl10_G1G2G3_1', 'G1+G2+G3'),
        ],
        'TWA28': [
            ('freeslab_lbl10_G1G2G3_1', 'G1+G2+G3'),
        ]
    }
    # check only 1 run for each target
    assert len(runs['TWA27A']) == 1, 'Only 1 run for TWA27A'
    assert len(runs['TWA28']) == 1, 'Only 1 run for TWA28'
    
    return runs

def load_data(path, target, run, cache=True):
    cwd = os.getcwd()
    if target not in cwd:
        os.chdir(f'{path}/{target}')
        print(f'Changed directory to {target}')
    
    config_file = 'config_jwst.txt'    
    conf = Config(path=path, target=target, run=run)(config_file)
    
    log_g_posterior_file = f'{conf.prefix}data/log_g_posterior.npy'
    chem_posterior_file = f'{conf.prefix}data/chem_posterior.h5'
    files = [log_g_posterior_file, chem_posterior_file]
    
    posterior = None
    
    # Check if we need to regenerate data
    need_regenerate = not cache or not all(os.path.exists(file) for file in files)
    
    # Also check if HDF5 file exists but is incomplete
    if not need_regenerate and os.path.exists(chem_posterior_file):
        try:
            with h5py.File(chem_posterior_file, 'r') as f:
                # Check if required groups exist
                if 'COH_posterior' not in f or 'VMRs_posterior' not in f:
                    need_regenerate = True
                    print("HDF5 file incomplete, regenerating...")
        except Exception as e:
            need_regenerate = True
            print(f"Error reading HDF5 file, regenerating: {e}")
    
    if need_regenerate:
        ret = Retrieval(conf=conf, evaluation=False)
        _, posterior = ret.PMN_analyze()
        log_g_index = list(ret.Param.param_keys).index('log_g')
        log_g_posterior = posterior[:,log_g_index]
        np.save(log_g_posterior_file, log_g_posterior)
        print(f'Saved {log_g_posterior_file}')
        
        _ = ret.get_PT_mf_envelopes(posterior=posterior, n_samples=None, cache=cache)
        
        # Save chemistry posteriors as separate datasets in HDF5
        with h5py.File(chem_posterior_file, 'w') as f:
            # Save COH_posterior dictionary
            coh_grp = f.create_group('COH_posterior')
            for key, value in ret.Chem.COH_posterior.items():
                coh_grp.create_dataset(key, data=value, compression='gzip', compression_opts=9)
            
            # Save VMRs_posterior dictionary
            vmrs_grp = f.create_group('VMRs_posterior')
            for key, value in ret.Chem.VMRs_posterior.items():
                vmrs_grp.create_dataset(key, data=value, compression='gzip', compression_opts=9)
            
            # Save metadata
            f.attrs['target'] = target.encode('utf-8')
            f.attrs['run'] = run.encode('utf-8')
            f.attrs['param_keys'] = [key.encode('utf-8') for key in ret.Param.param_keys]
            
        print(f'Saved chemistry posteriors to {chem_posterior_file}')
        
        # Return the chemistry objects
        COH_posterior = ret.Chem.COH_posterior
        VMRs_posterior = ret.Chem.VMRs_posterior
        
    else:
        # Load chemistry posteriors from HDF5
        with h5py.File(chem_posterior_file, 'r') as f:
            # Load COH_posterior dictionary
            COH_posterior = {}
            for key in f['COH_posterior'].keys():
                COH_posterior[key] = f['COH_posterior'][key][:]
            
            # Load VMRs_posterior dictionary
            VMRs_posterior = {}
            for key in f['VMRs_posterior'].keys():
                VMRs_posterior[key] = f['VMRs_posterior'][key][:]
        
        print(f'Loaded chemistry posteriors from {chem_posterior_file}')
             
    log_g_posterior = np.load(log_g_posterior_file)
    
    return COH_posterior, VMRs_posterior, log_g_posterior

def get_posteriors(COH_posterior, VMRs_posterior, log_g_posterior):
    isotope_ratios_pairs = [
        ('12CO', '13CO'),
        ('12CO', 'C18O'),
        ('12CO', 'C17O'),
        ('H2O', 'H2O_181')
    ]
    ratios = {}
    for pair in isotope_ratios_pairs:
        if pair[1] in VMRs_posterior.keys():
            ratios[pair[0]+'/'+pair[1]] = np.mean(VMRs_posterior[pair[0]] / VMRs_posterior[pair[1]], axis=-1)
            
    # copy 12CO/13CO to 12C/13C
    ratios['12C/13C'] = ratios['12CO/13CO']
    
    ratios['C/O'] = np.mean(COH_posterior['C'] / COH_posterior['O'], axis=-1)
    
    return ratios

def load_crires_data(path, target):
    """Load CRIRES data for TWA28"""
    file_crires = path / target / f'retrieval_outputs/final_full/test_data/bestfit_Chem.pkl'
    chem_crires = af.pickle_load(file_crires)
    
    crires_data = {
        '12C/13C': chem_crires.VMRs_posterior['12_13CO'],
    }
    return crires_data

def load_zhang2025_data():
    """Load data from Zhang et al. 2025 for TWA27A"""
    measurements = {
        '12C/13C': None,  # No measurement available
    }
    return measurements

def format_value_with_errors(median, lower, upper):
    """Format value with asymmetric errors in LaTeX format"""
    # round to nearest integer
    # print(median, lower, upper)
    median = int(np.round(median))
    lower = int(np.round(median - lower))
    upper = int(np.round(upper - median))
    return f"${median}_{{-{lower}}}^{{+{upper}}}$"

def carbon_oxygen_ratio_correction(gamma, carbon_oxygen_ratio):
    """Correct gamma factor for different carbon-to-oxygen ratios"""
    num = gamma * carbon_oxygen_ratio
    den = 1 + (gamma - 1) * carbon_oxygen_ratio
    return num / den

def calculate_corrected_isotope_ratios(data_dict):
    """Calculate calibrated carbon isotope ratios using gamma factor calibration"""
    calibrated_data = {}
    
    for target in data_dict.keys():
        # Calculate gamma factor: (H2O/H2O_181) * (12CO/C18O)^(-1)
        if 'H2O/H2O_181' in data_dict[target] and '12CO/C18O' in data_dict[target]:
            gamma_factor = data_dict[target]['H2O/H2O_181'] * (data_dict[target]['12CO/C18O']**(-1))
            
            # Check if gamma factor is close to 1.0 (within 0.1)
            gamma_median = np.median(gamma_factor)
            if abs(gamma_median - 1.0) <= 0.1:
                print(f"Gamma factor for {target} is {gamma_median:.3f}, close to 1.0. Skipping calibration.")
                calibrated_data[target] = {
                    'gamma_factor': gamma_factor,
                    '12C/13C_calibrated': None,  # Set to None to indicate no calibration needed
                    'C/O_calibrated': None  # Also set C/O to None for consistency
                }
            else:
                carbon_oxygen_ratio = data_dict[target]['C/O']
                
                # Apply gamma factor calibration to carbon isotope ratio
                calibrated_carbon_ratio = data_dict[target]['12C/13C'] * gamma_factor
                
                # Store calibrated values
                calibrated_data[target] = {
                    'gamma_factor': gamma_factor,
                    '12C/13C_calibrated': calibrated_carbon_ratio,
                    'C/O_calibrated': carbon_oxygen_ratio_correction(gamma_factor, carbon_oxygen_ratio),
                }
                print(f"Gamma factor for {target} is {gamma_median:.3f}, applying calibration.")
    
    return calibrated_data

def generate_latex_table(data_dict, calibrated_data):
    """Generate LaTeX table from the data dictionary"""
    
    # Table header
    latex_table = [
        r"\begin{table}",
        r"\centering",
        r"\caption{Isotope ratios derived from JWST/NIRSpec observations}",
        r"\label{tab:isotope_ratios}",
        r"\begin{tabular}{lcc}",
        r"\hline\hline",
        r"\rule{0pt}{3ex}Target & TWA 27A & TWA 28 \\[1ex]", # Add extra vertical space after header
        r"\hline",
        r"\rule{0pt}{3ex}" # Add extra vertical space after hline
    ]

    # Define the isotope ratios and their LaTeX labels
    isotope_pairs = [
        ('12CO/13CO', r'$^{12}$CO/$^{13}$CO'),
        ('12CO/13CO_calibrated', r'$^{12}$CO/$^{13}$CO (calibrated)'),
        ('12CO/C18O', r'$^{12}$CO/C$^{18}$O'),
        ('H2O/H2O_181', r'H$_2^{16}$O/H$_2^{18}$O'),
        # ('12CO/C17O', r'$^{12}$CO/C$^{17}$O'),

    ]

    # Add data for each isotope ratio
    for key, label in isotope_pairs:
        values = []
        for target in ['TWA27A', 'TWA28']:
            if key == '12CO/13CO_calibrated':
                # Use calibrated values
                if target in calibrated_data and '12C/13C_calibrated' in calibrated_data[target]:
                    if calibrated_data[target]['12C/13C_calibrated'] is not None:
                        posterior = calibrated_data[target]['12C/13C_calibrated']
                        q = [0.16, 0.5, 0.84]
                        quantiles = np.percentile(posterior, [q[0]*100, q[1]*100, q[2]*100])
                        values.append(format_value_with_errors(quantiles[1], quantiles[0], quantiles[2]))
                    else:
                        values.append("---")  # Gamma factor close to 1.0, no calibration needed
                else:
                    values.append("---")
            else:
                # Use original values
                if target in data_dict and key in data_dict[target] and data_dict[target][key] is not None:
                    posterior = data_dict[target][key]
                    q = [0.16, 0.5, 0.84]
                    quantiles = np.percentile(posterior, [q[0]*100, q[1]*100, q[2]*100])
                    values.append(format_value_with_errors(quantiles[1], quantiles[0], quantiles[2]))
                else:
                    values.append("---")
        
        # Add extra vertical space between rows with \rule{0pt}{4ex}
        latex_table.append(r"\rule{0pt}{1ex}" + label + " & " + " & ".join(values) + r" \\[1ex]")
    
    # Table footer
    latex_table.extend([
        r"\hline",
        r"\end{tabular}",
        r"\tablefoot{The uncertainties represent the 16th and 84th percentiles of the posterior distributions. " +
        r"The calibrated $^{12}$CO/$^{13}$CO ratios account for oxygen isotope homogeneity using the gamma factor " +
        r"$\gamma = ({\rm H_2^{16}O/H_2^{18}O}) \times ({\rm ^{12}CO/C^{18}O})^{-1}$ to ensure consistent " +
        r"oxygen isotope ratios between CO and H$_2$O molecules. For TWA 28, $\gamma \approx 1$ within uncertainties, " +
        r"so no calibration is applied.}",
        r"\end{table}"
    ])
    
    return "\n".join(latex_table)

def save_gamma_corrections(path, data_dict, calibrated_data, runs):
    """Save gamma calibration distributions for each target"""
    for target in runs.keys():
        target_runs = runs[target]
        
        # Get the run directory for saving
        for run, label in target_runs:
            target_dir = path / target / 'retrieval_outputs' / run / 'test_data'
            target_dir.mkdir(parents=True, exist_ok=True)
            
            gamma_corrections_file = target_dir / 'gamma_corrections.npz'
            
            if target in calibrated_data:
                # Prepare data to save
                save_data = {
                    'gamma_factor': calibrated_data[target]['gamma_factor'],
                    'target': target,
                    'run': run,
                    'label': label
                }
                
                # Add calibrated distributions if they exist
                if calibrated_data[target]['12C/13C_calibrated'] is not None:
                    save_data['12C_13C_corrected'] = calibrated_data[target]['12C/13C_calibrated']
                    save_data['CO_corrected'] = calibrated_data[target]['C/O_calibrated']
                    save_data['correction_applied'] = True
                else:
                    save_data['correction_applied'] = False
                
                # Save original distributions for reference
                save_data['12C_13C_original'] = data_dict[target]['12C/13C']
                save_data['CO_original'] = data_dict[target]['C/O']
                
                # Save to compressed numpy file
                np.savez_compressed(gamma_corrections_file, **save_data)
                print(f'Saved gamma calibrations to {gamma_corrections_file}')
            
            break  # Only save once per target (using first run)

def main():
    path, path_tables = setup_paths()
    runs = define_runs()
    
    # Dictionary to store all data
    data_dict = {}
    
    # Process each target
    for target in runs.keys():
        target_runs = runs[target]
        
        # Get JWST data
        for run, label in target_runs:
            COH_posterior, VMRs_posterior, log_g_posterior = load_data(path, target, run)
            isotope_ratios = get_posteriors(COH_posterior, VMRs_posterior, log_g_posterior)
            data_dict[target] = isotope_ratios

    # Calculate calibrated isotope ratios
    calibrated_data = calculate_corrected_isotope_ratios(data_dict)
    
    # Save gamma calibrations for use in other scripts
    save_gamma_corrections(path, data_dict, calibrated_data, runs)
    
    # Generate LaTeX table
    latex_table = generate_latex_table(data_dict, calibrated_data)
    
    # Save table to file
    table_file = path_tables / 'isotope_ratios_table.tex'
    with open(table_file, 'w') as f:
        f.write(latex_table)
    print(f'Saved LaTeX table to {table_file}')
    
    # Print detailed results table
    print_corrected_results_table(data_dict, calibrated_data, runs)
    
    # Optional: Print additional diagnostic information
    print("\nDiagnostic Information:")
    print("-" * 40)
    for target in runs.keys():
        gamma_factor = data_dict[target]['H2O/H2O_181'] * (data_dict[target]['12CO/C18O']**(-1))
        print(f'{target}: Gamma factor = {np.median(gamma_factor):.3f} ± {np.std(gamma_factor):.3f}')
        
        if target in calibrated_data and calibrated_data[target]['12C/13C_calibrated'] is not None:
            print(f'  → Calibration applied: Original 12CO/13CO shifted by factor of {np.median(gamma_factor):.2f}')
        else:
            print(f'  → No calibration applied (gamma ≈ 1.0)')
    print("-" * 40)

def format_quantiles(posterior, decimals=2):
    """Format posterior quantiles as median +/- uncertainties"""
    q = [0.16, 0.5, 0.84]
    quantiles = np.percentile(posterior, [q[0]*100, q[1]*100, q[2]*100])
    median = quantiles[1]
    lower_err = median - quantiles[0]
    upper_err = quantiles[2] - median
    return f"{median:.{decimals}f} +{upper_err:.{decimals}f} -{lower_err:.{decimals}f}"

def print_corrected_results_table(data_dict, calibrated_data, runs):
    """Print a nicely formatted table with calibrated values and propagated uncertainties"""
    
    print("\n" + "="*80)
    print("CALIBRATED ISOTOPE RATIOS WITH PROPAGATED UNCERTAINTIES")
    print("="*80)
    
    # Table header
    print(f"{'Target':<10} {'Parameter':<20} {'Original':<25} {'Calibrated':<25} {'Gamma':<15}")
    print("-"*80)
    
    for target in runs.keys():
        if target in calibrated_data:
            # Original values
            original_12co13co = format_quantiles(data_dict[target]['12C/13C'], decimals=1)
            original_co = format_quantiles(data_dict[target]['C/O'], decimals=3)
            
            # Gamma factor
            gamma_factor = calibrated_data[target]['gamma_factor']
            gamma_median = np.median(gamma_factor)
            
            if calibrated_data[target]['12C/13C_calibrated'] is not None:
                # Calibrated values with full uncertainty propagation
                calibrated_12co13co = format_quantiles(calibrated_data[target]['12C/13C_calibrated'], decimals=1)
                calibrated_co = format_quantiles(calibrated_data[target]['C/O_calibrated'], decimals=3)
                gamma_str = f"{gamma_median:.3f}"
            else:
                # No calibration applied
                calibrated_12co13co = "No calibration"
                calibrated_co = "No calibration"
                gamma_str = f"{gamma_median:.3f} (≈1)"
            
            # Print rows for this target
            print(f"{target:<10} {'12CO/13CO':<20} {original_12co13co:<25} {calibrated_12co13co:<25} {gamma_str:<15}")
            print(f"{'':<10} {'C/O':<20} {original_co:<25} {calibrated_co:<25} {'':<15}")
            
            if target != list(runs.keys())[-1]:  # Add separator between targets
                print("-"*80)
    
    print("="*80)
    print("Note: Uncertainties are 16th-84th percentile ranges from posterior distributions")
    print("      Gamma factor γ = (H₂¹⁶O/H₂¹⁸O) × (¹²CO/C¹⁸O)⁻¹")
    print("      No calibration applied when γ ≈ 1.0 within 0.1")
    print("="*80)

if __name__ == "__main__":
    main()