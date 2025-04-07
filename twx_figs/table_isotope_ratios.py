"""Generate LaTeX table with isotope ratios for different targets
"""
import numpy as np
import os
import pathlib
from retrieval_base.retrieval import Retrieval
import retrieval_base.auxiliary_functions as af
from retrieval_base.config import Config

def setup_paths():
    path = af.get_path(return_pathlib=True)
    path_tables = pathlib.Path('/home/dario/phd/twa2x_paper/tables')
    return path, path_tables

def define_runs():
    runs = {
        'TWA27A': [
            ('freeslab_lbl10_G1G2G3_0', 'G1+G2+G3'),
        ],
        'TWA28': [
            ('freeslab_lbl10_G1G2G3_0', 'G1+G2+G3'),
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
    
    PT_VMRs_COH_file = f'{path}/{target}/retrieval_outputs/{run}/test_data/temperature_VMRs_COH.npy'
    log_g_posterior_file = f'{conf.prefix}data/log_g_posterior.npy'
    files = [PT_VMRs_COH_file, log_g_posterior_file]
    
    posterior = None
    ret = Retrieval(conf=conf, evaluation=False)
    
    if not cache or not all(os.path.exists(file) for file in files):
        _, posterior = ret.PMN_analyze()
        log_g_index = list(ret.Param.param_keys).index('log_g')
        log_g_posterior = posterior[:,log_g_index]
        np.save(log_g_posterior_file, log_g_posterior)
        print(f'Saved {log_g_posterior_file}')
    
    log_g_posterior = np.load(log_g_posterior_file)
    _ = ret.get_PT_mf_envelopes(posterior=posterior, n_samples=None, cache=cache)
    
    return ret.Chem, log_g_posterior

def get_posteriors(chem, log_g_posterior):
    isotope_ratios_pairs = [
        ('12CO', '13CO'),
        ('12CO', 'C18O'),
        ('12CO', 'C17O'),
        ('H2O', 'H2O_181')
    ]
    ratios = {}
    for pair in isotope_ratios_pairs:
        if pair[1] in chem.VMRs_posterior.keys():
            ratios[pair[0]+'/'+pair[1]] = np.mean(chem.VMRs_posterior[pair[0]] / chem.VMRs_posterior[pair[1]], axis=-1)
            
    # copy 12CO/13CO to 12C/13C
    ratios['12C/13C'] = ratios['12CO/13CO']
    
    ratios['C/O'] = np.mean(chem.COH_posterior['C'] / chem.COH_posterior['O'], axis=-1)
    
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

def generate_latex_table(data_dict):
    """Generate LaTeX table from the data dictionary"""
    
    # Table header
    latex_table = [
        r"\begin{table}",
        r"\centering",
        r"\caption{Isotope ratios derived from JWST/NIRSpec observations}",
        r"\label{tab:isotope_ratios}",
        r"\begin{tabular}{lcc}",
        r"\hline\hline",
        r"\rule{0pt}{4ex}Target & TWA 27A & TWA 28 \\[2ex]", # Add extra vertical space after header
        r"\hline",
        r"\rule{0pt}{4ex}" # Add extra vertical space after hline
    ]

    # Define the isotope ratios and their LaTeX labels
    isotope_pairs = [
        ('12CO/13CO', r'$^{12}$CO/$^{13}$CO'),
        ('12CO/C18O', r'$^{12}$CO/C$^{18}$O'),
        ('H2O/H2O_181', r'H$_2^{16}$O/H$_2^{18}$O'),
        ('12CO/C17O', r'$^{12}$CO/C$^{17}$O'),

    ]

    # Add data for each isotope ratio
    for key, label in isotope_pairs:
        values = []
        for target in ['TWA27A', 'TWA28']:
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
        r"\tablefoot{The uncertainties represent the 16th and 84th percentiles of the posterior distributions.}",
        r"\end{table}"
    ])
    
    return "\n".join(latex_table)

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
            chem, log_g_posterior = load_data(path, target, run)
            isotope_ratios = get_posteriors(chem, log_g_posterior)
            data_dict[target] = isotope_ratios

    
    # Generate LaTeX table
    latex_table = generate_latex_table(data_dict)
    
    # Save table to file
    table_file = path_tables / 'isotope_ratios_table.tex'
    with open(table_file, 'w') as f:
        f.write(latex_table)
    print(f'Saved LaTeX table to {table_file}')
    
    # calculate and print gamma factor defined as (H2O/H2O_181) * (12CO/C18O)**(-1)
    for target in runs.keys():
        gamma_factor = np.median(data_dict[target]['H2O/H2O_181']) * (np.median(data_dict[target]['12CO/C18O'])**(-1))
        # print(f'Gamma factor for {target}: {gamma_factor}')
        
    def carbon_oxygen_ratio_correction(gamma, carbon_oxygen_ratio):
        """Correct gamma factor for different carbon-to-oxygen ratios"""
        num = gamma * carbon_oxygen_ratio
        den = 1 + (gamma - 1) * carbon_oxygen_ratio
        return num / den
    
    for target in runs.keys():
        gamma_factor = np.median(data_dict[target]['H2O/H2O_181']) * (np.median(data_dict[target]['12CO/C18O'])**(-1))
        carbon_oxygen_ratio = np.median(data_dict[target]['C/O'])
        print(f'Gamma factor for {target}: {gamma_factor}')
        print(f'Carbon-to-oxygen ratio for {target}: {carbon_oxygen_ratio}')
        print(f'Corrected gamma factor for {target}: {carbon_oxygen_ratio_correction(gamma_factor, carbon_oxygen_ratio)}')
        
        # apply correction to 12CO/13CO
        carbon_isotope_ratio = np.median(data_dict[target]['12C/13C'])
        print(f'Carbon isotope ratio for {target}: {carbon_isotope_ratio:.2f}')
        print(f'Corrected carbon isotope ratio for {target}: {carbon_isotope_ratio * gamma_factor:.2f}')

if __name__ == "__main__":
    main()