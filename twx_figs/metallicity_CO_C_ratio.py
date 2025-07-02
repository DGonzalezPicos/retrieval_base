"""Plot histograms for the metallicity and C/O and isotope ratios for different targets
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
from retrieval_base.retrieval import Retrieval
import retrieval_base.auxiliary_functions as af
from retrieval_base.config import Config
from tabulate import tabulate

def setup_paths():
    path = af.get_path(return_pathlib=True)
    path_figures = pathlib.Path('/home/dario/phd/twa2x_paper/figures')
    return path, path_figures

def define_runs_and_colors():
    runs = {
        'TWA27A': [
            ('freeslab_lbl10_G2G3_2', 'G2+G3'),
            ('freeslab_lbl10_G1G2G3_1', 'G1+G2+G3'),
            ('freeslab_lbl10_G2_0', 'G2')
        ],
        'TWA28': [
            ('freeslab_lbl10_G2G3_1', 'G2+G3'),
            ('freeslab_lbl10_G1G2G3_1', 'G1+G2+G3'),
            ('freeslab_lbl10_G2_0', 'G2')
        ]
    }
    
    # Using colorblind-friendly palette
    colors = {
        'TWA28': {
            'data': 'k',
            # 'model': ['#0072B2', 'gold', 'darkolivegreen'],  # Orange, Orangered, Green
            'model': ['green', 'navy', 'brown'],  # Orange, Orangered, Green
            # 'crires': '#009E73'  # Green
            # 'crires': '#CC79A7'
            'crires':'#65737e'
        },
        'TWA27A': {
            'data': '#733b27',
            # 'model': ['#0072B2', 'gold', 'darkolivegreen'],  # Pink, Dark blue, Green
            'model': ['green', 'navy', 'brown'],
            'zhang2025': 'black'
        }
    }
    
    return runs, colors

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
    CO_posterior = np.mean(chem.COH_posterior['C'] / chem.COH_posterior['O'], axis=-1)
    CH_posterior = af.solar_metallicity(
        np.mean(chem.COH_posterior['C'], axis=-1),
        np.mean(chem.COH_posterior['H'], axis=-1)
    )
    isotope_ratios_pairs = [
        ('12CO', '13CO'),
        ('12CO', 'C18O'),
        ('12CO', 'C17O'),
        ('H2O', 'H2O_181')
    
    ]
    isotope_ratios = {}
    for pair in isotope_ratios_pairs:
        if pair[1] in chem.VMRs_posterior.keys():
            isotope_ratios[pair[0]+'/'+pair[1]] = np.mean(chem.VMRs_posterior[pair[0]] / chem.VMRs_posterior[pair[1]], axis=-1)
            
    # copy 12CO/13CO to 12C/13C
    isotope_ratios['12C/13C'] = isotope_ratios['12CO/13CO']
    
    return CO_posterior, CH_posterior, isotope_ratios

def load_crires_data(path, target):
    """Load CRIRES data for TWA28"""
    file_crires = path / target / f'retrieval_outputs/final_full/test_data/bestfit_Chem.pkl'
    chem_crires = af.pickle_load(file_crires)
    
    log_g_crires_file = path / target / f'retrieval_outputs/final_full/test_data/log_g_posterior.npy'
    if os.path.exists(log_g_crires_file):
        log_g_crires = np.load(log_g_crires_file)
    else:
        import pymultinest
        conf = Config(path=path, target=target, run='final_full')('config_freechem.txt')
        analyzer = pymultinest.Analyzer(
            n_params=len(conf.free_params),
            outputfiles_basename=conf.prefix
        )
        posterior = analyzer.get_equal_weighted_posterior()
        posterior = posterior[:,:-1]
        log_g_index = list(conf.free_params).index('log_g')
        log_g_crires = posterior[:,log_g_index]
        np.save(log_g_crires_file, log_g_crires)
    
    crires_data = {
        'C/O': chem_crires.VMRs_posterior['C/O'],
        '[C/H]': chem_crires.VMRs_posterior['Fe/H'],
        '12C/13C': chem_crires.VMRs_posterior['12_13CO'],
        'log_g': log_g_crires
    }
    return crires_data

def load_zhang2025_data():
    
    measurements = {
        'C/O': (0.440, 0.012),
        '[C/H]': (-0.05, 0.03),
    }
    
    # create gaussian distributions for the measurements
    for key, value in measurements.items():
        measurements[key] = np.random.normal(value[0], value[1], size=int(1e4))
    
    # add None for the other parameters
    measurements['12C/13C'] = None
    measurements['log_g'] = None
    
    return measurements
    

def plot_hist(ax, data, color, label=None, bins=20, alpha=0.65, density=True, linestyle='-', fill_alpha=None, zorder=None):
    """Plot filled histogram with outline"""
    if data is None:
        return
    # Plot filled histogram
    if fill_alpha is not None or linestyle == '-':  # Fill for solid lines or when fill_alpha is specified
        fill_alpha = fill_alpha if fill_alpha is not None else alpha
        ax.hist(data, bins=bins, alpha=fill_alpha, color=color, density=density,
                histtype='stepfilled', edgecolor='k', label=label)
    # Plot outline
    # ax.hist(data, bins=bins, alpha=1.0, color=color, density=density,
    #         histtype='step', edgecolor=color, linestyle=linestyle, 
    #         label=label if fill_alpha is None and linestyle != '-' else None, zorder=zorder)

def plot_reference_values(ax, row):
    """Plot solar and ISM reference values"""
    solar = {
        'C/O': (0.59, 0.08),
        '12C/13C': (93.5, 3.1),
        'color': 'magenta',
        'label': 'Solar' if row == 0 else None
    }
    ism = {
        '12C/13C': (68, 14),
        'color': 'mediumseagreen',
        'label': 'ISM' if row == 0 else None
    }
    
    eb_args = dict(fmt='o', markersize=6, markeredgecolor='k', markeredgewidth=1.2)
    
    for value in [solar, ism]:
        eb_args['color'] = value['color']
        eb_args['label'] = value['label']
        
        if value.get('C/O') is not None:
            ax[row,0].errorbar(value['C/O'][0], 55, xerr=value['C/O'][1], **eb_args)
        if value.get('12C/13C') is not None:
            ax[row,1].errorbar(value['12C/13C'][0], 0.05, xerr=value['12C/13C'][1], **eb_args)

def setup_axes(fig, ax):
    """Setup axes labels and appearance"""
    param_labels = ['C/O', r'$\mathrm{^{12}C}/\mathrm{^{13}C}$', '[C/H]', 'log g']
    xlims = [(0.36, 0.70), (10, 160), (-0.8, 0.8), (3.0, 5.0)]
    
    # Add target names as row labels
    targets = ['TWA 27A', 'TWA 28']
    for row, target in enumerate(targets):
        # Add text to the left of the first subplot in each row
        fig.text(0.01, 0.75 - row*0.4, target, fontsize=16, fontweight='bold', 
                rotation=90, va='center')
    
    for row in range(2):
        for col in range(4):
            axi = ax[row,col]
            # Remove unnecessary spines
            for spine in ['top', 'right', 'left']:
                axi.spines[spine].set_visible(False)
            axi.set_yticks([])
            
            # Set x-label for both rows
            if row == 1:
                axi.set_xlabel(param_labels[col], fontsize=14)
            
            axi.set_xlim(xlims[col])
    
    return fig, ax

def main():
    path, path_figures = setup_paths()
    runs, colors = define_runs_and_colors()
    
    # Create figure with more width to accommodate row labels
    fig, ax = plt.subplots(2, 4, figsize=(12, 6))
    fig.subplots_adjust(hspace=0.12, left=0.05)  # Adjust left margin for row labels
    
    # Load CRIRES data for TWA28
    crires_data = load_crires_data(path, 'TWA28')
    
    # Load Zhang et al. (2025) data for TWA27A
    zhang2025_data = load_zhang2025_data()
    
    # Define the order of parameters to plot
    param_order = ['C/O', '12C/13C', '[C/H]', 'log_g']
    
    for row, target in enumerate(runs.keys()):
        target_runs = runs[target]
        
        # Plot JWST data
        for r, run_name in enumerate(target_runs):
            run, label = run_name
            chem, log_g_posterior = load_data(path, target, run)
            CO_posterior, CH_posterior, isotope_ratios = get_posteriors(chem, log_g_posterior)
            
            # Create data dictionary to match the new order
            data_dict = {
                'C/O': CO_posterior,
                '12C/13C': isotope_ratios['12C/13C'],
                '[C/H]': CH_posterior,
                'log_g': log_g_posterior
            }
            # use tabulate to print data_dict with 1,3 sigma uncertainties
            quantiles = np.array([0.16, 0.5, 0.84])
            quantiles_data = {k: np.percentile(v, quantiles*100) for k, v in data_dict.items()}
            print(f' --> {target} {run}')
            print(tabulate(quantiles_data, headers='keys', tablefmt='grid'))
            # print the C/O results as q50 +- q84-q16
            print(f' --> C/O: {quantiles_data["C/O"][1]:.3f} +- {quantiles_data["C/O"][2]-quantiles_data["C/O"][0]:.3f}')
            color = colors[target]['model'][r]
            # Simplified label without target name
            label = f'NIRSpec/{label}'
            
            quantiles_isotope_ratios = {k: np.percentile(v, quantiles*100) for k, v in isotope_ratios.items()}
            print(tabulate(quantiles_isotope_ratios, headers='keys', tablefmt='grid'))
            
            for col, param in enumerate(param_order):
                plot_hist(ax[row,col], data_dict[param], color, label=label,
                          fill_alpha=0.60)
                
                
        # Plot CRIRES data for TWA28
        if target == 'TWA28':
            for col, key in enumerate(param_order):
                plot_hist(ax[row,col], crires_data[key], colors[target]['crires'],
                         label=r'CRIRES$^\mathrm{+}$'+'/K2166\n(González Picos et al. 2024)', bins=20, alpha=0.40,
                         density=True, linestyle='-', fill_alpha=0.7)
        elif target == 'TWA27A':
            for col, key in enumerate(param_order):
                plot_hist(ax[row,col], zhang2025_data[key], 
                          colors[target]['zhang2025'],
                        # ['orangered'],
                         label='TWA 27 b\n(Zhang et al. 2025)', bins=20, alpha=0.65,
                         density=True, linestyle='-', fill_alpha=0.3)
        
        
        # Add reference values
        plot_reference_values(ax, row)
    
    # Setup axes appearance
    fig, ax = setup_axes(fig, ax)
    
    # Add legend to the last plot of each row, add box around legend with alpha=0.5
    for row in range(2):
        if row == 0:
            # combine legends from ax[row,1] and ax[row,2]
            handles, labels = [], []
            for col in range(1,3):
                h = ax[row,col].get_legend_handles_labels()[0]
                l = ax[row,col].get_legend_handles_labels()[1]
                # add only if not already in handles
                for h, l in zip(h, l):
                    if l not in labels:
                        handles.append(h)
                        labels.append(l)
            leg_elements = {k:v for k,v in zip(labels, handles)}
            # sort by label list
            label_list = ['NIRSpec/G1+G2+G3', 'NIRSpec/G2+G3', 'NIRSpec/G2', 'TWA 27 b\n(Zhang et al. 2025)', 'Solar', 'ISM']
            handles = [leg_elements[l] for l in label_list]
            labels = label_list
            # ax[row,2].legend(handles, labels, frameon=True, fontsize=10, loc=(-0.80+0.11*row, 0.5), facecolor='white', edgecolor='k')
        else:
            # sort by label list
            handles, labels = ax[row,1].get_legend_handles_labels()
            leg_elements = {k:v for k,v in zip(labels, handles)}
            # label_list = ['NIRSpec/G1+G2+G3', 'NIRSpec/G2+G3', 'NIRSpec/G2', r'CRIRES$^\mathrm{+}$'+'/K2166\n(González Picos et al. 2024)']
            label_list = [r'CRIRES$^\mathrm{+}$'+'/K2166\n(González Picos et al. 2024)']
            handles = [leg_elements[l] for l in label_list]
            
            labels = label_list
        ax[row,2].legend(handles, labels, frameon=True, fontsize=10, loc=(-0.80+0.11*row, 0.40+0.16*row), facecolor='white', edgecolor='k')

    # Save figure
    fig_name = path_figures / 'metallicity_CO_C_ratio.pdf'
    fig.savefig(fig_name, bbox_inches='tight', dpi=300)
    # save as png
    fig_name_png = path_figures / 'metallicity_CO_C_ratio.png'
    fig.savefig(fig_name_png, bbox_inches='tight', dpi=300)
    print(f'Saved {fig_name}')
    plt.close('all')

if __name__ == "__main__":
    main()