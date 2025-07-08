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
            ('freeslab_lbl10_G2_2', 'G2') # updated 2025-07-03
        ],
        'TWA28': [
            ('freeslab_lbl10_G2G3_1', 'G2+G3'),
            ('freeslab_lbl10_G1G2G3_1', 'G1+G2+G3'),
            ('freeslab_lbl10_G2_1', 'G2') # updated 2025-07-03
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
    files = [PT_VMRs_COH_file]
    
    posterior = None
    ret = Retrieval(conf=conf, evaluation=False)
    
    if not cache or not all(os.path.exists(file) for file in files):
        _, posterior = ret.PMN_analyze()
    
    _ = ret.get_PT_mf_envelopes(posterior=posterior, n_samples=None, cache=cache)
    
    return ret.Chem

def get_posteriors(chem):
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
    
    # Handle both oxygen isotope ratios
    if 'H2O/H2O_181' in isotope_ratios:
        isotope_ratios['16O/18O_H2O'] = isotope_ratios['H2O/H2O_181']
    
    if '12CO/C18O' in isotope_ratios:
        isotope_ratios['16O/18O_CO'] = isotope_ratios['12CO/C18O']
    
    return CO_posterior, CH_posterior, isotope_ratios

def load_crires_data(path, target):
    """Load CRIRES data for TWA28"""
    file_crires = path / target / f'retrieval_outputs/final_full/test_data/bestfit_Chem.pkl'
    chem_crires = af.pickle_load(file_crires)
    
    # print(chem_crires.VMRs_posterior.keys())
    crires_data = {
        'C/O': chem_crires.VMRs_posterior['C/O'],
        '[C/H]': chem_crires.VMRs_posterior['Fe/H'],
        '12C/13C': chem_crires.VMRs_posterior['12_13CO'],
        '16O/18O_H2O': chem_crires.VMRs_posterior['H2_16_18O'],
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
    
    return measurements

def load_gamma_corrections(path, target, run):
    """Load gamma calibration distributions for a target"""
    gamma_corrections_file = path / target / 'retrieval_outputs' / run / 'test_data' / 'gamma_corrections.npz'
    
    if gamma_corrections_file.exists():
        data = np.load(gamma_corrections_file)
        corrections = {
            'gamma_factor': data['gamma_factor'],
            'correction_applied': bool(data['correction_applied']),
            '12C_13C_original': data['12C_13C_original'],
            'CO_original': data['CO_original']
        }
        
        if corrections['correction_applied']:
            corrections['12C_13C_calibrated'] = data['12C_13C_corrected']
            corrections['CO_calibrated'] = data['CO_corrected']
        else:
            corrections['12C_13C_calibrated'] = None
            corrections['CO_calibrated'] = None
            
        print(f'Loaded gamma calibrations for {target} {run}')
        return corrections
    else:
        print(f'No gamma calibrations found for {target} {run}')
        return None

def apply_gamma_corrections(data_dict, corrections):
    """Apply gamma calibrations to create calibrated data dictionary"""
    if corrections is None or not corrections['correction_applied']:
        return None
    
    calibrated_dict = {
        'C/O': corrections['CO_calibrated'],
        '12C/13C': corrections['12C_13C_calibrated'],
        '16O/18O_H2O': data_dict['16O/18O_H2O'],  # No calibration for H2O oxygen isotopes
        '16O/18O_CO': data_dict['16O/18O_CO'],    # No calibration for CO oxygen isotopes
    }
    
    # Apply gamma correction to 16O/18O_CO if available
    if data_dict['16O/18O_CO'] is not None and corrections['gamma_factor'] is not None:
        # The gamma factor corrects the 16O/18O ratio: corrected_ratio = original_ratio * gamma_factor
        calibrated_dict['16O/18O_CO'] = data_dict['16O/18O_CO'] * corrections['gamma_factor']
    
    return calibrated_dict

def plot_hist(ax, data, color, label=None, bins=20, alpha=0.65, density=True, linestyle='-', fill_alpha=None, zorder=None, fill=True):
    """Plot filled histogram with outline"""
    if data is None:
        return
    
    # Plot filled histogram only if fill=True
    if fill and (fill_alpha is not None or linestyle == '-'):
        fill_alpha = fill_alpha if fill_alpha is not None else alpha
        ax.hist(data, bins=bins, alpha=fill_alpha, color=color, density=density,
                histtype='stepfilled', edgecolor='k', label=label)
    else:
        # Plot outline only (for corrected distributions)
        ax.hist(data, bins=bins, alpha=1.0, color=color, density=density,
                histtype='step', edgecolor=color, linestyle=linestyle, linewidth=2,
                label=label, zorder=zorder if zorder is not None else 10)

def plot_reference_values(ax, row):
    """Plot solar and ISM reference values"""
    solar = {
        'C/O': (0.59, 0.08),
        '12C/13C': (93.5, 3.1),
        '16O/18O': (511, 10), # Ayres+2013
        'color': 'magenta',
        'label': 'Solar' if row == 0 else None
    }
    ism = {
        '12C/13C': (68, 14),
        '16O/18O': (557, 30), # Wilson+1999
        'color': 'deepskyblue',
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
        if value.get('16O/18O') is not None:
            # Plot on both oxygen isotope ratio plots
            ax[row,2].errorbar(value['16O/18O'][0], 0.002, xerr=value['16O/18O'][1], **eb_args)
            ax[row,3].errorbar(value['16O/18O'][0], 0.002, xerr=value['16O/18O'][1], **eb_args)

def setup_axes(fig, ax):
    """Setup axes labels and appearance"""
    param_labels = ['C/O', r'$\mathrm{^{12}C}/\mathrm{^{13}C}$', r'$\mathrm{^{16}O}/\mathrm{^{18}O}$ (H$_2$O)', r'$\mathrm{^{16}O}/\mathrm{^{18}O}$ (CO)']
    xlims = [(0.36, 0.70), (10, 160), (200, 1400), (200, 1400)]
    
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

def add_correction_annotations(ax, row):
    """Add text annotations with arrows pointing to calibrated distributions"""
    if row == 0:  # TWA27A - has calibrations
        import matplotlib.patheffects as pe
        pe_white = pe.withStroke(linewidth=8, foreground='w')

        # Single text box positioned between the first two plots
        text_x = 0.25  # Position between C/O and 12C/13C plots
        text_y = 0.65  # Upper part of the figure
        bbox = dict(boxstyle='round,pad=0.3', facecolor='navy', 
                         edgecolor='navy', alpha=0.1, linewidth=1)
        bbox_copy = bbox.copy()
        bbox_copy['facecolor'] = 'none'
        bbox_copy['edgecolor'] = 'navy'
        bbox_copy['alpha'] = 0.9
        bbox_copy['linewidth'] = 1.5
        # Add single text annotation in figure coordinates
        # fig = ax[0, 0].figure
        # fig.text(text_x, text_y, 
        #         #  'calibrated', 
        #         # r'${\rm H_2^{16}O/H_2^{18}O} \approx {\rm ^{12}CO/C^{18}O}$',
        #         'calibrated',
        #         fontsize=10, ha='center', va='center',
        #         bbox=bbox,
        #         zorder=20)
        
        
        # fig.text(text_x, text_y, 
        #         #  'calibrated', 
        #         # r'${\rm H_2^{16}O/H_2^{18}O} \approx {\rm ^{12}CO/C^{18}O}$',
        #         'calibrated',
        #         fontsize=10, ha='center', va='center',
        #         zorder=20,
        #         bbox=bbox_copy)
        
        # Arrow pointing to C/O calibrated distribution (column 0) with white edge
        # arrow1 = ax[row, 0].annotate('', 
        #                    xy=(0.565, 11), xytext=(0.75, 25),  # Point to calibrated peak
        #                    arrowprops=dict(arrowstyle='->', color='gray', 
        #                                  lw=1.5, alpha=0.8, zorder=-1),
        #                    zorder=0)
        # arrow1.arrow_patch.set_path_effects([pe_white])
        
        # # Arrow pointing to 12C/13C calibrated distribution (column 1) with white edge
        # arrow2 = ax[row, 1].annotate('', 
        #                    xy=(77, 0.028), xytext=(-1, 0.052),  # Point to calibrated peak at ~79.4
        #                    arrowprops=dict(arrowstyle='->', color='gray', 
        #                                  lw=1.5, alpha=0.8, zorder=-1),
        #                    zorder=0)
        # arrow2.arrow_patch.set_path_effects([pe_white])
        fig = ax[0, 3].figure
        # Add annotation for 16O/18O_CO calibration (column 3) - same style as C/O
        text_x_co = 0.84  # Position near the 16O/18O_CO plot
        text_y_co = 0.65  # Same height as other annotation
        
        fig.text(text_x_co, text_y_co, 
                'calibrated',  # Same text as C/O panel
                fontsize=10, ha='center', va='center',
                bbox=bbox,
                zorder=20)
        fig.text(text_x_co, text_y_co, 
                'calibrated',  # Same text as C/O panel
                fontsize=10, ha='center', va='center',
                bbox=bbox_copy,
                zorder=20)
        
        arrow1 = ax[row, 0].annotate('', 
                           xy=(0.58, 0.05), xytext=(0.48, 0.05),  # Point to calibrated peak
                           arrowprops=dict(arrowstyle='->', color='gray', 
                                         lw=1.5, alpha=0.8, zorder=-1),
                           zorder=0)
        arrow1.arrow_patch.set_path_effects([pe_white, pe.Normal()])
        
        # Arrow pointing to 16O/18O_CO calibrated distribution (column 3) with white edge
        arrow3 = ax[row, 3].annotate('', 
                           xy=(500, 0.0035), xytext=(450, 0.0055),  # Point to calibrated distribution
                           arrowprops=dict(arrowstyle='->', color='gray', 
                                         lw=1.5, alpha=0.8, zorder=-1),
                           zorder=0)
        arrow3.arrow_patch.set_path_effects([pe_white, pe.Normal()])
    # For TWA28 (row 1), no annotations needed since no calibrations are applied

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
    param_order = ['C/O', '12C/13C', '16O/18O_H2O', '16O/18O_CO']
    
    for row, target in enumerate(runs.keys()):
        target_runs = runs[target]
        
        # Plot JWST data
        for r, run_name in enumerate(target_runs):
            run, label = run_name
            chem = load_data(path, target, run)
            CO_posterior, CH_posterior, isotope_ratios = get_posteriors(chem)
            
            # Create data dictionary to match the new order
            data_dict = {
                'C/O': CO_posterior,
                '12C/13C': isotope_ratios['12C/13C'],
                '16O/18O_H2O': isotope_ratios.get('16O/18O_H2O'),
                '16O/18O_CO': isotope_ratios.get('16O/18O_CO'),
            }
            
            # Load gamma calibrations only for G1+G2+G3 runs
            corrections = None
            if 'G1G2G3' in run:  # Only load calibrations for G1+G2+G3 runs
                corrections = load_gamma_corrections(path, target, run)
            
            # use tabulate to print data_dict with 1,3 sigma uncertainties
            quantiles = np.array([0.16, 0.5, 0.84])
            quantiles_data = {k: np.percentile(v, quantiles*100) if v is not None else None for k, v in data_dict.items()}
            print(f' --> {target} {run}')
            print(tabulate(quantiles_data, headers='keys', tablefmt='grid'))
            # print the C/O results as q50 +- q84-q16
            print(f' --> C/O: {quantiles_data["C/O"][1]:.3f} +- {quantiles_data["C/O"][2]-quantiles_data["C/O"][0]:.3f}')
            color = colors[target]['model'][r]
            # Simplified label without target name
            label = f'NIRSpec/{label}'
            
            quantiles_isotope_ratios = {k: np.percentile(v, quantiles*100) if v is not None else None for k, v in isotope_ratios.items()}
            print(tabulate(quantiles_isotope_ratios, headers='keys', tablefmt='grid'))
            
            for col, param in enumerate(param_order):
                plot_hist(ax[row,col], data_dict[param], color, label=label,
                          fill_alpha=0.60, fill=True)
            
            # Plot calibrated distributions (unfilled, same color) for G1+G2+G3 runs
            if corrections is not None:
                calibrated_dict = apply_gamma_corrections(data_dict, corrections)
                if calibrated_dict is not None:
                    calibrated_label = f'NIRSpec/{label} (γ-calibrated)'
                    for col, param in enumerate(param_order):
                        if param in ['C/O', '12C/13C']:  # Plot C/O and 12C/13C calibrations
                            plot_hist(ax[row,col], calibrated_dict[param], color, 
                                     label=calibrated_label if col == 0 else None,
                                     fill=True, fill_alpha=0.1, linestyle='-', alpha=0.9)
                            plot_hist(ax[row,col], calibrated_dict[param], color, 
                                     label=calibrated_label if col == 0 else None,
                                     fill=False, linestyle='-', alpha=0.9)
                            
                            # Add arrow showing the shift for C/O and 12C/13C
                            if target == 'TWA27A' and data_dict[param] is not None and calibrated_dict[param] is not None:
                                original_median = np.median(data_dict[param])
                                calibrated_median = np.median(calibrated_dict[param])
                                # Set arrow height based on parameter
                                arrow_heights = {'C/O': 5, '12C/13C': 0.016, '16O/18O_H2O': 0.003, '16O/18O_CO': 0.003}
                                arrow_height = arrow_heights[param]
                                # Add arrow pointing from original to calibrated with white edge
                                import matplotlib.patheffects as pe
                                arrow = ax[row,col].annotate('', 
                                                   xy=(calibrated_median, arrow_height), 
                                                   xytext=(original_median, arrow_height),
                                                   arrowprops=dict(arrowstyle='->', color=color, 
                                                                 lw=2, alpha=0.8, zorder=15),
                                                   zorder=15)
                                arrow.arrow_patch.set_path_effects([pe.withStroke(linewidth=3, foreground='w')])
                        elif param == '16O/18O_CO' and target == 'TWA27A':  # Plot 16O/18O_CO calibration for TWA27A only
                            # Fill histogram with alpha=0.1
                            plot_hist(ax[row,col], calibrated_dict[param], color, 
                                     label=calibrated_label if col == 3 else None,
                                     fill=True, fill_alpha=0.1, linestyle='-', alpha=0.9)
                            # Unfilled outline
                            plot_hist(ax[row,col], calibrated_dict[param], color, 
                                     label=None,
                                     fill=False, linestyle='-', alpha=0.9, zorder=10)
                            
                            # Add arrow showing the shift for 16O/18O_CO
                            if data_dict[param] is not None and calibrated_dict[param] is not None:
                                original_median = np.median(data_dict[param])
                                calibrated_median = np.median(calibrated_dict[param])
                                # Add arrow pointing from original to calibrated with white edge
                                import matplotlib.patheffects as pe
                                arrow = ax[row,col].annotate('', 
                                                   xy=(calibrated_median, 0.0015), 
                                                   xytext=(original_median, 0.0015),
                                                   arrowprops=dict(arrowstyle='->', color=color, 
                                                                 lw=2, alpha=0.8, zorder=15),
                                                   zorder=15)
                                arrow.arrow_patch.set_path_effects([pe.withStroke(linewidth=3, foreground='w')])
                
        # Plot CRIRES data for TWA28
        if target == 'TWA28':
            for col, key in enumerate(param_order):
                if key in crires_data:
                    median, lower, upper = np.percentile(crires_data[key], [50, 16, 84])
                    print(f' --> {target} {key}: {median:.3f} (+ {upper-median:.3f} - {median-lower:.3f})')
                    plot_hist(ax[row,col], crires_data[key], colors[target]['crires'],
                             label=r'CRIRES$^\mathrm{+}$'+'/K2166\n(González Picos et al. 2024)', bins=20, alpha=0.40,
                             density=True, linestyle='-', fill_alpha=0.7, fill=True)
        elif target == 'TWA27A':
            for col, key in enumerate(param_order):
                if key in zhang2025_data:
                    plot_hist(ax[row,col], zhang2025_data[key], 
                              colors[target]['zhang2025'],
                            # ['orangered'],
                         label='TWA 27 b\n(Zhang et al. 2025)', bins=20, alpha=0.65,
                         density=True, linestyle='-', fill_alpha=0.3, fill=True)
        
        
        # Add reference values
        plot_reference_values(ax, row)
    
    # Add correction annotations for each row
    for row in range(2):
        add_correction_annotations(ax, row)
    
    # Setup axes appearance
    fig, ax = setup_axes(fig, ax)
    
    # Add legend to the last plot of each row, add box around legend with alpha=0.5
    for row in range(2):
        if row == 0:
            # combine legends from multiple columns including 16O/18O_CO calibrations
            handles, labels = [], []
            # Include columns 0, 1, and 3 (C/O, 12C/13C, and 16O/18O_CO)
            for col in [0, 1, 2, 3]:
                h, l = ax[row,col].get_legend_handles_labels()
                # add only if not already in handles
                for hi, li in zip(h, l):
                    if li not in labels:
                        handles.append(hi)
                        labels.append(li)
            leg_elements = {k:v for k,v in zip(labels, handles)}
            # sort by label list - include calibrated distributions
            label_list = ['NIRSpec/G1+G2+G3', 'NIRSpec/G2+G3', 'NIRSpec/G2', 'TWA 27 b\n(Zhang et al. 2025)', 'Solar', 'ISM']
            handles = [leg_elements[l] for l in label_list if l in leg_elements]
            labels = [l for l in label_list if l in leg_elements]
        else:
            # sort by label list
            handles, labels = ax[row,0].get_legend_handles_labels()
            leg_elements = {k:v for k,v in zip(labels, handles)}
            # Include calibrated distributions for TWA28
            label_list = [r'CRIRES$^\mathrm{+}$'+'/K2166\n(González Picos et al. 2024)']
            handles = [leg_elements[l] for l in label_list if l in leg_elements]
            labels = [l for l in label_list if l in leg_elements]
        
        ax[row,2].legend(handles, labels, frameon=True, fontsize=9, loc=(-0.80+0.11*row, 0.40+0.16*row), facecolor='white', edgecolor='k')

    # Save figure with calibrated suffix
    fig_name_calibrated = path_figures / 'carbon_oxygen_isotope_ratios_calibrated.pdf'
    fig.savefig(fig_name_calibrated, bbox_inches='tight', dpi=300)
    print(f'Saved {fig_name_calibrated}')
    save_png = False
    if save_png:
        # save as png
        fig_name_calibrated_png = path_figures / 'carbon_oxygen_isotope_ratios_calibrated.png'
        fig.savefig(fig_name_calibrated_png, bbox_inches='tight', dpi=300)
        print(f'Saved {fig_name_calibrated_png}')
    
    # Also save original figure (for compatibility)
    fig_name = path_figures / 'carbon_oxygen_isotope_ratios.pdf'
    fig.savefig(fig_name, bbox_inches='tight', dpi=300)
    print(f'Saved {fig_name}')
    if save_png:
        # save as png
        fig_name_png = path_figures / 'carbon_oxygen_isotope_ratios.png'
        fig.savefig(fig_name_png, bbox_inches='tight', dpi=300)
        print(f'Saved {fig_name_png}')
    
    plt.close('all')

if __name__ == "__main__":
    main()