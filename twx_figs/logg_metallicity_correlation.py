import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
from retrieval_base.retrieval import Retrieval
import retrieval_base.auxiliary_functions as af
from retrieval_base.config import Config
import seaborn as sns
from matplotlib.patches import Rectangle

fontsize = 14
plt.rcParams['font.size'] = fontsize
plt.rcParams['axes.linewidth'] = 2.0

def setup_paths():
    path = af.get_path(return_pathlib=True)
    path_figures = pathlib.Path('/home/dario/phd/twa2x_paper/figures')
    return path, path_figures

def define_runs_and_colors():
    runs = {
        'TWA27A': [
            ('freeslab_lbl10_G2G3_2', 'G2+G3'), # update to index 2
            ('freeslab_lbl10_G1G2G3_1', 'G1+G2+G3'),
        ],
        'TWA28': [
            ('freeslab_lbl10_G2G3_1', 'G2+G3'), # update to index 1
            ('freeslab_lbl10_G1G2G3_1', 'G1+G2+G3'),
        ]
    }
    
    # Publication-quality colorblind-friendly palette with better contrast
    colors = {
        'TWA28': {
            'data': '#2C2C2C',  # Dark gray for data
            'model': ['#FF6B35', '#1F77B4'],  # Orange, Blue
            'crires': '#2E8B57',  # Sea green
            'object_color': '#FF6B35'  # Main color for TWA28
        },
        'TWA27A': {
            'data': '#733b27',
            'model': ['#9467BD', '#737373'],  # Purple, Grey
            'zhang2025': 'black',
            'object_color': '#9467BD'  # Main color for TWA27A
        }
    }
    
    cmaps = {
        'TWA28': ['Oranges', 'Blues', 'BuGn'],
        'TWA27A': ['Purples', 'Greys']
    }
    
    return runs, colors, cmaps

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
    isotope_ratios = {
        '12C/13C': np.mean(chem.VMRs_posterior['12CO'] / chem.VMRs_posterior['13CO'], axis=-1)
    }
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

def clean_data(log_g_posterior, CH_posterior):
    """Clean the data by removing infinities and NaNs."""
    log_g_posterior = np.where(np.isinf(log_g_posterior), np.nan, log_g_posterior)
    CH_posterior = np.where(np.isinf(CH_posterior), np.nan, CH_posterior)
    mask = ~np.isnan(log_g_posterior) & ~np.isnan(CH_posterior)
    return log_g_posterior[mask], CH_posterior[mask]

def create_correlation_plot(ax, log_g_clean, CH_clean, color, label, cmap, alpha=0.6):
    """Create a correlation plot with hexbin and regression line."""
    
    # Get slope and intercept of the regression line
    slope, intercept = np.polyfit(CH_clean, log_g_clean, 1)
    correlation = np.corrcoef(log_g_clean, CH_clean)[0, 1]
    
    # Create hexbin plot with reduced alpha for better visibility
    hb = ax.hexbin(CH_clean, log_g_clean, gridsize=50, cmap=cmap, mincnt=1, 
                   alpha=alpha, linewidths=0.2, edgecolors='white')
    
    # Add regression line with improved styling
    sns.regplot(x=CH_clean, y=log_g_clean, ax=ax, scatter=False, 
                line_kws={'color': color, 'linewidth': 2.5, 'alpha': 0.9})
    
    # Mark intercept with improved styling
    ax.scatter(0, intercept, color=color, marker='s', s=40, 
              edgecolor='white', linewidth=1, alpha=0.9, zorder=10)
    
    return correlation, slope, intercept, hb

def setup_plot():
    """Create and setup the plot figure and axis with improved styling."""
    # plt.style.use('default')  # Ensure clean style
    fig, ax = plt.subplots(figsize=(6, 4))  # Increased figure size
    
    # Enhanced axis styling
    ax.set_xlabel(r'[C/H]', fontsize=fontsize)
    ax.set_ylabel(r'$\log(g)$', fontsize=fontsize)
    
    # Improve tick styling with increased width
    ax.tick_params(axis='both', which='major', labelsize=fontsize, width=2.0, length=6)
    ax.tick_params(axis='both', which='minor', width=1.5, length=3, labelsize=fontsize)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    # ax.set_axisbelow(True)
    
    return fig, ax

def create_custom_legend(ax, correlations, colors):
    """Create a custom legend structure with object groupings."""
    
    # Create legend elements manually for better control
    legend_elements = []
    
    # TWA28 section - just text title
    dummy_patch = plt.Line2D([0], [0], color='none')
    legend_elements.append((dummy_patch, 'TWA 28 (r, a, b)'))
    
    # TWA28 entries
    for i, (run, label) in enumerate([('freeslab_lbl10_G2G3_0', 'G2+G3'), 
                                     ('freeslab_lbl10_G1G2G3_1', 'G1+G2+G3')]):
        color = colors['TWA28']['model'][i]
        corr = correlations.get(f"TWA28 {label}", 0)
        slope = correlations.get(f"TWA28 {label}_slope", 0)
        intercept = correlations.get(f"TWA28 {label}_intercept", 0)
        
        line_patch = plt.Line2D([0], [0], color=color, linewidth=2.5, alpha=0.9)
        label_text = f"  {label}: ({corr:.2f}, {slope:.2f}, {intercept:.2f})"
        legend_elements.append((line_patch, label_text))
    
    # CRIRES entry
    crires_corr = correlations.get('CRIRES', 0)
    crires_slope = correlations.get('CRIRES_slope', 0)
    crires_intercept = correlations.get('CRIRES_intercept', 0)
    crires_patch = plt.Line2D([0], [0], color=colors['TWA28']['crires'], 
                             linewidth=2.5, alpha=0.9)
    crires_text = f"  {'CRIRES' + r'$^{+}$'}: ({crires_corr:.2f}, {crires_slope:.2f}, {crires_intercept:.2f})"
    legend_elements.append((crires_patch, crires_text))
    
    # TWA27A section - just text title
    dummy_patch2 = plt.Line2D([0], [0], color='none')
    legend_elements.append((dummy_patch2, 'TWA 27A (r, a, b)'))
    
    # TWA27A entries
    for i, (run, label) in enumerate([('freeslab_lbl10_G2G3_0', 'G2+G3'), 
                                     ('freeslab_lbl10_G1G2G3_1', 'G1+G2+G3')]):
        color = colors['TWA27A']['model'][i]
        corr = correlations.get(f"TWA27A {label}", 0)
        slope = correlations.get(f"TWA27A {label}_slope", 0)
        intercept = correlations.get(f"TWA27A {label}_intercept", 0)
        
        line_patch = plt.Line2D([0], [0], color=color, linewidth=2.5, alpha=0.9)
        label_text = f"  {label}: ({corr:.2f}, {slope:.2f}, {intercept:.2f})"
        legend_elements.append((line_patch, label_text))
    
    # Create the legend above the plot without frame
    handles, labels = zip(*legend_elements)
    legend = ax.legend(handles, labels, 
                      bbox_to_anchor=(0.5, 1.02), 
                      loc='lower center',
                      ncol=2, 
                      frameon=False,
                      fontsize=fontsize*0.8,
                      columnspacing=2.0,
                      handlelength=2.0,
                      handletextpad=0.5)
    
    return legend

def add_information_box(ax):
    """Add information box with correlation and regression definitions."""
    info_text = (
        "r: Pearson coefficient\n" +
        "log(g) = a × [C/H] + b"
        # "□: Solar metallicity intercept"
    )
    
    text_bbox = dict(facecolor='white', alpha=0.85, edgecolor='gray', 
                     linewidth=1,
                     boxstyle='round,pad=0.2')
    # increase spacing between rows of text
    ax.text(0.03, 0.94, info_text,
            fontsize=fontsize,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=text_bbox,
            zorder=15,
            )

def finalize_plot(fig, ax, correlations, colors):
    """Add final touches to the plot with improved styling."""
    
    # Create custom legend
    legend = create_custom_legend(ax, correlations, colors)
    
    # Add information box
    add_information_box(ax)
    
    # Style improvements
    ax.set_ylim(None, 4.7)
    ax.axvline(0, color='black', linestyle='--', alpha=0.6, linewidth=1.5, zorder=0)
    
    # Add subtle background
    # ax.set_facecolor('#FAFAFA')
    
    # Improve spine styling with increased width
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
        spine.set_color('gray')
    
    # Adjust layout to accommodate legend
    plt.subplots_adjust(top=0.85, bottom=0.12, left=0.12, right=0.95)

def main():
    path, path_figures = setup_paths()
    runs, colors, cmaps = define_runs_and_colors()
    
    fig, ax = setup_plot()
    correlations = {}
    
    for target in runs.keys():
        for i, (run, label) in enumerate(runs[target]):
            # Load and process data
            chem, log_g_posterior = load_data(path, target, run)
            _, CH_posterior, _ = get_posteriors(chem, log_g_posterior)
            log_g_clean, CH_clean = clean_data(log_g_posterior, CH_posterior)
            
            # Create plot with enhanced styling
            color = colors[target]['model'][i]
            plot_label = f"{target} {label}"
            correlation, slope, intercept, _ = create_correlation_plot(
                ax, log_g_clean, CH_clean, color, plot_label, cmaps[target][i]
            )
            
            # Store all statistics
            correlations[plot_label] = correlation
            correlations[f"{plot_label}_slope"] = slope
            correlations[f"{plot_label}_intercept"] = intercept
            
    # Add CRIRES data for TWA28
    crires_data = load_crires_data(path, 'TWA28')
    crires_correlation, crires_slope, crires_intercept, _ = create_correlation_plot(
        ax, crires_data['log_g'], crires_data['[C/H]'], 
        colors['TWA28']['crires'], 'TWA 28 (CRIRES)', cmaps['TWA28'][-1]
    )
    
    # Store CRIRES statistics
    correlations['CRIRES'] = crires_correlation
    correlations['CRIRES_slope'] = crires_slope
    correlations['CRIRES_intercept'] = crires_intercept
    
    finalize_plot(fig, ax, correlations, colors)
    
    # Save the figure with high quality
    output_path = path_figures / 'logg_metallicity_correlation.pdf'
    plt.savefig(output_path, bbox_inches='tight', dpi=300, 
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Publication-ready figure saved to {output_path}")

if __name__ == "__main__":
    main()

