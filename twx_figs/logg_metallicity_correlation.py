import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
from retrieval_base.retrieval import Retrieval
import retrieval_base.auxiliary_functions as af
from retrieval_base.config import Config
import seaborn as sns

def setup_paths():
    path = af.get_path(return_pathlib=True)
    path_figures = pathlib.Path('/home/dario/phd/twa2x_paper/figures')
    return path, path_figures

def define_runs_and_colors():
    runs = {
        'TWA27A': [
            ('freeslab_lbl10_G2G3_0', 'G2+G3'),
            ('freeslab_lbl10_G1G2G3_0', 'G1+G2+G3'),
        ],
        'TWA28': [
            ('freeslab_lbl10_G2G3_0', 'G2+G3'),
            ('freeslab_lbl10_G1G2G3_0', 'G1+G2+G3'),
        ]
    }
    
    # Using colorblind-friendly palette
    colors = {
        'TWA28': {
            'data': 'k',
            # 'model': ['#E69F00', 'orangered'],  # Orange, Orangered
            'model': ['orange', 'dodgerblue'],
            'crires': '#009E73'  # Green
        },
        'TWA27A': {
            'data': '#733b27',
            # 'model': ['#CC79A7', '#0072B2'],  # Pink, Dark blue
            'model': ['purple', 'forestgreen'],
            'zhang2025': 'black'
        }
    }
    
    cmaps = {
        'TWA28': ['Oranges', 'Blues', 'BuGn'],
        'TWA27A': ['Purples', 'Greens']
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

def create_correlation_plot(ax, log_g_clean, CH_clean, color, label, cmap):
    """Create a correlation plot with hexbin and regression line."""
    
    # get slope and intercept of the regression line
    slope, intercept = np.polyfit(CH_clean, log_g_clean, 1)
    correlation = np.corrcoef(log_g_clean, CH_clean)[0, 1]
    label = f"{label}\n$r={correlation:.2f}$\n$m={slope:.2f}$\n$b={intercept:.2f}$"
    hb = ax.hexbin(CH_clean, log_g_clean, gridsize=60, cmap=cmap, mincnt=1, alpha=0.5,
                   label=None)
    sns.regplot(x=CH_clean, y=log_g_clean, ax=ax, scatter=False, label=label,
                line_kws={'color': color, 'linewidth': 1.5, 'alpha': 0.6})
    # draw scater point for intercept
    ax.scatter(0, intercept, color=color, marker='s', s=30, edgecolor='k', alpha=0.7, zorder=10)
    return correlation, hb

def setup_plot():
    """Create and setup the plot figure and axis."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlabel(r'[C/H]', fontsize=12)
    ax.set_ylabel(r'$\log(g)$', fontsize=12)
    return fig, ax

def finalize_plot(fig, ax, correlations):
    """Add final touches to the plot."""
    # add spacing between rows of legend
    ax.legend(
            loc=(1.01, 0.0),
            # loc='upper left',
              ncol=1, frameon=True,
              edgecolor='k',
            #   columnspacing=1.0,
            #   handletextpad=0.5,
              fontsize=8,
              framealpha=0.5
    )
    
    # add text in bottom left showing the definition of the correlation coefficient
    text_bbox = dict(facecolor='white', alpha=0.9, edgecolor='w', boxstyle='round,pad=0.5')
    ax.text(0.04, 0.12, r"$r$" + ": Pearson's correlation\ncoefficient",
            fontsize=8,
            transform=ax.transAxes,
            bbox=text_bbox)
    # also show text with the slope and intercept of the regression definition as log(g) = m*[C/H] + b
    ax.text(0.04, 0.05, r"$\log(g) = m \cdot \text{[C/H]} + b$",
            fontsize=8,
            transform=ax.transAxes,
            bbox=text_bbox)
    ax.set_ylim(None, 4.7)
    ax.axvline(0, color='k', linestyle='-', alpha=0.3, zorder=-10)
    plt.tight_layout()

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
            
            # Create plot
            color = colors[target]['model'][i]
            plot_label = f"{target} {label}"
            correlation, _ = create_correlation_plot(ax, log_g_clean, CH_clean, color, plot_label, cmaps[target][i])
            correlations[plot_label] = correlation
            
    # add CRIRES data for TWA28
    crires_data = load_crires_data(path, 'TWA28')
    crires_correlation, _ = create_correlation_plot(ax, crires_data['log_g'], crires_data['[C/H]'], colors['TWA28']['crires'], 'TWA 28 (CRIRES)', cmaps['TWA28'][-1])
    correlations['CRIRES'] = crires_correlation
    
    finalize_plot(fig, ax, correlations)
    
    # Save the figure
    output_path = path_figures / 'logg_metallicity_correlation.pdf'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Figure saved to {output_path}")

if __name__ == "__main__":
    main()

