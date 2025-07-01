""" 
Generate P-T profile plots with proper pressure scaling

date: 2024-09-17
"""
import pathlib
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import copy

from retrieval_base.retrieval import Retrieval
import retrieval_base.auxiliary_functions as af
from retrieval_base.config import Config
import seaborn as sns

# Configuration
path = pathlib.Path(af.get_path())
path_figures = pathlib.Path('/home/dario/phd/twa2x_paper/figures')
config_file = 'config_jwst.txt'
w_set = 'NIRSpec'

runs = dict(
    TWA27A='freeslab_lbl10_G1G2G3_1',
    TWA28='freeslab_lbl10_G1G2G3_1',
)

colors = dict(
    TWA28={
        'data': 'k', 
        'model': '#e89c4b',
        'crires': 'brown'
    },
    TWA27A={
        'data': '#733b27',
        'model': 'seagreen',
    }
)

targets_params = dict(
    TWA27A={'teff': (2430, 20)},
    TWA28={'teff': (2382, 42)},
)

def check_dir(target: str) -> None:
    """Change to target directory if not already there."""
    cwd = os.getcwd()
    if target not in cwd:
        os.chdir(f'{path}/{target}')
        print(f'Changed directory to {target}')

def get_PT(path: pathlib.Path, target: str, run: str, config_file: str = 'config_jwst.txt', cache: bool = True) -> tuple:
    """
    Get pressure-temperature profiles and related data.
    
    Returns:
        tuple: (pressure, temperature_envelopes, integrated_contribution_function, log_g)
    """
    envelopes_dir = path / target / f'retrieval_outputs/{run}/test_data' / 'envelopes'
    envelopes_dir.mkdir(parents=True, exist_ok=True)

    PT_envelopes_file = envelopes_dir / 'PT_envelopes.npy'
    logg_posterior_file = path / target / f'retrieval_outputs/{run}/test_data' / 'log_g_posterior.npy'

    if cache and PT_envelopes_file.exists() and logg_posterior_file.exists():
        print(f' --> Found {PT_envelopes_file} and {logg_posterior_file}')
        PT_envelopes_data = np.load(PT_envelopes_file)
        pressure = PT_envelopes_data[0]
        temperature = PT_envelopes_data[1:-1]
        icf = PT_envelopes_data[-1]
        logg_posterior = np.load(logg_posterior_file)
        logg = np.median(logg_posterior)
        print(f' --> Loaded PT_envelopes.npy with shape {temperature.shape}')
        print(f' --> log_g = {logg:.2f}')
    else:
        print(f' Calculating PT envelopes for {run}')
        check_dir(target)
        conf = Config(path=path, target=target, run=run)(config_file)        
            
        ret = Retrieval(conf=conf, evaluation=False)
        bestfit_params, posterior = ret.PMN_analyze()
        print(f'posterior.shape = {posterior.shape}')
        bestfit_params_dict = dict(zip(ret.Param.param_keys, bestfit_params))
        print(f' --> Best-fit parameters: {bestfit_params_dict}')
        bestfit_params = np.array(list(bestfit_params_dict.values()))

        ret.evaluate_model(bestfit_params)
        ret.evaluation = True
        ret.PMN_lnL_func()
        ret.get_PT_mf_envelopes(posterior)
        
        logg_posterior = posterior[:,list(ret.Param.param_keys).index('log_g')]
        np.save(logg_posterior_file, logg_posterior)
        print(f' --> Saved {logg_posterior_file}')
        
        # Save PT envelopes as npy file with pressure and temperature envelopes
        ret.copy_integrated_contribution_emission()
        np.save(PT_envelopes_file, np.vstack([ret.PT.pressure, ret.PT.temperature_envelopes, ret.PT.int_contr_em['NIRSpec']]))
        print(f' --> Saved {PT_envelopes_file}')
        ret.Chem.get_VMRs_posterior(save_to=envelopes_dir)
        return ret.PT.pressure, ret.PT.temperature_envelopes, ret.PT.int_contr_em['NIRSpec'], np.median(logg_posterior)
        
    return pressure, temperature, icf, logg

def scale_pressure(pressure: np.ndarray, log_g: float, scaling_mode: str) -> np.ndarray:
    """
    Scale pressure with surface gravity.
    
    Parameters:
        pressure: Pressure array in bar
        log_g: log10 of surface gravity in cm/s²
        scaling_mode: 'multiply' to get P*g, 'divide' to get P/g, 'none' for no scaling
    
    Returns:
        Scaled pressure array with appropriate units
    """
    if scaling_mode == 'multiply':
        return pressure * 10.0**log_g
    elif scaling_mode == 'divide':
        return pressure / 10.0**log_g
    else:
        return pressure

def calculate_photospheric_temperature(temperature_envelopes: np.ndarray, 
                                      contribution_function: np.ndarray, 
                                      quantile_threshold: float = 0.9) -> tuple:
    """
    Calculate photospheric temperature from the contribution function.
    
    Parameters:
        temperature_envelopes: Temperature envelope array (7 x n_layers)
        contribution_function: Contribution function array
        quantile_threshold: Quantile threshold for photosphere definition (default 0.9)
    
    Returns:
        tuple: (photospheric_temperature, temperature_uncertainty)
    """
    # Define photosphere as region where contribution function > quantile threshold
    cf_threshold = np.quantile(contribution_function, quantile_threshold)
    photosphere_mask = contribution_function >= cf_threshold
    
    if not photosphere_mask.any():
        print(f'Warning: No points above {quantile_threshold:.1%} quantile threshold')
        return np.nan, np.nan
    
    # Get median temperature profile (index 3 is the median)
    median_temperature = temperature_envelopes[3, :]
    
    # Calculate photospheric temperature and uncertainty
    T_phot = np.mean(median_temperature[photosphere_mask])
    T_phot_err = np.std(median_temperature[photosphere_mask])
    
    return T_phot, T_phot_err

def plot_envelopes(pressure: np.ndarray, 
                   temperature_envelopes: np.ndarray, 
                   log_g: float,
                   scaling_mode: str = 'none',
                   ax=None, 
                   contribution_function=None,
                   target_name: str = '',
                   **kwargs) -> plt.Axes:
    """
    Plot temperature envelopes vs scaled pressure.
    
    Parameters:
        pressure: Pressure array
        temperature_envelopes: Temperature envelope array (7 x n_layers)
        log_g: log10 of surface gravity
        scaling_mode: Pressure scaling mode
        ax: Matplotlib axes
        contribution_function: Optional contribution function to overlay
        target_name: Name of the target for photospheric temperature printing
        **kwargs: Additional plotting arguments
    """
    ax = ax or plt.gca()
    assert len(temperature_envelopes.shape) > 1, f'Expected 2D array, got {temperature_envelopes.shape}'
    assert temperature_envelopes.shape[0] == 7, f'Expected 7 envelopes, got {temperature_envelopes.shape[0]}'
    
    color = kwargs.pop('color', 'brown')
    alpha = kwargs.pop('alpha', 0.2)
    label = kwargs.pop('label', '')
    
    # Scale pressure
    y = scale_pressure(pressure, log_g, scaling_mode)
    
    # Calculate and print photospheric temperature if contribution function is provided
    if contribution_function is not None:
        T_phot, T_phot_err = calculate_photospheric_temperature(temperature_envelopes, contribution_function)
        if not np.isnan(T_phot):
            print(f' --> {target_name} photospheric temperature: {T_phot:.1f} ± {T_phot_err:.1f} K (90% quantile of contribution function)')
    
    # Plot confidence envelopes
    for i in range(3):
        ax.fill_betweenx(y, 
                        temperature_envelopes[i,:],
                        temperature_envelopes[-(i+1),:], 
                        color=color, alpha=alpha, lw=0, 
                        label=label if i == 0 else '')
    
    # Plot median temperature profile
    ax.plot(temperature_envelopes[3,:], y, color=color, lw=1.2, 
            ls=kwargs.pop('ls', '-'), alpha=0.75)

    # Overlay contribution function if provided
    if contribution_function is not None:
        fill_cf = kwargs.pop('fill_cf', False)
        ax_cf = ax.twiny()
        ls_cf = kwargs.pop('ls_cf', ':')
        lw_cf = kwargs.pop('lw_cf', 2.5)
        ax_cf.plot(contribution_function, y, color=color, lw=lw_cf, ls=ls_cf, alpha=0.75)
        ax_cf.set_xticks([])
        ax_cf.set_yticks([])
        ax_cf.set_xlim(0, np.max(contribution_function) * 4.5)
        if fill_cf:
            ax_cf.fill_between(contribution_function, y, color=color, alpha=0.05)
        
    return ax

def plot_crires_data(ax: plt.Axes, scaling_mode: str) -> plt.Axes:
    """Plot CRIRES+ data for TWA28."""
    run_full = 'final_full'
    p, t, cf, log_g = get_PT(path, 'TWA28', run=run_full, config_file='config_freechem.txt')
    
    return plot_envelopes(p, t, log_g, scaling_mode, ax=ax, contribution_function=cf, 
                         color=colors['TWA28']['crires'], alpha=0.2, 
                         label='TWA 28\n' + r'(CRIRES$^{+}$)', fill_cf=True,
                         target_name='TWA28 (CRIRES+)',
                         ls='--', ls_cf='--', lw_cf=1.0)

def get_ylabel(scaling_mode: str) -> str:
    """Get appropriate y-axis label based on scaling mode."""
    if scaling_mode == 'multiply':
        return r'Scaled pressure (P $\times$ g) / bar$\cdot$cm$\cdot$s$^{-2}$'
    elif scaling_mode == 'divide':
        return r'Scaled pressure (P / g) / bar$\cdot$s$^{2}$$\cdot$cm$^{-1}$'
    else:
        return r'Pressure / bar'

def get_figure_suffix(scaling_mode: str) -> str:
    """Get appropriate figure name suffix based on scaling mode."""
    if scaling_mode == 'multiply':
        return '_scaled_multiply'
    elif scaling_mode == 'divide':
        return '_scaled_divide'
    else:
        return ''

def main():
    """Main plotting function."""
    # Set pressure scaling mode
    scaling_mode = 'divide'  # Options: 'multiply', 'divide', 'none'
    
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), tight_layout=True)
    
    # Plot effective temperature lines and data
    for target in runs.keys():
        Teff = targets_params[target]['teff']
        label_teff = f'{Teff[0]:.0f} K'
        ax.axvline(Teff[0], color=colors[target]['model'], ls=':', lw=2, 
                  zorder=-10, alpha=0.8, label=label_teff)
        
        # Add CRIRES+ data for TWA28
        if target == 'TWA28':
            plot_crires_data(ax, scaling_mode)
        
        # Plot main retrieval results
        p, t, cf, logg = get_PT(path, target, runs[target], cache=True)
        ax = plot_envelopes(p, t, logg, scaling_mode, ax=ax, contribution_function=cf, 
                           color=colors[target]['model'], alpha=0.4, fill_cf=True,
                           label='TWA ' + target.replace('TWA', ''),
                           target_name=target,
                           ls_cf='-', lw_cf=1.0)

    # Set plot limits and labels
    y_scaled = scale_pressure(p, logg, scaling_mode)
    ylim = (np.max(y_scaled), np.min(y_scaled))
    ylabel = get_ylabel(scaling_mode)
    
    ax.set(yscale='log', ylim=ylim, ylabel=ylabel, xlabel='Temperature (K)')
    ax.set_xlim(None, 5000)
    
    # Configure legend
    handles, labels = ax.get_legend_handles_labels()
    labels_sort = ['2430 K', 'TWA 27A', '2382 K', 'TWA 28', 'TWA 28\n(CRIRES$^{+}$)']
    legend_dict = dict(zip(labels, handles))
    handles_sort = [legend_dict[label] for label in labels_sort]
    
    leg = ax.legend(handles_sort, labels_sort, 
                    prop={'size': 10, 'weight': 'bold'}, 
                    loc=(0.54, 0.6),
                    frameon=False, ncol=1)
    
    # Style legend patches
    for patch in leg.get_patches():
        patch.set_alpha(0.70)

    # Save figure
    fig_suffix = get_figure_suffix(scaling_mode)
    fig_name = path_figures / f'fig_PTs{fig_suffix}.pdf'
    fig.savefig(fig_name, bbox_inches='tight')
    print(f' --> Saved {fig_name}')
    plt.close(fig)

if __name__ == "__main__":
    main()