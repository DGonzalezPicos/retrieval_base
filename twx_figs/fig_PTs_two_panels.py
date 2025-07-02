""" 
Generate P-T profile plots with separate panels for each target showing temperature profiles and residuals

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
from matplotlib import patheffects as path_effects
pe_white = [path_effects.withStroke(linewidth=2.0, foreground='w')]
# Configuration
path = pathlib.Path(af.get_path())
path_figures = pathlib.Path('/home/dario/phd/twa2x_paper/figures')
config_file = 'config_jwst.txt'
w_set = 'NIRSpec'

runs = dict(
    TWA27A=['freeslab_lbl10_G1G2G3_1', 'freeslab_lbl10_G2G3_2'],
    TWA28=['freeslab_lbl10_G1G2G3_1', 'freeslab_lbl10_G2G3_1']
)
colors_grating = ['navy', 'green', 'brown']
colors = dict(
    TWA28={
        'data': 'k', 
        'model': colors_grating[0],
        'model2': colors_grating[1],
        'crires': '#4f5b66',
    },
    TWA27A={
        'data': 'gray',
        'model': colors_grating[0],
        'model2': colors_grating[1],
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
                   ax=None, 
                   contribution_function=None,
                   target_name: str = '',
                   **kwargs) -> plt.Axes:
    """
    Plot temperature envelopes vs pressure.
    
    Parameters:
        pressure: Pressure array
        temperature_envelopes: Temperature envelope array (7 x n_layers)
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
    
    # Calculate and print photospheric temperature if contribution function is provided
    if contribution_function is not None:
        T_phot, T_phot_err = calculate_photospheric_temperature(temperature_envelopes, contribution_function)
        if not np.isnan(T_phot):
            print(f' --> {target_name} photospheric temperature: {T_phot:.1f} ± {T_phot_err:.1f} K (90% quantile of contribution function)')
    
    # Plot confidence envelopes
    for i in range(3):
        ax.fill_betweenx(pressure, 
                        temperature_envelopes[i,:],
                        temperature_envelopes[-(i+1),:], 
                        color=color, alpha=alpha, lw=0, 
                        label=label if i == 0 else '')
    
    # Plot median temperature profile
    ax.plot(temperature_envelopes[3,:], pressure, color=color, lw=1.5, 
            ls=kwargs.pop('ls', '-'), alpha=0.8)

    # Overlay contribution function if provided
    if contribution_function is not None:
        fill_cf = kwargs.pop('fill_cf', False)
        ax_cf = ax.twiny()
        ls_cf = kwargs.pop('ls_cf', ':')
        lw_cf = kwargs.pop('lw_cf', 2.5)
        ax_cf.plot(contribution_function, pressure, color=color, lw=lw_cf, ls=ls_cf, alpha=0.75)
        ax_cf.set_xticks([])
        ax_cf.set_yticks([])
        ax_cf.set_xlim(0, np.max(contribution_function) * 4.5)
        if fill_cf:
            ax_cf.fill_between(contribution_function, pressure, color=color, alpha=0.05)
        
    return ax

def plot_crires_data(ax: plt.Axes) -> tuple:
    """
    Plot CRIRES+ data for TWA28.
    
    Returns:
        tuple: (pressure, temperature_envelopes, contribution_function, log_g)
    """
    run_full = 'final_full'
    p, t, cf, log_g = get_PT(path, 'TWA28', run=run_full, config_file='config_freechem.txt')
    
    plot_envelopes(p, t, ax=ax, contribution_function=cf, 
                  color=colors['TWA28']['crires'], alpha=0.2, 
                  label='CRIRES$^{+}$',
                  target_name='TWA28 (CRIRES+)',
                  ls='--', ls_cf='--', lw_cf=1.0)
    
    return p, t, cf, log_g

def plot_temperature_residuals(pressure: np.ndarray, 
                              temp_ref: np.ndarray, 
                              temp_comp: np.ndarray,
                              pressure_comp: np.ndarray=None,
                              ax: plt.Axes=None,
                              color: str = 'blue',
                              label: str = '',
                              **kwargs) -> plt.Axes:
    """
    Plot temperature residuals (comp - ref) vs pressure.
    
    Parameters:
        pressure: Pressure array
        temp_ref: Reference temperature profile (median, index 3)
        temp_comp: Comparison temperature profile (median, index 3)
        ax: Matplotlib axes
        color: Color for the residual plot
        label: Label for the plot
        **kwargs: Additional plotting arguments
    """
    ax = ax or plt.gca()
    # Calculate residuals using median profiles (index 3)
    if len(temp_comp[3, :]) != len(temp_ref[3, :]):
        assert pressure_comp is not None, 'pressure_comp must be provided if temp_comp and temp_ref have different lengths'
        temp_comp_int = np.interp(pressure, pressure_comp, temp_comp[3, :])
    else:
        temp_comp_int = temp_comp[3, :]
    temp_diff = temp_comp_int - temp_ref[3, :]
    
    # Plot residuals
    ax.plot(temp_diff, pressure, color=color, lw=1.5, alpha=0.8, label=label,
            ls=kwargs.get('ls', '-'))
    
    # Add zero line
    ax.axvline(0, color='gray', lw=0.8, alpha=0.5, ls='--')
    
    return ax

def main():
    """Main plotting function."""
    fig, axes = plt.subplots(2, 2, figsize=(4, 6), 
                            gridspec_kw={'width_ratios': [3, 1], 'hspace': 0.20, 'wspace': 0.15})
    
    # Get axes for each target
    ax_twa27a_main, ax_twa27a_res = axes[0, 0], axes[0, 1]
    ax_twa28_main, ax_twa28_res = axes[1, 0], axes[1, 1]
    
    # Store reference profiles for residual calculations
    reference_profiles = {}
    crires_data = {}
    
    # Process each target
    for i, target in enumerate(['TWA27A', 'TWA28']):
        main_ax = ax_twa27a_main if target == 'TWA27A' else ax_twa28_main
        res_ax = ax_twa27a_res if target == 'TWA27A' else ax_twa28_res
        
        # Plot effective temperature line
        Teff = targets_params[target]['teff']
        main_ax.axvline(Teff[0], 
                        # color=colors[target]['model'], 
                        color='k',
                        ls=':', lw=2, 
                        alpha=0.8, label=f'{Teff[0]:.0f} K')
        
        # Add CRIRES+ data for TWA28
        if target == 'TWA28':
            p_crires, t_crires, cf_crires, logg_crires = plot_crires_data(main_ax)
            crires_data['TWA28'] = (p_crires, t_crires, cf_crires, logg_crires)
        
        # Process each run for this target
        for j, run in enumerate(runs[target]):
            p, t, cf, logg = get_PT(path, target, run, cache=True)
            
            # Choose color based on run index
            if j == 0:
                color = colors[target]['model']
                label = f'G1G2G3'
                # Store reference profile for residuals
                reference_profiles[target] = t
            else:
                color = colors[target]['model2']
                label = f'G2G3'
            
            # Plot temperature envelopes with contribution function for all runs
            plot_envelopes(p, t, ax=main_ax, contribution_function=cf, 
                          color=color, alpha=0.3, 
                          label=label,
                          target_name=f'{target} ({run})',
                          fill_cf=True, ls_cf='-', lw_cf=1.0)
            
            # Plot residuals (only for non-reference runs)
            if j > 0:
                plot_temperature_residuals(p, reference_profiles[target], t, ax=res_ax,
                                         color=color, label=label)
        
        # Add CRIRES residuals for TWA28
        if target == 'TWA28' and 'TWA28' in crires_data:
            # Plot residuals between reference and CRIRES
            plot_temperature_residuals(p, reference_profiles[target], t_crires, pressure_comp=p_crires, ax=res_ax,
                                     color=colors[target]['crires'], label='CRIRES$^{+}$',
                                     ls='--')
        
        # Set up main panel
        main_ax.set(yscale='log', ylim=(np.max(p), np.min(p)), 
                   ylabel='Pressure (bar)' if target == 'TWA28' else '',
                   xlabel='Temperature (K)' if target == 'TWA28' else '')
        main_ax.set_xlim(1000, 4500)
        
        # Set up residuals panel
        res_ax.set(yscale='log', ylim=(np.max(p), np.min(p)),
                  xlabel='ΔT (K)' if target == 'TWA28' else '')
        res_ax.set_xlim(-500, 500)
        
        # Add target label
        main_ax.text(0.50, 0.95, f'TWA {target.replace("TWA", "")}', 
                    transform=main_ax.transAxes, fontsize=14, fontweight='bold',
                    va='top', ha='left',
                    path_effects=pe_white)
        
        # Configure legend for main panel
        if target == 'TWA27A':
            handles, labels = main_ax.get_legend_handles_labels()
            main_ax.legend(handles, labels, loc=(0.50,0.55), fontsize=10, frameon=False)
        elif target == 'TWA28':
            handles, labels = main_ax.get_legend_handles_labels()
            main_ax.legend(handles, labels, loc=(0.50,0.45), fontsize=10, frameon=False)
        
        # Configure legend for residuals panel
        add_legend_residuals = False
        if add_legend_residuals:
            if len(runs[target]) > 1 or target == 'TWA28':
                handles_res, labels_res = res_ax.get_legend_handles_labels()
                if handles_res:
                    res_ax.legend(handles_res, labels_res, loc='upper right', fontsize=9, frameon=False)
    
    # Remove x-axis labels for top row
    ax_twa27a_main.set_xlabel('')
    ax_twa27a_res.set_xlabel('')
    
    # Remove y-axis labels for right column
    ax_twa27a_res.set_ylabel('')
    ax_twa28_res.set_ylabel('')
    
    # move y-ticks to the right of the plot
    ax_twa27a_res.yaxis.set_ticks_position('right')
    ax_twa28_res.yaxis.set_ticks_position('right')
    
    ax_twa27a_main.set_ylabel('Pressure (bar)', fontsize=10)
    # add a,b,c,d labels to each panel for easy reference
    panels = ['a', 'b', 'c', 'd']
    ax_twa27a_main.text(0.02, 0.95, panels[0], transform=ax_twa27a_main.transAxes, fontsize=12, fontweight='bold',
                        ha='left', va='top', path_effects=pe_white)
    ax_twa27a_res.text(0.06, 0.95, panels[1], transform=ax_twa27a_res.transAxes, fontsize=12, fontweight='bold',
                        ha='left', va='top', path_effects=pe_white)
    ax_twa28_main.text(0.02, 0.95, panels[2], transform=ax_twa28_main.transAxes, fontsize=12, fontweight='bold',    
                        ha='left', va='top', path_effects=pe_white)
    ax_twa28_res.text(0.06, 0.95, panels[3], transform=ax_twa28_res.transAxes, fontsize=12, fontweight='bold',
                        ha='left', va='top', path_effects=pe_white)
    
    # Save figure
    fig_name = path_figures / 'fig_PTs_two_panels.pdf'
    fig.savefig(fig_name, bbox_inches='tight', dpi=300)
    print(f' --> Saved {fig_name}')
    plt.close(fig)

if __name__ == "__main__":
    main()