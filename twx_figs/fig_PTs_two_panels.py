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
import sys


# Add Sonora Diamondback path to sys.path
sonora_path = '/home/dario/phd/SonoraDiamondBack/pressure-temperature_profiles'
if sonora_path not in sys.path:
    sys.path.insert(0, sonora_path)

from retrieval_base.retrieval import Retrieval
import retrieval_base.auxiliary_functions as af
from retrieval_base.config import Config
import seaborn as sns
from matplotlib import patheffects as path_effects
pe_white = [path_effects.withStroke(linewidth=2.0, foreground='w')]

# set plt to default style
plt.style.use('default')
# increase line width of axis for thicker axes lines
plt.rcParams.update({'axes.linewidth': 1.5})

# Import Sonora Diamondback functions
try:
    from plot_chemistry_pt_profiles import load_pt_profile, find_closest_parameters, parse_filename_parameters
    SONORA_AVAILABLE = True
    print("✓ Successfully imported Sonora Diamondback functions")
except ImportError as e:
    SONORA_AVAILABLE = False
    print(f"⚠ Warning: Could not import Sonora Diamondback functions: {e}")
    print("  Sonora Diamondback models will not be plotted")

# Configuration
path = pathlib.Path(af.get_path())
path_figures = pathlib.Path('/home/dario/phd/twa2x_paper/figures')
config_file = 'config_jwst.txt'
w_set = 'NIRSpec'

# Sonora Diamondback configuration
sonora_config = {
    'data_dir': '/home/dario/phd/SonoraDiamondBack/pressure-temperature_profiles',
    'teff': 2400,           # Effective temperature in K
    'logg': 4.0,            # log10 surface gravity
    'metallicity': 0.0,     # Solar metallicity [M/H]
    'c_o_ratio': 1.0,       # Carbon-to-oxygen ratio
    'fsed': None,           # No clouds
    'color': 'magenta',      # Plot color
    'linestyle': '--',
    'linewidth': 2,          # Line width
    'alpha': 0.7,           # Transparency
    'zorder': 5             # Plotting order
}
sonora_config['label'] = f'Sonora: T={sonora_config["teff"]}K, log(g)={sonora_config["logg"]:.1f}'

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

def get_sonora_pt_profile(sonora_data_dir: pathlib.Path, 
                          target_teff: int = 2400,
                          target_logg: float = 4.0,
                          target_metallicity: float = 0.0,
                          target_c_o_ratio: float = 1.0,
                          target_fsed: float = None) -> tuple:
    """
    Load Sonora Diamondback PT profile for specified parameters.
    
    Parameters:
        sonora_data_dir: Directory containing Sonora PT profile files
        target_teff: Target effective temperature in K
        target_logg: Target log10 surface gravity
        target_metallicity: Target metallicity [M/H]
        target_c_o_ratio: Target carbon-to-oxygen ratio
        
    Returns:
        tuple: (pressure, temperature) or (None, None) if not available
    """
    if not SONORA_AVAILABLE:
        return None, None
        
    try:
        # Convert log(g) to gravity in m/s²
        target_gravity = (10.0**target_logg) / 100.0 # cm/s² --> m/s²
        
        # Find PT profile files
        pt_files = list(sonora_data_dir.glob("*.pt"))
        
        if not pt_files:
            print(f"⚠ No PT profile files found in {sonora_data_dir}")
            return None, None
        
        print(f"🔍 Searching for Sonora model with parameters:")
        print(f"   T_eff = {target_teff} K")
        print(f"   log(g) = {target_logg:.1f} (g = {target_gravity:.0f} m/s²)")
        print(f"   [M/H] = {target_metallicity:.1f}")
        print(f"   C/O = {target_c_o_ratio:.1f}")
        if target_fsed is None:
            print("   Clouds = No clouds")
        else:
            print(f"   Clouds = fsed={target_fsed}")
        
        # Target parameters for Sonora model
        target_params = {
            'teff': target_teff,
            'gravity': target_gravity,
            'fsed': target_fsed,  # Use config value
            'metallicity': target_metallicity,
            'c_o_ratio': target_c_o_ratio
        }
        
        # Tolerance for parameter matching
        tolerance = {
            'teff': 50,        # ±50K
            'gravity': 10,     # ±10 m/s²
            'fsed': 1,         # ±1 (but we want None for no clouds)
            'metallicity': 0.1,  # ±0.1
            'c_o_ratio': 0.1     # ±0.1
        }
        
        # Find closest matching file
        closest_file = find_closest_parameters(pt_files, target_params, tolerance)
        file_params = parse_filename_parameters(closest_file.name)
        
        print(f"✓ Found Sonora model: {closest_file.name}")
        print(f"  Actual parameters: T={file_params['teff']}K, g={file_params['gravity']:.0f} m/s², "
              f"m={file_params['metallicity']:.1f}, C/O={file_params['c_o_ratio']:.1f}")
        if file_params['fsed'] is not None:
            print(f"  Cloud parameter: fsed={file_params['fsed']}")
        else:
            print("  Cloud parameter: No clouds (nc)")
        
        # Load PT profile
        pressure, temperature = load_pt_profile(closest_file)
        
        return pressure, temperature
        
    except Exception as e:
        print(f"⚠ Error loading Sonora PT profile: {e}")
        return None, None

def find_multiple_sonora_models(sonora_data_dir: pathlib.Path, 
                               target_params_list: list) -> list:
    """
    Find multiple Sonora Diamondback PT profiles for different parameter sets.
    
    Parameters:
        sonora_data_dir: Directory containing Sonora PT profile files
        target_params_list: List of parameter dictionaries
        
    Returns:
        list: List of tuples (pressure, temperature, params) for found models
    """
    if not SONORA_AVAILABLE:
        return []
        
    found_models = []
    
    for params in target_params_list:
        try:
            p, t = get_sonora_pt_profile(
                sonora_data_dir,
                target_teff=params['teff'],
                target_logg=params['logg'],
                target_metallicity=params['metallicity'],
                target_c_o_ratio=params['c_o_ratio']
            )
            if p is not None and t is not None:
                found_models.append((p, t, params))
        except Exception as e:
            print(f"⚠ Error loading Sonora model for params {params}: {e}")
            continue
    
    return found_models

def list_available_sonora_models(sonora_data_dir: pathlib.Path) -> None:
    """
    List available Sonora Diamondback models in the directory for debugging.
    
    Parameters:
        sonora_data_dir: Directory containing Sonora PT profile files
    """
    if not SONORA_AVAILABLE:
        print("⚠ Sonora functions not available")
        return
        
    try:
        pt_files = list(sonora_data_dir.glob("*.pt"))
        
        if not pt_files:
            print(f"⚠ No PT profile files found in {sonora_data_dir}")
            return
        
        print(f"📁 Found {len(pt_files)} Sonora PT profile files:")
        
        # Parse and group files by parameters
        model_groups = {}
        for filepath in pt_files[:20]:  # Show first 20 files
            try:
                params = parse_filename_parameters(filepath.name)
                key = (params['teff'], params['gravity'], params['metallicity'])
                if key not in model_groups:
                    model_groups[key] = []
                model_groups[key].append(filepath.name)
            except ValueError:
                continue
        
        # Display grouped models
        for (teff, gravity, metallicity), files in sorted(model_groups.items()):
            logg = np.log10(gravity)
            print(f"   T={teff}K, log(g)={logg:.1f}, [M/H]={metallicity:.1f}: {len(files)} files")
            
        if len(pt_files) > 20:
            print(f"   ... and {len(pt_files) - 20} more files")
            
    except Exception as e:
        print(f"⚠ Error listing Sonora models: {e}")

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
    print(f'temp_comp.shape: {np.shape(temp_comp)}')
    print(f'temp_ref.shape: {np.shape(temp_ref)}')
    
    temp_comp_median = temp_comp[3, :] if len(np.shape(temp_comp)) > 1 else temp_comp
    temp_ref_median = temp_ref[3, :] if len(np.shape(temp_ref)) > 1 else temp_ref
    
    print(f'temp_comp_median.shape: {np.shape(temp_comp_median)}')
    print(f'temp_ref_median.shape: {np.shape(temp_ref_median)}')
    print(f'pressure_comp.shape: {np.shape(pressure_comp)} ')
    print(f'pressure.shape: {np.shape(pressure)}')
    
    if len(temp_comp_median) != len(temp_ref_median):
        assert pressure_comp is not None, 'pressure_comp must be provided if temp_comp and temp_ref have different lengths'
        temp_comp_int = np.interp(pressure, pressure_comp, temp_comp_median)
        temp_diff = temp_comp_int - temp_ref_median
    else:
        temp_diff = temp_comp_median - temp_ref_median
    
    # Plot residuals
    ax.plot(temp_diff, pressure, color=color, lw=1.5, alpha=0.8, label=label,
            ls=kwargs.get('ls', '-'))
    
    # Add zero line
    ax.axvline(0, color='gray', lw=0.8, alpha=0.5, ls='--')
    
    return ax

def main():
    """Main plotting function."""
    # Load Sonora Diamondback PT profiles
    sonora_data_dir = pathlib.Path(sonora_config['data_dir'])
    
    # List available models for debugging
    print("\n" + "="*60)
    print("SEARCHING FOR SONORA DIAMONDBACK MODELS")
    print("="*60)
    list_available_sonora_models(sonora_data_dir)
    print("="*60)


    
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
        
        # plot sonora cloudy and sonora cloudless
        
        # for fsed in [None, 8.0]:
        for fsed in [None]:
            sonora_config['fsed'] = fsed
            # Plot Sonora Diamondback model for this target if available
            target_sonora_model = get_sonora_pt_profile(sonora_data_dir, 
                                                        target_teff=sonora_config['teff'],
                                                        target_logg=sonora_config['logg'], 
                                                        target_metallicity=sonora_config['metallicity'],
                                                        target_c_o_ratio=sonora_config['c_o_ratio'],
                                                        target_fsed=fsed)
            
            if target_sonora_model is not None:
                sonora_p, sonora_t = target_sonora_model
                cloudy_label = 'SDB\ncloudless' if fsed is None else 'SDB\ncloudy'

                # Plot Sonora model on main panel
                main_ax.plot(sonora_t, sonora_p, color=sonora_config['color'], 
                            lw=sonora_config['linewidth'], ls=sonora_config['linestyle'], 
                            alpha=sonora_config['alpha'], 
                            # label=sonora_config["label"], 
                            label=cloudy_label,
                            zorder=sonora_config['zorder'])
                
                # Plot Sonora residuals on residuals panel
                if target in reference_profiles:
                    plot_temperature_residuals(p, reference_profiles[target], sonora_t, 
                                            pressure_comp=sonora_p, ax=res_ax,
                                            color=sonora_config['color'], 
                                            label=cloudy_label,
                                            ls=sonora_config['linestyle'])
                
                print(f"✓ Plotted Sonora model for {target}: {sonora_config['label']}")
        
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
            main_ax.legend(handles, labels, loc=(0.52,0.42), fontsize=9, frameon=False)
        elif target == 'TWA28':
            handles, labels = main_ax.get_legend_handles_labels()
            main_ax.legend(handles, labels, loc=(0.52,0.32), fontsize=9, frameon=False)
        
        # Configure legend for residuals panel
        add_legend_residuals = False
        if add_legend_residuals:
            if len(runs[target]) > 1 or target == 'TWA28':
                handles_res, labels_res = res_ax.get_legend_handles_labels()
                if handles_res:
                    res_ax.legend(handles_res, labels_res, loc='upper right', fontsize=9, frameon=False)
    
    # Print summary of Sonora models found and plotted
    if target_sonora_model is not None:
        print(f"✓ Plotted Sonora model for {target}: {sonora_config['label']}")
    else:
        print(f"⚠ No Sonora Diamondback model was found for {target}")
    
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
    # panels = ['a', 'b', 'c', 'd']
    # ax_twa27a_main.text(0.02, 0.95, panels[0], transform=ax_twa27a_main.transAxes, fontsize=12, fontweight='bold',
    #                     ha='left', va='top', path_effects=pe_white)
    # ax_twa27a_res.text(0.06, 0.95, panels[1], transform=ax_twa27a_res.transAxes, fontsize=12, fontweight='bold',
    #                     ha='left', va='top', path_effects=pe_white)
    # ax_twa28_main.text(0.02, 0.95, panels[2], transform=ax_twa28_main.transAxes, fontsize=12, fontweight='bold',    
    #                     ha='left', va='top', path_effects=pe_white)
    # ax_twa28_res.text(0.06, 0.95, panels[3], transform=ax_twa28_res.transAxes, fontsize=12, fontweight='bold',
    #                     ha='left', va='top', path_effects=pe_white)
    panels = ['a', 'b']
    ax_twa27a_main.text(-0.30, 1.06, panels[0], transform=ax_twa27a_main.transAxes, fontsize=12, fontweight='bold',
                        ha='left', va='top', path_effects=pe_white)
    ax_twa28_main.text(-0.30, 1.06, panels[1], transform=ax_twa28_main.transAxes, fontsize=12, fontweight='bold',
                        ha='left', va='top', path_effects=pe_white)
   
    # Save figure
    fig_name = path_figures / 'fig_PTs_two_panels.pdf'
    fig.savefig(fig_name, bbox_inches='tight', dpi=300)
    print(f' --> Saved {fig_name}')
    plt.close(fig)

if __name__ == "__main__":
    main()