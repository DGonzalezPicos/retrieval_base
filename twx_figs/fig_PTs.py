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
import sys

# Add Sonora Diamondback path to sys.path
sonora_path = '/home/dario/phd/SonoraDiamondBack/pressure-temperature_profiles'
if sonora_path not in sys.path:
    sys.path.insert(0, sonora_path)

from retrieval_base.retrieval import Retrieval
import retrieval_base.auxiliary_functions as af
from retrieval_base.config import Config
import seaborn as sns

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
    'linewidth': 2,          # Line width
    'alpha': 0.7,           # Transparency
    'zorder': 5             # Plotting order
}

runs = dict(
    TWA27A='freeslab_lbl10_G1G2G3_1',
    TWA28='freeslab_lbl10_G1G2G3_1',
)

colors = dict(
    TWA28={
        'data': 'k', 
        'model': '#D55E00',
        'crires': 'royalblue',
    },
    TWA27A={
        'data': 'gray',
        'model': '#009E73',
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

def get_sonora_pt_profile(sonora_data_dir: pathlib.Path, 
                          target_teff: int = 2400,
                          target_logg: float = 4.0,
                          target_metallicity: float = 0.0,
                          target_c_o_ratio: float = 1.0) -> tuple:
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
        target_gravity = 10.0**target_logg
        
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
        if sonora_config['fsed'] is None:
            print("   Clouds = No clouds")
        else:
            print(f"   Clouds = fsed={sonora_config['fsed']}")
        
        # Target parameters for Sonora model
        target_params = {
            'teff': target_teff,
            'gravity': target_gravity,
            'fsed': sonora_config['fsed'],  # Use config value
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

def main():
    """Main plotting function."""
    # Set pressure scaling mode
    scaling_mode = 'none'  # Options: 'multiply', 'divide', 'none'
    
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), tight_layout=True)
    
    # Load Sonora Diamondback PT profiles
    sonora_data_dir = pathlib.Path(sonora_config['data_dir'])
    
    # List available models for debugging
    print("\n" + "="*60)
    print("SEARCHING FOR SONORA DIAMONDBACK MODELS")
    print("="*60)
    list_available_sonora_models(sonora_data_dir)
    print("="*60)
    
    # Define multiple Sonora models to search for
    sonora_models = [
        {'teff': 2400, 'logg': 4.0, 'metallicity': 0.0, 'c_o_ratio': 1.0, 'label': 'T=2400K, log(g)=4.0'},
        {'teff': 2400, 'logg': 4.5, 'metallicity': 0.0, 'c_o_ratio': 1.0, 'label': 'T=2400K, log(g)=4.5'},
        {'teff': 2200, 'logg': 4.0, 'metallicity': 0.0, 'c_o_ratio': 1.0, 'label': 'T=2200K, log(g)=4.0'},
        {'teff': 2600, 'logg': 4.0, 'metallicity': 0.0, 'c_o_ratio': 1.0, 'label': 'T=2600K, log(g)=4.0'},
    ]
    
    found_sonora_models = find_multiple_sonora_models(sonora_data_dir, sonora_models)
    
    if found_sonora_models:
        print(f"✓ Found {len(found_sonora_models)} Sonora Diamondback models")
    else:
        print("⚠ No Sonora Diamondback models found")
    
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

    # Plot Sonora Diamondback models if available
    for i, (sonora_p, sonora_t, params) in enumerate(found_sonora_models):
        # Scale pressure if needed
        sonora_y = scale_pressure(sonora_p, params['logg'], scaling_mode)
        
        # Plot Sonora model with specified styling
        ax.plot(sonora_t, sonora_y, color=sonora_config['color'], 
                lw=sonora_config['linewidth'], ls='-', 
                alpha=sonora_config['alpha'], 
                label=f'Sonora: {params["label"]}', 
                zorder=sonora_config['zorder'])
        
        print(f"✓ Plotted Sonora model: {params['label']}")
    
    if not found_sonora_models:
        print("⚠ Sonora Diamondback models not plotted (unavailable)")

    # Set plot limits and labels
    y_scaled = scale_pressure(p, logg, scaling_mode)
    ylim = (np.max(y_scaled), np.min(y_scaled))
    ylabel = get_ylabel(scaling_mode)
    
    ax.set(yscale='log', ylim=ylim, ylabel=ylabel, xlabel='Temperature (K)')
    ax.set_xlim(None, 5000)
    
    # Configure legend
    handles, labels = ax.get_legend_handles_labels()
    
    # Update legend labels to include Sonora models
    if found_sonora_models:
        # Create labels for all Sonora models
        sonora_labels = [f'Sonora: {params["label"]}' for _, _, params in found_sonora_models]
        labels_sort = ['2430 K', 'TWA 27A', '2382 K', 'TWA 28', 'TWA 28\n(CRIRES$^{+}$)'] + sonora_labels
    else:
        labels_sort = ['2430 K', 'TWA 27A', '2382 K', 'TWA 28', 'TWA 28\n(CRIRES$^{+}$)']
    
    legend_dict = dict(zip(labels, handles))
    handles_sort = [legend_dict[label] for label in labels_sort if label in legend_dict]
    
    # Adjust legend position and size for better readability
    leg = ax.legend(handles_sort, labels_sort, 
                    prop={'size': 8, 'weight': 'bold'}, 
                    loc=(0.54, 0.5),
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