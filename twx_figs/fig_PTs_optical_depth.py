""" 
Generate a model for G235+G395 with the best-fit parameters from G235 alone 
Inspect the residuals, disk emission?

date: 2024-09-17
"""
import pathlib
import numpy as np
import os
import matplotlib.pyplot as plt
import h5py
from matplotlib.backends.backend_pdf import PdfPages
import copy

from retrieval_base.retrieval import Retrieval
import retrieval_base.auxiliary_functions as af
from retrieval_base.config import Config
import seaborn as sns


def calculate_optical_depth(pressure: np.ndarray, temperature: np.ndarray, total_tau: np.ndarray) -> np.ndarray:
    """
    Calculate wavelength-averaged optical depth from total_tau array.
    
    Parameters:
    -----------
    pressure : np.ndarray
        Pressure array
    temperature : np.ndarray 
        Temperature array
    total_tau : np.ndarray
        Total tau array with shape (n_wave, n_species, n_layers)
        
    Returns:
    --------
    optical_depth : np.ndarray
        Wavelength-averaged optical depth for each layer
    """
    # Average over wavelength and species dimensions
    optical_depth = np.nanmean(total_tau, axis=(0, 1))
    return optical_depth


def save_pt_data_h5(file_path: pathlib.Path, pressure: np.ndarray, temperature_envelopes: np.ndarray, 
                   optical_depth: np.ndarray, icf: np.ndarray, logg: float, target: str, run: str) -> None:
    """Save PT data to H5 file for faster loading."""
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('pressure', data=pressure)
        f.create_dataset('temperature_envelopes', data=temperature_envelopes)
        f.create_dataset('optical_depth', data=optical_depth)
        f.create_dataset('icf', data=icf)
        f.create_dataset('logg', data=logg)
        f.attrs['target'] = target
        f.attrs['run'] = run


def load_pt_data_h5(file_path: pathlib.Path) -> tuple:
    """Load PT data from H5 file."""
    with h5py.File(file_path, 'r') as f:
        pressure = f['pressure'][:]
        temperature_envelopes = f['temperature_envelopes'][:]
        optical_depth = f['optical_depth'][:]
        icf = f['icf'][:]
        logg = f['logg'][()]
    return pressure, temperature_envelopes, optical_depth, icf, logg


path = pathlib.Path(af.get_path())
path_figures = pathlib.Path('/home/dario/phd/twa2x_paper/figures')

config_file = 'config_jwst.txt'
w_set = 'NIRSpec'


def check_dir(target: str) -> None:
    """Change to target directory if not already there."""
    cwd = os.getcwd()
    if target not in cwd:
        os.chdir(f'{path}/{target}')
        print(f'Changed directory to {target}')


def get_bestfit_params(target: str, run: str) -> dict:
    """Get best-fit parameters for a given target and run."""
    check_dir(target)
    
    conf = Config(path=path, target=target, run=run)(config_file)        
        
    ret = Retrieval(conf=conf, evaluation=False)
    bestfit_params, _ = ret.PMN_analyze()
    bestfit_params_dict = dict(zip(ret.Param.param_keys, bestfit_params))
    return bestfit_params_dict


runs = dict(
    TWA27A='freeslab_lbl10_G1G2G3_0',
    TWA28='freeslab_lbl10_G1G2G3_0',
)

colors = dict(
    TWA28={'data':'k', 
           'model':'#e89c4b',
           'crires': 'brown'},
    TWA27A={'data':'#733b27',
            'model':'seagreen'}
)


def get_PT(path: pathlib.Path, target: str, run: str, config_file: str = 'config_jwst.txt', cache: bool = True):
    """
    Get pressure-temperature profile data with H5 caching for fast loading.
    
    Returns either the cached data or calculates new data and saves to H5.
    """
    envelopes_dir = path / target / f'retrieval_outputs/{run}/test_data' / 'envelopes'
    envelopes_dir.mkdir(parents=True, exist_ok=True)

    # Use H5 format for faster I/O
    pt_data_file = envelopes_dir / 'pt_optical_depth_data.h5'

    if cache and pt_data_file.exists():
        print(f' --> Found {pt_data_file}')
        pressure, temperature_envelopes, optical_depth, icf, logg = load_pt_data_h5(pt_data_file)
        print(f' --> Loaded PT data with temperature shape {temperature_envelopes.shape}')
        print(f' --> logg = {logg:.2f}')
        return pressure, temperature_envelopes, optical_depth, icf, logg
    else:
        print(f' Calculating PT envelopes and optical depth for {run}')
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
        logg = np.median(logg_posterior)
        
        # Get optical depth data
        total_tau = ret.pRT_atm['NIRSpec'].atm[0].total_tau[0]  # shape (n_wave, n_species, n_layers)
        optical_depth = calculate_optical_depth(ret.PT.pressure, ret.PT.temperature_envelopes[3,:], total_tau)
        
        # Copy integrated contribution emission
        ret.copy_integrated_contribution_emission()
        icf = ret.PT.int_contr_em['NIRSpec']
        
        # Save to H5 file
        save_pt_data_h5(pt_data_file, ret.PT.pressure, ret.PT.temperature_envelopes, 
                       optical_depth, icf, logg, target, run)
        print(f' --> Saved {pt_data_file}')
        
        # Save VMRs
        ret.Chem.get_VMRs_posterior(save_to=envelopes_dir)
        
        return ret.PT.pressure, ret.PT.temperature_envelopes, optical_depth, icf, logg


def plot_envelopes(y_coord: str, pressure: np.ndarray, temperature_envelopes: np.ndarray, 
                  optical_depth: np.ndarray, ax=None, cf=None, **kwargs):
    """
    Plot temperature envelopes as a function of pressure or optical depth.
    
    Parameters:
    -----------
    y_coord : str
        Either 'pressure' or 'optical_depth'
    pressure : np.ndarray
        Pressure array
    temperature_envelopes : np.ndarray
        Temperature envelopes array with shape (7, n_layers)
    optical_depth : np.ndarray
        Optical depth array
    """
    ax = ax or plt.gca()
    assert len(temperature_envelopes.shape) > 1, f'Expected 2D array, got {temperature_envelopes.shape}'
    assert temperature_envelopes.shape[0] == 7, f'Expected 7 envelopes, got {temperature_envelopes.shape[0]}'
    
    color = kwargs.pop('color', 'brown')
    alpha = kwargs.pop('alpha', 0.2)
    label = kwargs.pop('label', '')
    
    # Choose y-axis coordinate
    if y_coord == 'pressure':
        y = pressure
    elif y_coord == 'optical_depth':
        y = optical_depth
    else:
        raise ValueError(f"y_coord must be 'pressure' or 'optical_depth', got {y_coord}")
    
    # Plot temperature envelopes
    for i in range(3):
        ax.fill_betweenx(y, 
                        temperature_envelopes[i,:],
                        temperature_envelopes[-(i+1),:], 
                        color=color, alpha=alpha, lw=0, 
                        label=label if i == 0 else '',
                        )
    ax.plot(temperature_envelopes[3,:], y, color=color, lw=1.2, ls=kwargs.pop('ls', '-'), alpha=0.75)

    if cf is not None:
        fill_cf = kwargs.pop('fill_cf', False)
        ax_cf = ax.twiny()
        ls = kwargs.pop('ls_cf', ':')
        lw = kwargs.pop('lw_cf', 2.5)
        ax_cf.plot(cf, y, color=color, lw=lw, ls=ls, alpha=0.75)
        ax_cf.set_xticks([])
        ax_cf.set_yticks([])
        ax_cf.set_xlim(0, np.max(cf)*4.5)
        if fill_cf:
            ax_cf.fill_between(cf, y, color=color, alpha=0.05)
        
    return ax


# Choose y-axis coordinate: either 'pressure' or 'optical_depth'
y_coordinate = 'optical_depth'  # Change this to 'pressure' if needed

fig, ax = plt.subplots(1, 1, figsize=(4, 4), tight_layout=True)


def plot_crires(ax, y_coord: str):
    """Plot CRIRES data for TWA28."""
    run_full = 'final_full'
    p, t_env, opt_depth, cf, log_g = get_PT(path, 'TWA28', run=run_full, config_file='config_freechem.txt')
    
    ax = plot_envelopes(y_coord, p, t_env, opt_depth, ax=ax, cf=cf, 
                       color=colors['TWA28']['crires'], alpha=0.2, 
                       label='TWA 28\n' + r'(CRIRES$^{+}$)', fill_cf=True,
                       ls='--', ls_cf='--', lw_cf=1.0)
    return ax


targets_params = dict(
    TWA27A={'teff': (2430, 20)},
    TWA28={'teff': (2382, 42)},
)

for t, target in enumerate(runs.keys()):
    Teff = targets_params[target]['teff']
    label_teff = f'{Teff[0]:.0f} K'
    ax.axvline(Teff[0], color=colors[target]['model'], ls=':', lw=2, zorder=-10, alpha=0.8, label=label_teff)
    
    if target == 'TWA28':
        plot_crires(ax, y_coordinate)
        
    # Get PT data with optical depth
    p, t_env, opt_depth, cf, logg = get_PT(path, target, runs[target], cache=True)
    
    print(f'Target {target}: temperature shape = {t_env.shape}')
    ax = plot_envelopes(y_coordinate, p, t_env, opt_depth, ax=ax, cf=cf, 
                       color=colors[target]['model'], alpha=0.4, fill_cf=True,
                       label='TWA ' + target.replace('TWA', ''),
                       ls_cf='-', lw_cf=1.0)


# Set axis properties based on y-coordinate choice
if y_coordinate == 'pressure':
    y_values = p
    ylim = (np.max(y_values), np.min(y_values))
    yscale = 'log'
    ylabel = 'Pressure / bar'
elif y_coordinate == 'optical_depth':
    y_values = opt_depth
    ylim = (np.min(y_values), np.max(y_values))
    yscale = 'log'
    ylabel = r'Optical depth $\tau$'

ax.set(yscale=yscale, ylim=ylim, ylabel=ylabel, xlabel='Temperature (K)')
ax.set_xlim(None, 5000)

# Configure legend
ax.legend(prop={'size': 14, 'weight': 'bold'}, loc='upper right')
handles, labels = ax.get_legend_handles_labels()
labels_sort = ['2430 K', 'TWA 27A', '2382 K', 'TWA 28', 'TWA 28\n(CRIRES$^{+}$)']
legend_dict = dict(zip(labels, handles))
handles_sort = [legend_dict[label] for label in labels_sort]
leg = ax.legend(handles_sort, labels_sort, 
                prop={'size': 10, 'weight': 'bold'}, 
                loc=(0.54, 0.6),
                frameon=False, ncol=1)

for p, patch in enumerate(leg.get_patches()):
    patch.set_alpha(0.70)

# Save figure
y_coord_suffix = y_coordinate.replace('_', '')
fig_name = path_figures / f'fig_PTs_{y_coord_suffix}.pdf'
fig.savefig(fig_name, bbox_inches='tight')
print(f' --> Saved {fig_name}')
plt.close(fig)