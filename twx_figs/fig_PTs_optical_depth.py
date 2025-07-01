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
from datetime import datetime
# increase font size
# plt.style.use('/home/dario/phd/retsupjup/GQLupB/paper/gqlupb.mplstyle')
# pdf pages
from matplotlib.backends.backend_pdf import PdfPages
import copy

from retrieval_base.retrieval import Retrieval
import retrieval_base.auxiliary_functions as af
from retrieval_base.config import Config
import seaborn as sns

def pressure_to_altitude(P, T, log_g):
    return P * 10.0**log_g
    # return P * np.e**(10.0**log_g * 1e-2)

def pressure_to_altitude_old(P, T, log_g, mu=2.3):
    """
    Convert pressure to altitude using the hydrostatic equation.
    Assumes an isothermal atmosphere for simplicity.
    
    Parameters:
    P : array-like
        Pressure values in bar.
    T : float
        Temperature in Kelvin (assumed constant for simplicity).
    log_g : float
        Surface gravity in m/s^2.
    mu : float, optional
        Mean molecular weight in atomic mass units (default is 2.3 for H2-dominated atmosphere).
    
    Returns:
    z : array-like
        Corresponding altitude values in km.
    """
    k_B = 1.380649e-23  # Boltzmann constant (J/K)
    m_H = 1.6735575e-27 # Hydrogen atom mass (kg)
    mu_kg = mu * m_H    # Mean molecular weight in kg
    # Convert log_g (log10[cm/s²]) to g in m/s²
    g = (10.0**log_g) / 100  # Convert cm/s² to m/s²
    
    H = k_B * T / (mu_kg * g)  # Now using correct SI units (m/s²)
    P0 = np.max(P)
    # sort pressure from high to low
    P = np.sort(P)[::-1]
    print(f'P0 = {P0:.2e}, P = {P[0]:.2e}')
    z = H * np.log(P0 / P)  # Altitude in meters
    return z / 1000  # Convert to km

def save_pt_data_h5(filepath: pathlib.Path, pressure: np.ndarray, temperature_envelopes: np.ndarray, 
                   integrated_contribution: np.ndarray, logg_posterior: np.ndarray, 
                   optical_depth: np.ndarray = None, target: str = '', run: str = ''):
    """
    Save PT profile data to HDF5 file with proper metadata and structure.
    
    Parameters:
    -----------
    filepath : pathlib.Path
        Path to save the HDF5 file
    pressure : np.ndarray
        Pressure grid
    temperature_envelopes : np.ndarray  
        Temperature envelopes (7 x n_pressure)
    integrated_contribution : np.ndarray
        Integrated contribution function
    logg_posterior : np.ndarray
        Log g posterior samples
    optical_depth : np.ndarray, optional
        Optical depth profile
    target : str
        Target name for metadata
    run : str
        Run name for metadata
    """
    with h5py.File(filepath, 'w') as f:
        # Add metadata
        f.attrs['created_date'] = datetime.now().isoformat()
        f.attrs['target'] = target
        f.attrs['run'] = run
        f.attrs['n_pressure_points'] = len(pressure)
        f.attrs['n_envelopes'] = temperature_envelopes.shape[0]
        f.attrs['logg_median'] = np.median(logg_posterior)
        f.attrs['logg_std'] = np.std(logg_posterior)
        
        # Save main datasets with clear names and descriptions
        dset_pressure = f.create_dataset('pressure', data=pressure)
        dset_pressure.attrs['units'] = 'bar'
        dset_pressure.attrs['description'] = 'Pressure grid'
        
        dset_temp = f.create_dataset('temperature_envelopes', data=temperature_envelopes)
        dset_temp.attrs['units'] = 'K'
        dset_temp.attrs['description'] = 'Temperature envelopes: [16%, 84%], [2.5%, 97.5%], [0.15%, 99.85%], median'
        dset_temp.attrs['envelope_order'] = '16%, 84%, 2.5%, 97.5%, 0.15%, 99.85%, median'
        
        dset_icf = f.create_dataset('integrated_contribution_function', data=integrated_contribution)
        dset_icf.attrs['units'] = 'dimensionless'
        dset_icf.attrs['description'] = 'Integrated contribution to emission'
        
        dset_logg = f.create_dataset('logg_posterior', data=logg_posterior)
        dset_logg.attrs['units'] = 'log10(cm/s^2)'
        dset_logg.attrs['description'] = 'Surface gravity posterior samples'
        
        # Save optical depth if available
        if optical_depth is not None:
            dset_od = f.create_dataset('optical_depth', data=optical_depth)
            dset_od.attrs['units'] = 'dimensionless'
            dset_od.attrs['description'] = 'Mean optical depth profile'
            f.attrs['has_optical_depth'] = True
        else:
            f.attrs['has_optical_depth'] = False
            
    print(f' --> Saved PT data to {filepath}')


def load_pt_data_h5(filepath: pathlib.Path):
    """
    Load PT profile data from HDF5 file.
    
    Parameters:
    -----------
    filepath : pathlib.Path
        Path to the HDF5 file
        
    Returns:
    --------
    tuple
        (pressure, temperature_envelopes, integrated_contribution, logg_median, optical_depth)
        optical_depth is None if not available in file
    """
    with h5py.File(filepath, 'r') as f:
        pressure = f['pressure'][:]
        temperature_envelopes = f['temperature_envelopes'][:]
        integrated_contribution = f['integrated_contribution_function'][:]
        logg_posterior = f['logg_posterior'][:]
        logg_median = np.median(logg_posterior)
        
        # Handle optional optical depth
        optical_depth = None
        if f.attrs.get('has_optical_depth', False) and 'optical_depth' in f:
            optical_depth = f['optical_depth'][:]
            
        print(f' --> Loaded PT data from {filepath}')
        print(f'     Target: {f.attrs.get("target", "unknown")}')
        print(f'     Run: {f.attrs.get("run", "unknown")}')
        print(f'     Temperature envelopes shape: {temperature_envelopes.shape}')
        print(f'     Log g median: {logg_median:.2f}')
        print(f'     Has optical depth: {optical_depth is not None}')
        
    return pressure, temperature_envelopes, integrated_contribution, logg_median, optical_depth

def inspect_pt_h5_file(filepath: pathlib.Path):
    """
    Inspect the contents of a PT profile HDF5 file for debugging.
    
    Parameters:
    -----------
    filepath : pathlib.Path
        Path to the HDF5 file
    """
    print(f"\n=== Inspecting HDF5 file: {filepath} ===")
    
    try:
        with h5py.File(filepath, 'r') as f:
            print("\nFile attributes:")
            for key, value in f.attrs.items():
                print(f"  {key}: {value}")
                
            print("\nDatasets:")
            for key in f.keys():
                dataset = f[key]
                print(f"  {key}: shape={dataset.shape}, dtype={dataset.dtype}")
                if hasattr(dataset, 'attrs'):
                    for attr_key, attr_value in dataset.attrs.items():
                        print(f"    {attr_key}: {attr_value}")
                        
    except Exception as e:
        print(f"Error reading file: {e}")
    
    print("=" * 50)

# import config_jwst as conf

path = pathlib.Path(af.get_path())
# path_figures = pathlib.Path('/home/dario/phd/retrieval_base/twx_figs')
path_figures = pathlib.Path('/home/dario/phd/twa2x_paper/figures')

config_file = 'config_jwst.txt'
# target = 'TWA28'
w_set='NIRSpec'


def check_dir(target):
    cwd = os.getcwd()
    if target not in cwd:
        os.chdir(f'{path}/{target}')
        print(f'Changed directory to {target}')

def get_bestfit_params(target,run):
    
    check_dir(target)
    
    conf = Config(path=path, target=target, run=run)(config_file)        
        
    ret = Retrieval(
        conf=conf, 
        evaluation=False
        )

    bestfit_params, _ = ret.PMN_analyze()
    bestfit_params_dict = dict(zip(ret.Param.param_keys, bestfit_params))
    return bestfit_params_dict
# run with both gratings
# run = 'lbl15_K2'
# run = 'lbl15_G2G3_3'
# run = 'lbl12_G1G2G3_fastchem_1'

# runs = dict(
#     # TWA27A=['lbl11_G1G2G3_fastchem_0'],
#     TWA28=['lbl11_G1G2G3_fastchem_0', 'lbl11_G2G3_fastchem_0', 'lbl11_G2_fastchem_0'],
#             )

runs = dict(
    TWA27A='freeslab_lbl10_G1G2G3_1',
    TWA28='freeslab_lbl10_G1G2G3_1',
            )
colors = dict(TWA28={'data':'k', 
                    #  'model':'orange',
                    'model':'#e89c4b',
                     'crires': 'brown'
                     },
                    # 'crires': '#CC79A7'
              TWA27A={'data':'#733b27',
                    #   'model':'#0a74da',
                    'model':'seagreen',
                      })

def get_PT(path, target, run, config_file='config_jwst.txt', cache=True):
    """
    Get PT profile data, either from cached HDF5 file or by running retrieval analysis.
    
    Parameters:
    -----------
    path : pathlib.Path
        Base path for the project
    target : str
        Target name (e.g., 'TWA27A', 'TWA28')
    run : str
        Run identifier 
    config_file : str
        Configuration file name
    cache : bool
        Whether to use cached data if available
        
    Returns:
    --------
    tuple
        (pressure, temperature_envelopes, integrated_contribution, logg_median, optical_depth)
    """
    envelopes_dir = path / target / f'retrieval_outputs/{run}/test_data'/ 'envelopes'
    envelopes_dir.mkdir(parents=True, exist_ok=True)

    # Use HDF5 file instead of separate npy files
    pt_data_file = envelopes_dir / 'pt_profile_data.h5'

    if cache and pt_data_file.exists():
        print(f' --> Found cached HDF5 file: {pt_data_file}')
        try:
            pressure, temperature, icf, logg, optical_depth = load_pt_data_h5(pt_data_file)
            return pressure, temperature, icf, logg, optical_depth
        except Exception as e:
            print(f' --> Error loading HDF5 file: {e}')
            print(' --> Will recalculate...')
    
    print(f' --> Calculating PT envelopes for {target}/{run}')
    check_dir(target)
    conf = Config(path=path, target=target, run=run)(config_file)        
        
    ret = Retrieval(
        conf=conf, 
        evaluation=False
        )

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
    
    # Get integrated contribution function
    ret.copy_integrated_contribution_emission()
    
    # Handle optical depth with error checking
    optical_depth = None
    try:
        if hasattr(ret.pRT_atm['NIRSpec'].atm[0], 'total_tau'):
            optical_depth = np.mean(ret.pRT_atm['NIRSpec'].atm[0].total_tau[0], axis=0)
            print(f' --> optical_depth.shape = {optical_depth.shape}')
        else:
            print(' --> Warning: total_tau not available, optical_depth will be None')
    except (AttributeError, KeyError, IndexError) as e:
        print(f' --> Warning: Could not calculate optical depth: {e}')
        optical_depth = None
    
    # Save all data to HDF5 file
    save_pt_data_h5(
        filepath=pt_data_file,
        pressure=ret.PT.pressure,
        temperature_envelopes=ret.PT.temperature_envelopes,
        integrated_contribution=ret.PT.int_contr_em['NIRSpec'],
        logg_posterior=logg_posterior,
        optical_depth=optical_depth,
        target=target,
        run=run
    )
    
    # Also save VMR posteriors
    ret.Chem.get_VMRs_posterior(save_to=envelopes_dir)
    
    logg_median = np.median(logg_posterior)
    return ret.PT.pressure, ret.PT.temperature_envelopes, ret.PT.int_contr_em['NIRSpec'], logg_median, optical_depth


def plot_envelopes(p, t_env, ax=None, cf=None, use_altitude=False, logg=None, **kwargs):
    
    ax = ax or plt.gca()
    assert len(t_env.shape) > 1, f'Expected 2D array, got {t_env.shape}'
    assert t_env.shape[0] == 7, f'Expected 7 envelopes, got {t_env.shape[0]}'
    
    color = kwargs.pop('color', 'brown')
    alpha = kwargs.pop('alpha', 0.2)
    label = kwargs.pop('label', '')
    
    if use_altitude and logg is not None:
        y = pressure_to_altitude(p, t_env[3,:], logg)
    else:
        y = p
    
    for i in range(3):
        ax.fill_betweenx(y, 
                            t_env[i,:],
                            t_env[-(i+1),:], color=color, alpha=alpha, lw=0, 
                            label=label if i ==0 else '',
                            # ls='--',
                            )
    ax.plot(t_env[3,:], y, color=color, lw=1.2, ls=kwargs.pop('ls', '-'), alpha=0.75)

    if cf is not None:
        fill_cf = kwargs.pop('fill_cf', False)
        ax_cf = ax.twiny()
        ls = kwargs.pop('ls_cf', ':')
        lw = kwargs.pop('lw_cf', 2.5)
        ax_cf.plot(cf, y, color=color, lw=lw, ls=ls, alpha=0.75)
        ax_cf.set_xticks([])
        ax_cf.set_yticks([])
        # ax_cf.set_yticks([], minor=True)
        ax_cf.set_xlim(0, np.max(cf)*4.5)
        if fill_cf:
            ax_cf.fill_between(cf, y, color=color, alpha=0.05)
        
    return ax

def plot_crires(ax, use_altitude=False):
    """Plot CRIRES data for TWA28"""
    run_full = 'final_full'
    p, t, cf, log_g, _ = get_PT(path, 'TWA28', run=run_full, config_file='config_freechem.txt')
    # print(f' temperature = {t}')
    ax = plot_envelopes(p, t, ax=ax, cf=cf, color=colors['TWA28']['crires'] , alpha=0.2, label='TWA 28\n' + r'(CRIRES$^{+}$)', fill_cf=True,
                        ls='--', ls_cf='--', lw_cf=1.0, use_altitude=use_altitude, logg=log_g)
    return ax


def main():
    """Main plotting function"""
    fig, ax = plt.subplots(1,1,figsize=(4,4), tight_layout=True)
    use_altitude = False
    altitude_label = '_altitude' if use_altitude else ''

    targets_params = dict(
        TWA27A={'teff': (2430, 20)},
        TWA28={'teff': (2382, 42)},
                )

    for t, target in enumerate(runs.keys()):
        Teff = targets_params[target]['teff']
        # label_teff = r'T$_{\rm eff}$' + f' = {Teff[0]} K'
        label_teff = f'{Teff[0]:.0f} K'
        # ax.axvspan(Teff[0]-Teff[1], Teff[0]+Teff[1], color=colors[target]['model'], alpha=0.3, label=label_teff, lw=0, zorder=-1)
        ax.axvline(Teff[0], color=colors[target]['model'], ls=':', lw=2, zorder=-10, alpha=0.8, label=label_teff)
        
        if target == 'TWA28':
            try:
                plot_crires(ax, use_altitude=use_altitude)
            except Exception as e:
                print(f' --> Error plotting CRIRES: {e}')
                continue
            
        # for r, run in enumerate(runs[target]):
        p, t, cf, logg, optical_depth = get_PT(path, target, runs[target], cache=True)
        # print(t)
        ax = plot_envelopes(p, t, ax=ax, cf=cf, color=colors[target]['model'], alpha=0.4, fill_cf=True,
                            label='TWA ' + target.replace('TWA', ''),
                            ls_cf='-', lw_cf=1.0, use_altitude=use_altitude, logg=logg)

        

    z = pressure_to_altitude(p, t[3,:], logg) if use_altitude else p
    # ylim = (np.min(z), np.max(z)) if use_altitude else (np.max(p), np.min(p))
    ylim = (np.max(z), np.min(z))
    yscale = 'log'
    ylabel = r'Scaled pressure ' + r'(P$\cdot$ g)' if use_altitude else 'Pressure (bar)'
    ax.set(yscale=yscale, ylim=ylim, ylabel=ylabel, xlabel='Temperature (K)')
    ax.set_xlim(None, 5000)
    # make legend labels bold
    # ax.legend(fontsize=8, 
    ax.legend(prop={'size': 14, 'weight': 'bold'}, loc='upper right')
    handles, labels = ax.get_legend_handles_labels()
    # Filter out labels that don't exist (e.g., CRIRES if commented out)
    labels_sort = ['2430 K', 'TWA 27A', '2382 K', 'TWA 28', 'TWA 28\n(CRIRES$^{+}$)']
    legend_dict = dict(zip(labels, handles))
    handles_sort = [legend_dict[label] for label in labels_sort if label in legend_dict]
    labels_sort = [label for label in labels_sort if label in legend_dict]
    leg = ax.legend(handles_sort, labels_sort, 
                    prop={'size': 10, 'weight': 'bold'}, 
                    loc=(0.54, 0.6),
                    frameon=False, ncol=1)

    # for lh in leg.get_lines():
    #     lh.set_alpha(1.0)
        
    for p, patch in enumerate(leg.get_patches()):
        # print(patch)
        patch.set_alpha(0.70)
        # add edge to patch
        # if p == 2:
        #     patch.set_edgecolor('orange')
        #     patch.set_linewidth(0.95)
            # patch.set_linestyle('dashed')

    # plt.show()
    fig_name = path_figures / f'fig_PTs{altitude_label}.pdf'
    fig.savefig(fig_name, bbox_inches='tight')
    # also save as transparent png
    # fig.savefig(fig_name.with_suffix('.png'), dpi=300, transparent=True)
    print(f' --> Saved {fig_name}')
    plt.close(fig)


if __name__ == "__main__":
    main()