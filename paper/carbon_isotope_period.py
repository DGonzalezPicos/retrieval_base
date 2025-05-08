"""Script for analyzing and plotting carbon and oxygen isotope ratios in M-dwarf atmospheres.

This module generates a figure with four subpanels showing carbon and oxygen isotope ratios
as a function of metallicity and age, including comparison with solar, ISM and model values.
"""

from retrieval_base.retrieval import Retrieval
import retrieval_base.figures as figs
from retrieval_base.config import Config
from retrieval_base.auxiliary_functions import (
    spirou_sample, read_spirou_sample_csv, find_run, 
    load_romano_models, axhspan_gradient
)
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import matplotlib.patheffects as pe
import scienceplots
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any

# Configure plotting style
plt.style.use('default')
plt.style.use(['sans'])
plt.rcParams.update({"font.size": 8})

# Global paths
BASE_PATH = '/home/dario/phd/retrieval_base/'
NAT_PATH = '/home/dario/phd/nat/figures/'

# Constants and configuration
SIGMA_COLORS = {
    '3': 'k',
    '2': '#0C823E',
    '1': '#ff6a90'
}

Y_LABELS = {
    'oxygen': r'$^{16}$O/$^{18}$O', 
    'carbon': r'$^{12}$C/$^{13}$C'
}

def load_stellar_data() -> Tuple[Dict[str, float], Dict[str, str], plt.Normalize]:
    """Load and process stellar data from CSV."""
    df = read_spirou_sample_csv()
    df = df.iloc[::-1]  # Flip rows
    names = df['Star'].to_list()
    
    teff = dict(zip(names, [float(t.split('+-')[0]) for t in df['Teff (K)'].to_list()]))
    spt = dict(zip(names, [t.split('+-')[0] for t in df['SpT'].to_list()]))
    
    norm = plt.Normalize(3000.0, 3900.0)
    return teff, spt, norm

def load_reference_values() -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Tuple[float, float]]]:
    """Load solar and ISM reference values."""
    sun_dict = {
        'oxygen': (529.7, 1.7),  # solar wind McKeegan et al. 2011
        'carbon': (93.5, 3.0),
        'age': (4.56, 0.07),
        'metallicity': (0.0, 0.0)
    }
    
    ism_dict = {
        'oxygen': (557, 30),  # ISM value from Wilson et al. 1999
        'carbon': (68.0, 14.0)
    }
    
    return sun_dict, ism_dict

def load_crossfield_data() -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Load Crossfield+2019 values for Gl 745 AB."""
    return {
        'A': {
            'carbon_isotope': (296, 45),
            'oxygen_isotope': (1220, 260),
            'teff': (3454, 31),
            'metallicity': (-0.43, 0.05),
            'age': (np.nan, np.nan)
        },
        'B': {
            'carbon_isotope': (224, 26),
            'oxygen_isotope': (1550, 360),
            'teff': (3440, 31),
            'metallicity': (-0.39, 0.05),
            'age': (np.nan, np.nan)
        }
    }

def process_isotope_data(
    target: str, 
    isotope: str, 
    x: float, 
    xerr: float, 
    ax: plt.Axes,
    run: Optional[str] = None,
    label: str = '',
    color: str = 'k',
    xytext: Optional[Tuple[float, float]] = None,
    **kwargs
) -> Optional[np.ndarray]:
    """Process and plot isotope data for a given target."""
    if target not in os.getcwd():
        os.chdir(BASE_PATH + target)
        
    outputs = pathlib.Path(BASE_PATH) / target / 'retrieval_outputs'
    dirs = [d for d in outputs.iterdir() if d.is_dir() and 'fc' in d.name and '_' not in d.name]
    runs = [int(d.name.split('fc')[-1]) for d in dirs]
    
    if not runs:
        return None
        
    run = f'fc{run if run else max(runs)}'
    test_output = outputs / run / 'test_output'
    
    if not test_output.exists() or not list(test_output.iterdir()):
        return None
        
    # Load sigma for isotope detection
    sigma = 10.0  # default
    species_sigma = 'C18O' if isotope == 'oxygen' else '13CO'
    sigma_file = test_output / f'lnB_sigma_{species_sigma}.dat'
    
    if sigma_file.exists():
        lnB, sigma = np.loadtxt(sigma_file)
        sigma = 0.0 if np.isnan(sigma) else min(sigma, 100.0)
    
    # Load or calculate isotope ratio
    isotope_posterior_file = f'{BASE_PATH}{target}/retrieval_outputs/{run}/CO_{isotope}_isotope_posterior.npy'
    
    if not os.path.exists(isotope_posterior_file):
        config_file = 'config_freechem.txt'
        conf = Config(path=BASE_PATH, target=target, run=run)(config_file)
        ret = Retrieval(conf=conf, evaluation=False)
        bestfit_params, posterior = ret.PMN_analyze()
        
        param_keys = list(ret.Param.param_keys)
        key = 'log_H2O/H2O_181' if isotope == 'oxygen' else 'log_12CO/13CO'
        log_ratio_id = param_keys.index(key)
        isotope_posterior = 10.0**posterior[:, log_ratio_id]
        np.save(isotope_posterior_file, isotope_posterior)
    else:
        isotope_posterior = np.load(isotope_posterior_file)
    
    # Plot data point with appropriate error styling
    isotope_quantiles = np.quantile(isotope_posterior, [0.16, 0.5, 0.84])
    fmt = 'o'
    
    # Handle x-axis error bars
    if xerr is None:
        xerr = np.array([[0], [0]])
    elif isinstance(xerr, (int, float)):
        xerr = np.array([[xerr], [xerr]])  # Symmetric errors
    elif isinstance(xerr, (list, np.ndarray)):
        if len(np.array(xerr).shape) == 1:
            xerr = np.array([[xerr[0]], [xerr[1]]])
        else:
            xerr = np.array(xerr)
    
    # Handle y-axis error bars
    yerr = [[isotope_quantiles[1]-isotope_quantiles[0]], 
            [isotope_quantiles[2]-isotope_quantiles[1]]]
    
    if sigma > 3.0:
        edge_color = SIGMA_COLORS['3']
    elif sigma > 2.0:
        edge_color = SIGMA_COLORS['2']
    elif sigma > 1.0:
        edge_color = SIGMA_COLORS['1']
    else:
        return isotope_quantiles
        
    ax.errorbar(
        x, isotope_quantiles[1],
        xerr=xerr,
        yerr=yerr,
        fmt=fmt,
        label=label.replace('gl', 'Gl '),
        alpha=0.96,
        markeredgecolor=edge_color,
        markeredgewidth=0.8,
        capsize=2,
        capthick=0.8,
        ecolor='gray',
        elinewidth=0.8,
        color=color
    )
    
    if xytext:
        ax.annotate(
            label.replace('gl', 'Gl '), 
            (x, isotope_quantiles[1]),
            textcoords="offset points",
            xytext=xytext,
            ha='left',
            fontsize=8,
            color=color,
            alpha=0.9
        )
    
    return isotope_quantiles

def setup_panel(
    ax: plt.Axes,
    x_param: str,
    isotope: str,
    sun_dict: Dict[str, Tuple[float, float]],
    ism_dict: Dict[str, Tuple[float, float]],
    x_span: Dict[str, np.ndarray],
    plot_crossfield: bool = True,
    crossfield: Optional[Dict] = None
) -> Tuple[Any, Any]:
    """Setup an individual panel with reference values and styling."""
    ism = ism_dict[isotope]
    sun = sun_dict[isotope]
    
    # Plot solar value
    x_sun = sun_dict[x_param][0]
    ax.plot(x_sun, sun[0], color='gold', marker='*', ms=16, 
            label='Sun', alpha=0.8, markeredgecolor='black', 
            markeredgewidth=0.8, zorder=100)
    
    # Plot ISM band
    rgb_color = np.array([10, 191, 134]) / 255.0 * 0.7
    poly, ism_label = axhspan_gradient(
        ax, x_span[x_param], 
        y_range=(ism[0]-ism[1], ism[0]+ism[1]), 
        rgb_color=rgb_color, gamma=3, n=120,
        label='ISM'
    )
    
    # Plot Crossfield values if requested
    if plot_crossfield and crossfield:
        for AB_i, v in crossfield.items():
            fmt = 's' if x_param == 'metallicity' else 'D'
            ax.errorbar(
                v[x_param][0],
                v[f'{isotope}_isotope'][0],
                xerr=v[x_param][1],
                yerr=v[f'{isotope}_isotope'][1],
                fmt=fmt,
                label=AB_i,
                color=plt.cm.coolwarm_r(plt.Normalize(3000.0, 3900.0)(v['teff'][0])),
                markeredgecolor='black',
                markeredgewidth=0.8
            )
    
    return poly, ism_label

def create_figure(
    isotopes: List[str] = ['carbon', 'oxygen'],
    x_params: List[str] = ['metallicity', 'age']
) -> None:
    """Create the main figure with all panels."""
    teff, spt, norm = load_stellar_data()
    sun_dict, ism_dict = load_reference_values()
    crossfield = load_crossfield_data()
    
    fig, axes_grid = plt.subplots(2, 2, figsize=(9, 9))
    y_keys = ['carbon', 'oxygen']
    x_keys = ['metallicity', 'age']
    axes_keys = [f'{y}_{x}' for y in y_keys for x in x_keys]
    axes = {k: axes_grid.flatten()[i] for i, k in enumerate(axes_keys)}
    

    
    # Load metallicity data
    metallicity_ref = 'C23'
    table_id = 3
    c23 = np.loadtxt(f'{BASE_PATH}paper/data/c23_table{table_id}_mh.txt', dtype=object)
    c23_names = ['Gl '+n[2:] for n in c23[:,0]]
    metallicity = dict(zip(c23_names, c23[:,1].astype(float)))
    metallicity_err = dict(zip(c23_names, c23[:,2].astype(float)))
    
    # Load age data
    ages_file = f"{BASE_PATH}paper/data/rotation_age_estimates.csv"
    df_rotation_ages = pd.read_csv(ages_file)
    age = dict(zip(df_rotation_ages['Name'], df_rotation_ages['Age (Gyr)']))
    age_err = dict(zip(df_rotation_ages['Name'], df_rotation_ages['Age Error (Gyr)']))
    
    # Setup x-axis spans
    x_spans = {
        'metallicity': np.linspace(-0.4, 0.6, 100),
        'age': np.linspace(0.0, 14.0, 100)[::-1]
    }
    
    # Process each panel
    # for i, (isotope, x_param) in enumerate([(iso, x) for iso in isotopes for x in x_params]):
    for y_key in y_keys: # carbon and oxygen
        for x_key in x_keys: # metallicity and age
            ax = axes[f'{y_key}_{x_key}']
            poly, ism_label = setup_panel(ax, x_key, y_key, sun_dict, ism_dict, x_spans, True, crossfield)
            
            # Plot data points
            plot_teff_max = 4400.0
            for name in read_spirou_sample_csv()['Star'].to_list():
                if teff[name] > plot_teff_max:
                    continue
                    
                target = name.replace('Gl ', 'gl')
                color = plt.cm.coolwarm_r(norm(teff[name]))
                
                x = metallicity[name] if x_key == 'metallicity' else age.get(name, np.nan)
                x_err = metallicity_err[name] if x_key == 'metallicity' else age_err.get(name, np.nan)
                
                process_isotope_data(
                    target, y_key, x, x_err,
                    ax=ax, label='', color=color
                    )
    
    # Add Romano models
    mass_ranges = ['1_8', '3_8']
    gce_colors = ['black', 'purple']
    path_effects = [pe.Stroke(linewidth=2.5, foreground='white'), pe.Normal()]
    
    for i, mass_range in enumerate(mass_ranges):
        Z, c12c13, o16o18, time = load_romano_models(
            Z_min=-0.7, mass_range=mass_range, return_time=True
        )
        mass_range_label = mass_range.replace('_', '-') + r' M$_\odot$'
        
        for y_key in y_keys:
            ax_pair = [axes[f'{y_key}_metallicity'], axes[f'{y_key}_age']]
            ax_pair[0].plot(Z, c12c13 if y_key == 'carbon' else o16o18, color=gce_colors[i], lw=1.5,
                          label=mass_range_label, alpha=0.8, path_effects=path_effects)
            ax_pair[1].plot(time[::-1], c12c13 if y_key == 'carbon' else o16o18, color=gce_colors[i],
                          lw=1.5, label=mass_range_label, alpha=0.8, path_effects=path_effects)
    
    # Finalize figure
    axes['carbon_metallicity'].set_yscale('log')
    axes['carbon_metallicity'].set_ylabel(Y_LABELS['carbon'])
    axes['carbon_metallicity'].set_xlabel('metallicity')
    
    axes['carbon_age'].set_yscale('log')
    axes['carbon_age'].set_ylabel(Y_LABELS['carbon'])
    axes['carbon_age'].set_xlabel('age (Gyr)')
    
    axes['oxygen_metallicity'].set_yscale('log')
    axes['oxygen_metallicity'].set_ylabel(Y_LABELS['oxygen'])
    axes['oxygen_metallicity'].set_xlabel('metallicity')
    
    axes['oxygen_age'].set_yscale('log')
    # axes['oxygen_age'].set_xscale('log')
    axes['oxygen_age'].set_xlabel('age (Gyr)')
    
    y_ticks = {'oxygen': [200, 500, 1000, 2000, 4000], 'carbon': [40, 60, 100, 200, 300, 400]}
    y_lims = {'oxygen' :(np.min(y_ticks['oxygen']), np.max(y_ticks['oxygen'])),
              'carbon' :(np.min(y_ticks['carbon']), np.max(y_ticks['carbon']))}
    
    axes['carbon_metallicity'].set_yticks(y_ticks['carbon'])
    axes['carbon_metallicity'].set_yticklabels([str(t) for t in y_ticks['carbon']])
    axes['carbon_age'].set_yticks(y_ticks['carbon'])
    axes['carbon_age'].set_yticklabels([str(t) for t in y_ticks['carbon']])
    axes['oxygen_metallicity'].set_yticks(y_ticks['oxygen'])
    axes['oxygen_metallicity'].set_yticklabels([str(t) for t in y_ticks['oxygen']])
    axes['oxygen_age'].set_yticks(y_ticks['oxygen'])
    axes['oxygen_age'].set_yticklabels([str(t) for t in y_ticks['oxygen']])
    
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm_r, norm=norm)
    cbar_ax = fig.add_axes([0.92, 0.58, 0.027, 0.30])
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation='vertical', aspect=1)
    cbar.set_label(r'T$_{\mathrm{eff}}$ (K)')
    
    # Add legends
    axes['oxygen_age'].legend(ncol=1, frameon=False, fontsize=12, loc=(1.01, 0.5))
    # axes['carbon_age'].legend(ncol=1, frameon=False, fontsize=8, loc=(2.42, 0.5))
    # axes['oxygen_metallicity'].legend(ncol=1, frameon=False, fontsize=8, loc=(2.42, 0.5))
    # axes['oxygen_age'].legend(ncol=1, frameon=False, fontsize=8, loc=(2.42, 0.5))
    
    
    xlims = {'metallicity': (-0.6, 0.6), 'age': (0.0, 14.0)}
    axes['carbon_metallicity'].set_xlim(xlims['metallicity'])
    axes['carbon_age'].set_xlim(xlims['age'])
    axes['oxygen_metallicity'].set_xlim(xlims['metallicity'])
    axes['oxygen_age'].set_xlim(xlims['age'])
    
    # flip x-axis for age panel
    axes['carbon_age'].invert_xaxis()
    axes['oxygen_age'].invert_xaxis()
    
    
    # Add sigma legend
    sigma_handles = [plt.Line2D([0], [0], marker='o', color='w',
                               markeredgecolor=SIGMA_COLORS[sigma],
                               markersize=6, markeredgewidth=0.9)
                    for sigma in ['3', '2', '1']]
    sigma_labels = [f'≥{int(sigma)}σ' if sigma == '3' 
                   else f'{int(sigma)}σ - {int(sigma)+1}σ'
                   for sigma in ['3', '2', '1']]
    
    legend = axes['carbon_age'].legend(sigma_handles, 
                                       sigma_labels,
                                        framealpha=0.4, 
                                        fontsize=9)
    legend.get_frame().set_linewidth(0.5)
    
    # Save figure
    fig.savefig(f'{NAT_PATH}CO_isotopes_metallicity_age_4panel.pdf',
                bbox_inches='tight')
    print(f"Figure saved to: {NAT_PATH}CO_isotopes_metallicity_age_4panel.pdf")
    plt.close(fig)
    # saveas png
    fig.savefig(f'{NAT_PATH}/png/CO_isotopes_metallicity_age_4panel.png',
                bbox_inches='tight', dpi=300)
    print(f"Figure saved to: {NAT_PATH}/png/CO_isotopes_metallicity_age_4panel.png")

if __name__ == '__main__':
    create_figure()