#!/usr/bin/env python3
"""
Test script to demonstrate reading isotope data from HDF5 file.

This script shows how to:
- Load isotope ratio data with uncertainties
- Access stellar parameters and detection significance
- Plot isotope ratios vs metallicity
- Overplot GCE models and reference values

Author: Dario Gonzalez Picos  
Date: December 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib.patheffects as pe

# Set plotting style
plt.style.use('default')
plt.rcParams.update({"font.size": 10})

def load_isotope_data(filename: str = 'isotope_data.h5') -> dict:
    """
    Load isotope data from HDF5 file.
    
    Parameters
    ----------
    filename : str
        Path to HDF5 file
        
    Returns
    -------
    dict
        Complete dataset
    """
    
    data = {}
    
    with h5py.File(filename, 'r') as f:
        
        # Load target data
        targets_grp = f['targets']
        
        # Decode string arrays
        target_names = [name.decode('utf-8') for name in targets_grp['target_names'][:]]
        target_ids = [tid.decode('utf-8') for tid in targets_grp['target_ids'][:]]
        spectral_types = [spt.decode('utf-8') for spt in targets_grp['spectral_types'][:]]
        
        data['targets'] = {
            'names': target_names,
            'ids': target_ids,
            'teff': targets_grp['teff'][:],
            'teff_err': targets_grp['teff_err'][:],
            'spectral_types': spectral_types,
            'metallicity': targets_grp['metallicity'][:],
            'metallicity_err': targets_grp['metallicity_err'][:]
        }
        
        # Load carbon isotope data
        carbon_grp = targets_grp['carbon']
        carbon_runs = [run.decode('utf-8') if run else '' for run in carbon_grp['runs'][:]]
        
        data['carbon'] = {
            'ratio_median': carbon_grp['ratio_median'][:],
            'ratio_lower': carbon_grp['ratio_lower'][:],
            'ratio_upper': carbon_grp['ratio_upper'][:],
            'sigma': carbon_grp['sigma'][:],
            'runs': carbon_runs
        }
        
        # Load carbon posteriors if available
        if 'posteriors' in carbon_grp:
            data['carbon']['posteriors'] = {}
            for target_id in carbon_grp['posteriors'].keys():
                data['carbon']['posteriors'][target_id] = carbon_grp['posteriors'][target_id][:]
        
        # Load oxygen isotope data
        oxygen_grp = targets_grp['oxygen']
        oxygen_runs = [run.decode('utf-8') if run else '' for run in oxygen_grp['runs'][:]]
        
        data['oxygen'] = {
            'ratio_median': oxygen_grp['ratio_median'][:],
            'ratio_lower': oxygen_grp['ratio_lower'][:],
            'ratio_upper': oxygen_grp['ratio_upper'][:],
            'sigma': oxygen_grp['sigma'][:],
            'runs': oxygen_runs
        }
        
        # Load oxygen posteriors if available
        if 'posteriors' in oxygen_grp:
            data['oxygen']['posteriors'] = {}
            for target_id in oxygen_grp['posteriors'].keys():
                data['oxygen']['posteriors'][target_id] = oxygen_grp['posteriors'][target_id][:]
        
        # Load GCE models
        models_grp = f['gce_models']
        data['gce_models'] = {}
        
        for mass_range in models_grp.keys():
            model_grp = models_grp[mass_range]
            data['gce_models'][mass_range] = {
                'metallicity': model_grp['metallicity'][:],
                'carbon_ratio': model_grp['carbon_ratio'][:],
                'oxygen_ratio': model_grp['oxygen_ratio'][:],
                'description': model_grp.attrs['description']
            }
        
        # Load reference values
        refs_grp = f['reference_values']
        data['references'] = {}
        
        # Load Sun and ISM references
        for ref_name in ['sun', 'ism']:
            if ref_name in refs_grp:
                ref_grp = refs_grp[ref_name]
                data['references'][ref_name] = {}
                for isotope in ['carbon', 'oxygen']:
                    if isotope in ref_grp:
                        iso_grp = ref_grp[isotope]
                        data['references'][ref_name][isotope] = {
                            'ratio': iso_grp['ratio'][()],
                            'error': iso_grp['error'][()],
                            'reference': iso_grp.attrs['reference']
                        }
        
        # Load Crossfield+2019 references
        if 'crossfield_2019' in refs_grp:
            crossfield_grp = refs_grp['crossfield_2019']
            data['references']['crossfield_2019'] = {}
            
            for target in crossfield_grp.keys():
                target_grp = crossfield_grp[target]
                data['references']['crossfield_2019'][target] = {}
                
                for isotope in target_grp.keys():
                    iso_grp = target_grp[isotope]
                    data['references']['crossfield_2019'][target][isotope] = {
                        key: iso_grp[key][()] for key in iso_grp.keys()
                    }
        
        # Load metadata
        data['metadata'] = {}
        for key in f.attrs.keys():
            data['metadata'][key] = f.attrs[key]
    
    return data


def plot_isotope_metallicity(data: dict, save_path: str = None) -> None:
    """
    Create isotope ratio vs metallicity plot.
    
    Parameters
    ----------
    data : dict
        Loaded isotope data
    save_path : str, optional
        Path to save figure
    """
    
    fig, (ax_carbon, ax_oxygen) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Color coding by temperature
    teff = data['targets']['teff']
    norm = plt.Normalize(3000.0, 3900.0)
    cmap = plt.cm.coolwarm_r
    
    # Sigma detection colors
    sigma_colors = {'3': 'k', '2': '#0C823E', '1': '#ff6a90'}
    
    metallicity = data['targets']['metallicity']
    metallicity_err = data['targets']['metallicity_err']
    
    # Plot carbon isotope ratios
    carbon_ratios = data['carbon']['ratio_median']
    carbon_lower = data['carbon']['ratio_lower']
    carbon_upper = data['carbon']['ratio_upper']
    carbon_sigma = data['carbon']['sigma']
    
    for i, (name, target_id) in enumerate(zip(data['targets']['names'], data['targets']['ids'])):
        if np.isnan(carbon_ratios[i]) or np.isnan(metallicity[i]):
            continue
            
        color = cmap(norm(teff[i]))
        sigma = carbon_sigma[i]
        
        # Determine marker edge color based on sigma
        if sigma >= 3.0:
            edge_color = sigma_colors['3']
        elif sigma >= 2.0:
            edge_color = sigma_colors['2']
        elif sigma >= 1.0:
            edge_color = sigma_colors['1']
        else:
            continue  # Skip non-detections
        
        # Plot with error bars
        ax_carbon.errorbar(
            metallicity[i], carbon_ratios[i],
            xerr=metallicity_err[i],
            yerr=[[carbon_ratios[i] - carbon_lower[i]], [carbon_upper[i] - carbon_ratios[i]]],
            fmt='o', color=color, markeredgecolor=edge_color, markeredgewidth=0.8,
            capsize=2, capthick=0.8, ecolor='gray', elinewidth=0.8, alpha=0.96
        )
    
    # Plot oxygen isotope ratios
    oxygen_ratios = data['oxygen']['ratio_median']
    oxygen_lower = data['oxygen']['ratio_lower']
    oxygen_upper = data['oxygen']['ratio_upper']
    oxygen_sigma = data['oxygen']['sigma']
    
    for i, (name, target_id) in enumerate(zip(data['targets']['names'], data['targets']['ids'])):
        if np.isnan(oxygen_ratios[i]) or np.isnan(metallicity[i]):
            continue
            
        color = cmap(norm(teff[i]))
        sigma = oxygen_sigma[i]
        
        # Determine marker edge color based on sigma
        if sigma >= 3.0:
            edge_color = sigma_colors['3']
        elif sigma >= 2.0:
            edge_color = sigma_colors['2']
        elif sigma >= 1.0:
            edge_color = sigma_colors['1']
        else:
            continue  # Skip non-detections
        
        # Plot with error bars
        ax_oxygen.errorbar(
            metallicity[i], oxygen_ratios[i],
            xerr=metallicity_err[i],
            yerr=[[oxygen_ratios[i] - oxygen_lower[i]], [oxygen_upper[i] - oxygen_ratios[i]]],
            fmt='o', color=color, markeredgecolor=edge_color, markeredgewidth=0.8,
            capsize=2, capthick=0.8, ecolor='gray', elinewidth=0.8, alpha=0.96
        )
    
    # Plot GCE models
    gce_colors = ['black', 'purple']
    path_effects = [pe.Stroke(linewidth=2.5, foreground='white'), pe.Normal()]
    
    for i, (mass_range, model_data) in enumerate(data['gce_models'].items()):
        label = mass_range.replace('_', '-') + r' M$_\odot$'
        
        ax_carbon.plot(
            model_data['metallicity'], model_data['carbon_ratio'],
            color=gce_colors[i], lw=1.5, label=label, alpha=0.8, path_effects=path_effects
        )
        
        ax_oxygen.plot(
            model_data['metallicity'], model_data['oxygen_ratio'],
            color=gce_colors[i], lw=1.5, label=label, alpha=0.8, path_effects=path_effects
        )
    
    # Plot reference values
    refs = data['references']
    
    # Solar values
    sun_carbon = refs['sun']['carbon']
    sun_oxygen = refs['sun']['oxygen']
    
    ax_carbon.plot(0.0, sun_carbon['ratio'], color='gold', marker='*', ms=16, 
                   label='Sun', alpha=0.8, markeredgecolor='black', markeredgewidth=0.8, zorder=100)
    ax_oxygen.plot(0.0, sun_oxygen['ratio'], color='gold', marker='*', ms=16, 
                   label='Sun', alpha=0.8, markeredgecolor='black', markeredgewidth=0.8, zorder=100)
    
    # ISM values (as gradient bands)
    ism_carbon = refs['ism']['carbon']
    ism_oxygen = refs['ism']['oxygen']
    
    x_span = np.linspace(-0.6, 0.6, 100)
    rgb_color = np.array([10, 191, 134]) / 255.0 * 0.7  # light green
    
    ax_carbon.fill_between(x_span, ism_carbon['ratio'] - ism_carbon['error'],
                          ism_carbon['ratio'] + ism_carbon['error'],
                          color=rgb_color, alpha=0.3, label='ISM', zorder=-1)
    
    ax_oxygen.fill_between(x_span, ism_oxygen['ratio'] - ism_oxygen['error'],
                          ism_oxygen['ratio'] + ism_oxygen['error'],
                          color=rgb_color, alpha=0.3, label='ISM', zorder=-1)
    
    # Crossfield+2019 values
    if 'crossfield_2019' in refs:
        crossfield = refs['crossfield_2019']
        
        for j, (target, target_data) in enumerate(crossfield.items()):
            if 'carbon' in target_data and 'oxygen' in target_data:
                fmt = 's' if j == 0 else 'D'
                label = target.replace('_', ' ').replace('Gl', 'GJ')
                
                # Carbon
                c_data = target_data['carbon']
                ax_carbon.errorbar(c_data['metallicity'], c_data['ratio'],
                                 xerr=c_data['metallicity_err'], yerr=c_data['error'],
                                 fmt=fmt, label=label, color='red', 
                                 markeredgecolor='black', markeredgewidth=0.8)
                
                # Oxygen
                o_data = target_data['oxygen']
                ax_oxygen.errorbar(o_data['metallicity'], o_data['ratio'],
                                 xerr=o_data['metallicity_err'], yerr=o_data['error'],
                                 fmt=fmt, label=label, color='red',
                                 markeredgecolor='black', markeredgewidth=0.8)
    
    # Formatting
    axes = [ax_carbon, ax_oxygen]
    y_labels = [r'$^{12}$C/$^{13}$C', r'$^{16}$O/$^{18}$O']
    y_lims = [(40, 400), (200, 4000)]
    
    for ax, ylabel, ylim in zip(axes, y_labels, y_lims):
        ax.set_xlabel('[M/H]')
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(*ylim)
        ax.set_yscale('log')
        
        # Set y-ticks
        if ylabel.startswith(r'$^{12}$C'):
            ax.set_yticks([40, 60, 100, 200, 300, 400])
            ax.set_yticklabels(['40', '60', '100', '200', '300', '400'])
        else:
            ax.set_yticks([200, 500, 1000, 2000, 4000])
            ax.set_yticklabels(['200', '500', '1000', '2000', '4000'])
    
    # Add colorbar for temperature
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes, aspect=30, pad=0.02)
    cbar.set_label(r'T$_{\mathrm{eff}}$ (K)')
    
    # Add legends
    ax_carbon.legend(loc='upper left', fontsize=8)
    
    # Create sigma legend
    from matplotlib.lines import Line2D
    sigma_handles = []
    sigma_labels = []
    
    for sigma in ['3', '2', '1']:
        sigma_handles.append(Line2D([0], [0], marker='o', color='w', 
                                   markeredgecolor=sigma_colors[sigma], 
                                   markersize=6, markeredgewidth=0.9))
        if sigma == '3':
            sigma_labels.append(f'≥{int(sigma)}σ')
        else:
            sigma_labels.append(f'{int(sigma)}σ - {int(sigma)+1}σ')
    
    ax_oxygen.legend(sigma_handles, sigma_labels, loc='upper right', fontsize=8,
                    title='Detection significance')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def print_data_summary(data: dict) -> None:
    """
    Print summary of loaded data.
    
    Parameters
    ----------
    data : dict
        Loaded isotope data
    """
    
    print("=" * 60)
    print("ISOTOPE DATA SUMMARY")
    print("=" * 60)
    
    # Basic statistics
    n_targets = len(data['targets']['names'])
    n_carbon = np.sum(~np.isnan(data['carbon']['ratio_median']))
    n_oxygen = np.sum(~np.isnan(data['oxygen']['ratio_median']))
    
    print(f"Total targets: {n_targets}")
    print(f"Carbon detections: {n_carbon}")
    print(f"Oxygen detections: {n_oxygen}")
    
    # Detection significance breakdown
    carbon_sigma = data['carbon']['sigma'][~np.isnan(data['carbon']['ratio_median'])]
    oxygen_sigma = data['oxygen']['sigma'][~np.isnan(data['oxygen']['ratio_median'])]
    
    print(f"\nCarbon detection significance:")
    print(f"  ≥3σ: {np.sum(carbon_sigma >= 3)}")
    print(f"  2-3σ: {np.sum((carbon_sigma >= 2) & (carbon_sigma < 3))}")
    print(f"  1-2σ: {np.sum((carbon_sigma >= 1) & (carbon_sigma < 2))}")
    
    print(f"\nOxygen detection significance:")
    print(f"  ≥3σ: {np.sum(oxygen_sigma >= 3)}")
    print(f"  2-3σ: {np.sum((oxygen_sigma >= 2) & (oxygen_sigma < 3))}")
    print(f"  1-2σ: {np.sum((oxygen_sigma >= 1) & (oxygen_sigma < 2))}")
    
    # Reference values
    print(f"\nReference values:")
    refs = data['references']
    
    print(f"  Sun (carbon): {refs['sun']['carbon']['ratio']:.1f} ± {refs['sun']['carbon']['error']:.1f}")
    print(f"  Sun (oxygen): {refs['sun']['oxygen']['ratio']:.1f} ± {refs['sun']['oxygen']['error']:.1f}")
    print(f"  ISM (carbon): {refs['ism']['carbon']['ratio']:.1f} ± {refs['ism']['carbon']['error']:.1f}")
    print(f"  ISM (oxygen): {refs['ism']['oxygen']['ratio']:.1f} ± {refs['ism']['oxygen']['error']:.1f}")
    
    # GCE models
    print(f"\nGCE models available:")
    for mass_range, model_data in data['gce_models'].items():
        print(f"  {mass_range}: {model_data['description']}")
        print(f"    Metallicity range: {model_data['metallicity'].min():.2f} to {model_data['metallicity'].max():.2f}")
    
    print("=" * 60)


def main():
    """Main execution function."""
    
    # Load data
    filename = '/home/dario/phd/retrieval_base/paper/isotope_data.h5'
    
    try:
        data = load_isotope_data(filename)
        print(f"Successfully loaded data from {filename}")
    except FileNotFoundError:
        print(f"Data file {filename} not found.")
        print("Please run 'python collect_isotope_data.py' first to generate the data file.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Print summary
    print_data_summary(data)
    
    # Create plot
    save_path = '/home/dario/phd/retrieval_base/paper/test_isotope_plot.pdf'
    plot_isotope_metallicity(data, save_path=save_path)
    
    # Example: Access specific target data
    print(f"\nExample: Accessing data for specific targets")
    print("-" * 40)
    
    for i, (name, target_id) in enumerate(zip(data['targets']['names'][:3], data['targets']['ids'][:3])):
        print(f"\n{name} ({target_id}):")
        print(f"  Teff: {data['targets']['teff'][i]:.0f} ± {data['targets']['teff_err'][i]:.0f} K")
        print(f"  [M/H]: {data['targets']['metallicity'][i]:.2f} ± {data['targets']['metallicity_err'][i]:.2f}")
        
        if not np.isnan(data['carbon']['ratio_median'][i]):
            print(f"  ¹²C/¹³C: {data['carbon']['ratio_median'][i]:.1f} "
                  f"+{data['carbon']['ratio_upper'][i] - data['carbon']['ratio_median'][i]:.1f}"
                  f"-{data['carbon']['ratio_median'][i] - data['carbon']['ratio_lower'][i]:.1f} "
                  f"({data['carbon']['sigma'][i]:.1f}σ)")
        
        if not np.isnan(data['oxygen']['ratio_median'][i]):
            print(f"  ¹⁶O/¹⁸O: {data['oxygen']['ratio_median'][i]:.1f} "
                  f"+{data['oxygen']['ratio_upper'][i] - data['oxygen']['ratio_median'][i]:.1f}"
                  f"-{data['oxygen']['ratio_median'][i] - data['oxygen']['ratio_lower'][i]:.1f} "
                  f"({data['oxygen']['sigma'][i]:.1f}σ)")


if __name__ == '__main__':
    main() 