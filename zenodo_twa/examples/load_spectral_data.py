#!/usr/bin/env python3
"""
Example script to load and plot spectral data from Zenodo dataset

This script generates one figure per grating showing the complete spectrum
with best-fit model and residuals. The order structure is maintained for
data management but plots show the full grating coverage.

Usage:
    python load_spectral_data.py
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import json

def load_spectral_data(filename):
    """Load spectral data from HDF5 file"""
    with h5py.File(filename, 'r') as f:
        # Load metadata
        metadata = json.loads(f.attrs['metadata'])
        print(f"Target: {metadata['data_info']['target']}")
        print(f"Run: {metadata['data_info']['run']}")
        
        # Load data by grating
        data = {'metadata': metadata}
        
        # Get grating information
        grating_info = json.loads(f['info'].attrs['grating_info'])
        data['grating_info'] = grating_info
        
        for grating in ['g140h', 'g235h', 'g395h']:
            if grating in f:
                grating_data = {'orders': []}
                n_orders = f[grating].attrs['n_orders']
                
                for order_idx in range(n_orders):
                    order_key = f'order_{order_idx}'
                    if order_key in f[grating]:
                        order_data = {}
                        
                        # Load observational data
                        obs_grp = f[grating][order_key]['observational_data']
                        order_data['obs_wave'] = obs_grp['wavelength'][:]
                        order_data['obs_flux'] = obs_grp['flux'][:]
                        order_data['obs_err'] = obs_grp['flux_error'][:]
                        order_data['obs_mask'] = obs_grp['mask_isfinite'][:]
                        
                        # Load model data
                        model_grp = f[grating][order_key]['model_data']
                        order_data['model_wave'] = model_grp['wavelength'][:]
                        order_data['model_flux'] = model_grp['flux_total'][:]
                        order_data['model_bb'] = model_grp['flux_blackbody'][:]
                        
                        # Add order metadata
                        order_data['global_order_index'] = f[grating][order_key].attrs['global_order_index']
                        order_data['order_in_grating'] = f[grating][order_key].attrs['order_in_grating']
                        order_data['n_points'] = f[grating][order_key].attrs['n_points']
                        order_data['wavelength_range_nm'] = f[grating][order_key].attrs['wavelength_range_nm']
                        
                        grating_data['orders'].append(order_data)
                
                data[grating] = grating_data
        
        return data

def plot_grating_spectrum(data, grating='g235h', order_in_grating=0):
    """Plot observed and model spectra for a specific grating and order"""
    if grating not in data:
        print(f"Grating {grating} not found in data")
        return
    
    if order_in_grating >= len(data[grating]['orders']):
        print(f"Order {order_in_grating} not found in grating {grating}")
        return
    
    order_data = data[grating]['orders'][order_in_grating]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot spectra
    ax1.plot(order_data['obs_wave'], order_data['obs_flux'], 'k-', alpha=0.7, label='Observed')
    ax1.plot(order_data['model_wave'], order_data['model_flux'], 'r-', alpha=0.8, label='Model')
    ax1.plot(order_data['model_wave'], order_data['model_bb'], 'b--', alpha=0.6, label='Blackbody disk')
    
    ax1.set_ylabel('Flux (erg s⁻¹ cm⁻² nm⁻¹)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'{grating.upper()} - Order {order_in_grating} (Global Order {order_data["global_order_index"]})')
    
    # Plot residuals
    residuals = (order_data['obs_flux'] - order_data['model_flux']) / order_data['obs_flux']
    ax2.plot(order_data['obs_wave'], residuals, 'g-', alpha=0.7)
    ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('Relative residuals')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_full_grating_spectrum(data, grating='g235h', save_fig=False, figsize=(14, 10)):
    """Plot the complete grating spectrum by combining all orders"""
    if grating not in data:
        print(f"Grating {grating} not found in data")
        return
    
    # Combine all orders for this grating
    all_obs_wave = []
    all_obs_flux = []
    all_obs_err = []
    all_model_wave = []
    all_model_flux = []
    all_model_bb = []
    
    for order_data in data[grating]['orders']:
        # Only include finite data points
        mask = order_data['obs_mask']
        
        all_obs_wave.extend(order_data['obs_wave'][mask])
        all_obs_flux.extend(order_data['obs_flux'][mask])
        all_obs_err.extend(order_data['obs_err'][mask])
        all_model_wave.extend(order_data['model_wave'][mask])
        all_model_flux.extend(order_data['model_flux'][mask])
        all_model_bb.extend(order_data['model_bb'][mask])
    
    # Convert to numpy arrays and sort by wavelength
    all_obs_wave = np.array(all_obs_wave)
    all_obs_flux = np.array(all_obs_flux)
    all_obs_err = np.array(all_obs_err)
    all_model_wave = np.array(all_model_wave)
    all_model_flux = np.array(all_model_flux)
    all_model_bb = np.array(all_model_bb)
    
    # Sort by wavelength
    sort_idx = np.argsort(all_obs_wave)
    all_obs_wave = all_obs_wave[sort_idx]
    all_obs_flux = all_obs_flux[sort_idx]
    all_obs_err = all_obs_err[sort_idx]
    
    sort_idx_model = np.argsort(all_model_wave)
    all_model_wave = all_model_wave[sort_idx_model]
    all_model_flux = all_model_flux[sort_idx_model]
    all_model_bb = all_model_bb[sort_idx_model]
    
    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Top panel: Spectra
    ax1.plot(all_obs_wave, all_obs_flux, 'k-', alpha=0.7, linewidth=0.8, label='Observed')
    ax1.plot(all_model_wave, all_model_flux, 'r-', alpha=0.8, linewidth=1.0, label='Best-fit model')
    ax1.plot(all_model_wave, all_model_bb, 'b--', alpha=0.6, linewidth=1.0, label='Blackbody disk')
    
    # Add error envelope (optional, can be commented out if too cluttered)
    # ax1.fill_between(all_obs_wave, all_obs_flux - all_obs_err, all_obs_flux + all_obs_err, 
    #                  alpha=0.2, color='gray', label='1σ uncertainty')
    
    ax1.set_ylabel('Flux (erg s⁻¹ cm⁻² nm⁻¹)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Add grating information to title
    target = data['metadata']['data_info']['target']
    run = data['metadata']['data_info']['run']
    n_orders = len(data[grating]['orders'])
    n_points = sum(order['n_points'] for order in data[grating]['orders'])
    wave_min = np.min(all_obs_wave)
    wave_max = np.max(all_obs_wave)
    
    # Create a more informative title
    title = f'{target} - {grating.upper()} Spectrum\n'
    title += f'{n_orders} orders, {n_points} points, {wave_min:.0f}-{wave_max:.0f} nm'
    ax1.set_title(title, fontsize=12, pad=20)
    
    # Add run information as text
    ax1.text(0.02, 0.95, f'Run: {run}', transform=ax1.transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             verticalalignment='top', fontsize=10)
    
    # Bottom panel: Residuals
    # Interpolate model to observed wavelengths for residuals
    model_flux_interp = np.interp(all_obs_wave, all_model_wave, all_model_flux)
    residuals = (all_obs_flux - model_flux_interp) / all_obs_flux
    
    ax2.plot(all_obs_wave, residuals, 'g-', alpha=0.7, linewidth=0.8)
    ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
    
    # Add horizontal lines for ±1σ, ±2σ, ±3σ
    residual_std = np.nanstd(residuals)
    for sigma, alpha, color in [(1, 0.4, 'gray'), (2, 0.3, 'orange'), (3, 0.2, 'red')]:
        ax2.axhline(sigma * residual_std, color=color, linestyle=':', alpha=alpha, linewidth=1)
        ax2.axhline(-sigma * residual_std, color=color, linestyle=':', alpha=alpha, linewidth=1)
    
    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('Relative residuals')
    ax2.grid(True, alpha=0.3)
    
    # Add residual statistics
    residual_rms = np.sqrt(np.nanmean(residuals**2))
    residual_mean = np.nanmean(residuals)
    stats_text = f'RMS = {residual_rms:.4f}\nMean = {residual_mean:.4f}\nStd = {residual_std:.4f}'
    ax2.text(0.02, 0.95, stats_text, transform=ax2.transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             verticalalignment='top', fontsize=9)
    
    # Add sigma level legend
    ax2.text(0.98, 0.95, '1σ, 2σ, 3σ levels', transform=ax2.transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             verticalalignment='top', horizontalalignment='right', fontsize=9)
    
    plt.tight_layout()
    
    if save_fig:
        fig_name = f'{target}_{grating}_spectrum.png'
        plt.savefig(fig_name, dpi=300, bbox_inches='tight')
        print(f"  Saved figure: {fig_name}")
    
    plt.show()
    
    return fig

def generate_all_grating_figures(data, save_fig=True, figsize=(14, 10)):
    """Generate figures for all available gratings"""
    target = data['metadata']['data_info']['target']
    print(f"\nGenerating all grating figures for {target}...")
    
    figures = {}
    for grating in ['g140h', 'g235h', 'g395h']:
        if grating in data:
            print(f"  Plotting {grating.upper()} spectrum...")
            fig = plot_full_grating_spectrum(data, grating=grating, save_fig=save_fig, figsize=figsize)
            figures[grating] = fig
    
    print(f"Generated {len(figures)} grating figures")
    return figures

def plot_all_orders(data, grating='g235h'):
    """Plot all orders for a specific grating"""
    if grating not in data:
        print(f"Grating {grating} not found in data")
        return
    
    n_orders = len(data[grating]['orders'])
    fig, axes = plt.subplots(n_orders, 1, figsize=(12, 2*n_orders), sharex=True)
    
    if n_orders == 1:
        axes = [axes]
    
    for i, order_data in enumerate(data[grating]['orders']):
        ax = axes[i]
        
        # Plot spectra
        ax.plot(order_data['obs_wave'], order_data['obs_flux'], 'k-', alpha=0.7, label='Observed')
        ax.plot(order_data['model_wave'], order_data['model_flux'], 'r-', alpha=0.8, label='Model')
        ax.plot(order_data['model_wave'], order_data['model_bb'], 'b--', alpha=0.6, label='Blackbody disk')
        
        ax.set_ylabel('Flux\n(erg s⁻¹ cm⁻² nm⁻¹)')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Order {i} (Global Order {order_data["global_order_index"]})')
        
        if i == 0:
            ax.legend()
        if i == n_orders - 1:
            ax.set_xlabel('Wavelength (nm)')
    
    plt.suptitle(f'{grating.upper()} - All Orders')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    target = "TWA28"  # Change to TWA27A for the other target
    filename = f"{target}_spectral_data.h5"
    
    print(f"Loading spectral data for {target}...")
    data = load_spectral_data(filename)
    
    # Print summary
    print("\nData Summary:")
    for grating in ['g140h', 'g235h', 'g395h']:
        if grating in data:
            n_orders = len(data[grating]['orders'])
            total_points = sum(order['n_points'] for order in data[grating]['orders'])
            wave_ranges = [order['wavelength_range_nm'] for order in data[grating]['orders']]
            min_wave = min(wr[0] for wr in wave_ranges)
            max_wave = max(wr[1] for wr in wave_ranges)
            print(f"  {grating.upper()}: {n_orders} orders, {total_points} total points, {min_wave:.1f}-{max_wave:.1f} nm")
    
    # Generate one figure per grating
    print("\nGenerating grating spectra...")
    
    # Use the convenience function to generate all figures
    figures = generate_all_grating_figures(data, save_fig=True)
    
    print("\nAll grating spectra generated successfully!")
    
    # Optional: Also show examples of individual order plots and all orders
    print("\nOptional: Individual order examples (uncomment to use):")
    print("# plot_grating_spectrum(data, grating='g235h', order_in_grating=0)")
    print("# plot_all_orders(data, grating='g235h')")
    
    # Uncomment these lines to see individual order plots
    # plot_grating_spectrum(data, grating='g235h', order_in_grating=0)
    # plot_all_orders(data, grating='g235h')
