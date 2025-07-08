#!/usr/bin/env python3
"""
Example script to load and plot spectral data from Zenodo dataset organized by grating

Usage:
    python load_spectral_data.py
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import json
import os

def load_spectral_data(filename):
    """Load spectral data from HDF5 file organized by grating"""
    data = {}
    
    with h5py.File(filename, 'r') as f:
        # Load metadata
        metadata = json.loads(f.attrs['metadata'])
        data['metadata'] = metadata
        print(f"Target: {metadata['data_info']['target']}")
        print(f"Run: {metadata['data_info']['run']}")
        
        # Load data for each grating
        gratings = ['g140h', 'g235h', 'g395h']
        data['gratings'] = {}
        
        for grating in gratings:
            if grating in f:
                grating_data = {}
                
                # Load observational data
                obs_grp = f[grating]['observational_data']
                grating_data['obs_wave'] = obs_grp['wavelength'][:]
                grating_data['obs_flux'] = obs_grp['flux'][:]
                grating_data['obs_err'] = obs_grp['flux_error'][:]
                grating_data['obs_mask'] = obs_grp['mask_isfinite'][:]
                
                # Load model data
                model_grp = f[grating]['model_data']
                grating_data['model_wave'] = model_grp['wavelength'][:]
                grating_data['model_flux'] = model_grp['flux_total'][:]
                
                # Load grating metadata
                grating_data['wave_range'] = f[grating].attrs['wavelength_range_nm']
                grating_data['n_points'] = f[grating].attrs['n_points']
                
                data['gratings'][grating] = grating_data
                print(f"  {grating.upper()}: {grating_data['n_points']} points, "
                      f"{grating_data['wave_range'][0]:.0f}-{grating_data['wave_range'][1]:.0f} nm")
        
        return data

def plot_grating_spectrum(data, grating='g235h'):
    """Plot observed and model spectra for a specific grating with publication quality"""
    if grating not in data['gratings']:
        print(f"Grating {grating} not found in data")
        return
    
    grating_data = data['gratings'][grating]
    target = data['metadata']['data_info']['target']
    
    # Create figure with 3:1 height ratio
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                   gridspec_kw={'height_ratios': [3, 1]}, 
                                   sharex=True)
    
    # Filter out NaN values for plotting
    wave = grating_data['obs_wave']
    obs_flux = grating_data['obs_flux']
    obs_err = grating_data['obs_err']
    model_flux = grating_data['model_flux']
    
    valid = ~np.isnan(obs_flux) & grating_data['obs_mask']
    wave_valid = wave
    obs_flux_valid = np.where(valid, obs_flux, np.nan)
    obs_err_valid = np.where(valid, obs_err, np.nan)
    model_flux_valid = np.where(valid, model_flux, np.nan)

    # Define colors based on target (consistent with published paper)
    if target == 'TWA28':
        colors = {'data': 'k', 'model': '#D55E00'}  # Orange for TWA28
    else:  # TWA27A
        colors = {'data': 'gray', 'model': '#009E73'}  # Green for TWA27A
    
    # Plot spectra with publication colours (consistent with fig1_spec.py)
    lw = 0.9
    ax1.plot(wave_valid, obs_flux_valid, color=colors['data'], linewidth=lw, alpha=0.8, label='Data')
    ax1.plot(wave_valid, model_flux_valid, color=colors['model'], linewidth=lw, alpha=0.8, label='Model')
    
    # Format main plot (consistent with fig1_spec.py)
    y_label = r'$F_{\lambda}$' '  / 10$^{14}$ ' 'erg ' r'$\text{s}^{-1} \text{cm}^{-2} \text{nm}^{-1}$'
    ax1.set_ylabel(y_label)
    ax1.legend(loc='upper right', frameon=False, fontsize=10, handlelength=1.3)
    
    # Set y-axis to scientific notation if needed
    if np.max(obs_flux_valid) < 1e-10 or np.max(obs_flux_valid) > 1e4:
        ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Plot residuals (consistent with fig1_spec.py)
    residuals = (obs_flux_valid - model_flux_valid) / obs_flux_valid
    ax2.plot(wave_valid, residuals, color=colors['model'], linewidth=lw, alpha=0.8,
             ls='', marker='o', markersize=1.5)
    ax2.axhline(0, color='k', linewidth=0.5)
    
    # Format residuals plot
    ax2.set_xlabel(r'Wavelength / nm')
    ax2.set_ylabel(r'$\Delta F_{\lambda} / F_{\lambda}$')
    
    # Set reasonable y-limits for residuals (consistent with fig1_spec.py)
    ax2.set_ylim(-0.15, 0.15)
    
    # Add grating information as text
    wave_min, wave_max = np.min(wave_valid), np.max(wave_valid)
    ax1.text(0.02, 0.95, f'{target} - {grating.upper()}\n{wave_min:.0f}–{wave_max:.0f} nm', 
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0)  # Consistent with fig1_spec.py
    
    # Create figures directory if it doesn't exist
    os.makedirs("../figures", exist_ok=True)
    
    # Save figure with high quality
    fig_path = f"../figures/{target}_spectrum_{grating}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to {fig_path}")
    plt.close()  # Close figure to save memory

def plot_all_gratings(data):
    """Plot all gratings in a single figure for comparison"""
    target = data['metadata']['data_info']['target']
    
    # Define colors based on target
    if target == 'TWA28':
        colors = {'data': 'k', 'model': '#D55E00'}
    else:
        colors = {'data': 'gray', 'model': '#009E73'}
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                   gridspec_kw={'height_ratios': [3, 1]}, 
                                   sharex=True)
    
    # Plot each grating
    gratings = ['g140h', 'g235h', 'g395h']
    grating_colors = ['navy', 'green', 'brown']
    
    for grating, grating_color in zip(gratings, grating_colors):
        if grating in data['gratings']:
            grating_data = data['gratings'][grating]
            
            wave = grating_data['obs_wave']
            obs_flux = grating_data['obs_flux']
            model_flux = grating_data['model_flux']
            
            valid = ~np.isnan(obs_flux) & grating_data['obs_mask']
            wave_valid = wave
            obs_flux_valid = np.where(valid, obs_flux, np.nan)
            model_flux_valid = np.where(valid, model_flux, np.nan)
            
            # Plot spectra
            lw = 0.9
            ax1.plot(wave_valid, obs_flux_valid, color=colors['data'], linewidth=lw, alpha=0.8)
            ax1.plot(wave_valid, model_flux_valid, color=colors['model'], linewidth=lw, alpha=0.8)
            
            # Add grating band shading
            wave_range = grating_data['wave_range']
            ax1.axvspan(wave_range[0], wave_range[1], color=grating_color, alpha=0.12, lw=0)
            
            # Add grating label
            xc = wave_range[0] + (wave_range[1] - wave_range[0])/2
            ymax = ax1.get_ylim()[1]
            ax1.text(xc, ymax*0.9, grating.upper(), color=grating_color, fontsize=10,
                    ha='center', va='center', fontweight='bold')
            
            # Plot residuals
            residuals = (obs_flux_valid - model_flux_valid) / obs_flux_valid
            ax2.plot(wave_valid, residuals, color=colors['model'], linewidth=lw, alpha=0.8,
                     ls='', marker='o', markersize=1.5)
    
    # Format plots
    y_label = r'$F_{\lambda}$' '  / 10$^{14}$ ' 'erg ' r'$\text{s}^{-1} \text{cm}^{-2} \text{nm}^{-1}$'
    ax1.set_ylabel(y_label)
    ax1.legend(['Data', 'Model'], loc='upper right', frameon=False, fontsize=10, handlelength=1.3)
    
    ax2.axhline(0, color='k', linewidth=0.5)
    ax2.set_xlabel(r'Wavelength / nm')
    ax2.set_ylabel(r'$\Delta F_{\lambda} / F_{\lambda}$')
    ax2.set_ylim(-0.15, 0.15)
    
    # Add target information
    ax1.text(0.02, 0.95, f'{target} - All Gratings\n0.97–5.27 μm', 
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0)
    
    # Create figures directory if it doesn't exist
    os.makedirs("../figures", exist_ok=True)
    
    # Save figure
    fig_path = f"../figures/{target}_spectrum_all_gratings.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to {fig_path}")
    plt.close()

if __name__ == "__main__":
    # Example usage
    target = "TWA28"  # Change to TWA27A for the other target
    filename = f"../{target}_spectral_data.h5"
    
    print(f"Loading spectral data for {target}...")
    data = load_spectral_data(filename)
    
    print(f"\nWavelength coverage:")
    for grating in data['gratings'].keys():
        wave_range = data['gratings'][grating]['wave_range']
        n_points = data['gratings'][grating]['n_points']
        print(f"  {grating.upper()}: {n_points} points, {wave_range[0]:.0f}-{wave_range[1]:.0f} nm")
    
    # Plot individual gratings
    print(f"\nPlotting individual gratings...")
    for grating in ['g140h', 'g235h', 'g395h']:
        if grating in data['gratings']:
            print(f"Plotting {grating.upper()}...")
            plot_grating_spectrum(data, grating=grating)
    
    # Plot all gratings together
    print(f"\nPlotting all gratings together...")
    plot_all_gratings(data)
