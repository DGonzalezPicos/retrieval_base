#!/usr/bin/env python3
"""
Example script to load and plot pressure-temperature profiles from Zenodo dataset

Usage:
    python load_pt_profiles.py
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import json
import os

def load_pt_profiles(filename):
    """Load PT profile data from HDF5 file"""
    with h5py.File(filename, 'r') as f:
        # Load metadata
        metadata = json.loads(f.attrs['metadata'])
        print(f"Target: {metadata['data_info']['target']}")
        print(f"Run: {metadata['data_info']['run']}")
        
        # Load PT data
        pressure = f['pt_profiles/pressure'][:]
        temperature_envelopes = f['pt_profiles/temperature_envelopes'][:]
        integrated_contribution = f['pt_profiles/integrated_contribution'][:]
        
        return {
            'metadata': metadata,
            'pressure': pressure,
            'temperature_envelopes': temperature_envelopes,
            'integrated_contribution': integrated_contribution
        }

def plot_pt_profile(data):
    """Plot pressure-temperature profile with uncertainty envelopes"""
    target = data['metadata']['data_info']['target']
    
    # Create figures directory if it doesn't exist
    os.makedirs("../figures", exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    pressure = data['pressure']
    temp_envelopes = data['temperature_envelopes']
    
    # Plot temperature envelopes (consistent with fig_PTs.py)
    # Envelope order: 16%, 84%, 2.5%, 97.5%, 0.15%, 99.85%, median
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    alphas = [0.3, 0.2, 0.1]
    
    for i in range(3):
        ax1.fill_betweenx(pressure, temp_envelopes[i], temp_envelopes[-(i+1)], 
                         color=colors[i], alpha=alphas[i], 
                         label=f'{[68, 95, 99.7][i]}% confidence')
    
    # Plot median (consistent with fig_PTs.py)
    ax1.plot(temp_envelopes[3], pressure, 'k-', linewidth=2, label='Median')
    
    ax1.set_xlabel('Temperature (K)')
    ax1.set_ylabel('Pressure (bar)')
    ax1.set_yscale('log')
    ax1.invert_yaxis()
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Pressure-Temperature Profile')
    
    # Plot contribution function
    ax2.plot(data['integrated_contribution'], pressure, 'r-', linewidth=2)
    ax2.set_xlabel('Integrated Contribution Function')
    ax2.set_ylabel('Pressure (bar)')
    ax2.set_yscale('log')
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Contribution Function')
    
    # Add target title (consistent with fig_PTs.py)
    fig.suptitle(f'{target.replace("TWA", "TWA ")} Atmospheric Structure', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for suptitle
    
    # Save figure with high quality
    fig_path = f"../figures/{target}_pt_profile.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to {fig_path}")
    plt.close()  # Close figure to save memory

if __name__ == "__main__":
    # Example usage
    target = "TWA28"  # Change to TWA28 for the other target
    filename = f"../{target}_pt_profiles.h5"
    
    print(f"Loading PT profiles for {target}...")
    data = load_pt_profiles(filename)
    
    print(f"Pressure range: {np.min(data['pressure']):.2e} - {np.max(data['pressure']):.2e} bar")
    print(f"Temperature range: {np.min(data['temperature_envelopes']):.1f} - {np.max(data['temperature_envelopes']):.1f} K")
    
    # Plot PT profile
    plot_pt_profile(data)
