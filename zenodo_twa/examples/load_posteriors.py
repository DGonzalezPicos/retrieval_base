#!/usr/bin/env python3
"""
Example script to load and analyze posterior distributions from Zenodo dataset

Usage:
    python load_posteriors.py
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import json
import os

def load_posteriors(filename):
    """Load posterior data from HDF5 file"""
    with h5py.File(filename, 'r') as f:
        # Load metadata
        metadata = json.loads(f.attrs['metadata'])
        print(f"Target: {metadata['data_info']['target']}")
        print(f"Run: {metadata['data_info']['run']}")
        
        # Load parameter names
        param_keys = [k.decode('utf-8') for k in f['parameters/param_keys'][:]]
        
        # Load posterior samples
        posterior_dict = {}
        for param in param_keys:
            posterior_dict[param] = f[f'posterior_samples/{param}'][:]
        
        return {
            'metadata': metadata,
            'param_keys': param_keys,
            'posterior_dict': posterior_dict
        }

def plot_corner_subset(data, params_to_plot=None):
    """Plot corner plot for a subset of parameters"""
    target = data['metadata']['data_info']['target']
    
    if params_to_plot is None:
        # Default parameters to plot
        params_to_plot = ['R_p', 'T_0', 'log_g', 'log_12CO/13CO']
    
    # Create figures directory if it doesn't exist
    os.makedirs("../figures", exist_ok=True)
    
    # Try to use corner.py for professional corner plots
    try:
        import corner
        
        # Filter available parameters
        available_params = [p for p in params_to_plot if p in data['param_keys']]
        
        # Prepare data for corner plot
        samples_array = np.column_stack([data['posterior_dict'][param] for param in available_params])
        
        # Create corner plot with publication styling (consistent with fig_corner.py)
        fig = corner.corner(samples_array, labels=available_params, 
                          quantiles=[0.16, 0.5, 0.84], show_titles=True, 
                          title_kwargs={"fontsize": 12}, label_kwargs={"fontsize": 12},
                          color='#1f77b4', smooth=True, smooth1d=True)
        
        # Add target title
        fig.suptitle(f'{target.replace("TWA", "TWA ")} Posterior Distributions', 
                    fontsize=14, fontweight='bold')
        plt.subplots_adjust(top=0.93)
        
        print("Using corner.py for professional corner plot")
        
    except ImportError:
        print("corner.py not available, using basic corner plot implementation")
        
        # Filter available parameters
        available_params = [p for p in params_to_plot if p in data['param_keys']]
        n_params = len(available_params)
        
        fig, axes = plt.subplots(n_params, n_params, figsize=(12, 12))
        
        for i, param_y in enumerate(available_params):
            for j, param_x in enumerate(available_params):
                ax = axes[i, j] if n_params > 1 else axes
                
                if i == j:
                    # Diagonal: histograms
                    samples = data['posterior_dict'][param_y]
                    ax.hist(samples, bins=50, alpha=0.7, density=True, color='#1f77b4')
                    ax.set_ylabel('Density')
                    if i == n_params - 1:
                        ax.set_xlabel(param_x)
                        
                elif i > j:
                    # Lower triangle: scatter plots
                    x_samples = data['posterior_dict'][param_x]
                    y_samples = data['posterior_dict'][param_y]
                    ax.scatter(x_samples[::10], y_samples[::10], alpha=0.5, s=1, color='#1f77b4')
                    
                    if i == n_params - 1:
                        ax.set_xlabel(param_x)
                    if j == 0:
                        ax.set_ylabel(param_y)
                else:
                    # Upper triangle: hide
                    ax.set_visible(False)
        
        # Add target title
        fig.suptitle(f'{target.replace("TWA", "TWA ")} Posterior Distributions', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
    
    # Save figure with high quality
    fig_path = f"../figures/{target}_corner_plot.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to {fig_path}")
    plt.close()  # Close figure to save memory

def print_parameter_summary(data):
    """Print summary statistics for all parameters"""
    print("\nParameter Summary:")
    print("-" * 60)
    print(f"{'Parameter':<20} {'Median':<12} {'16%':<10} {'84%':<10}")
    print("-" * 60)
    
    for param in data['param_keys']:
        samples = data['posterior_dict'][param]
        median = np.median(samples)
        p16 = np.percentile(samples, 16)
        p84 = np.percentile(samples, 84)
        
        print(f"{param:<20} {median:<12.3f} {p16:<10.3f} {p84:<10.3f}")

if __name__ == "__main__":
    # Example usage
    target = "TWA28"  # Change to TWA28 for the other target
    filename = f"../{target}_posteriors.h5"
    
    print(f"Loading posteriors for {target}...")
    data = load_posteriors(filename)
    
    print(f"Number of parameters: {len(data['param_keys'])}")
    print(f"Number of samples: {len(data['posterior_dict'][data['param_keys'][0]])}")
    
    # Print parameter summary
    print_parameter_summary(data)
    
    # Plot corner plot for key parameters
    plot_corner_subset(data)
