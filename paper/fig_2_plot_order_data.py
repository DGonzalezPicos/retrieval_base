#!/usr/bin/env python3
"""
Minimal script to read and plot order data from text files.
Reads the four vectors: wavelength, data, best fit model, model 13CO, model C18O
"""

import numpy as np
import matplotlib.pyplot as plt
import pathlib
from typing import Dict, List, Tuple

def load_order_data(data_dir: pathlib.Path) -> Dict[int, np.ndarray]:
    """
    Load order data from text files.
    
    Parameters:
    -----------
    data_dir : pathlib.Path
        Directory containing the order data text files
        
    Returns:
    --------
    Dict[int, np.ndarray]
        Dictionary mapping order number to data array with columns:
        [wavelength, data, best_fit_model, model_no_13CO, model_no_C18O]
    """
    order_data = {}
    
    for order in [0, 1, 2]:
        file_path = data_dir / f'order_{order}_data.txt'
        if file_path.exists():
            # Load data, skipping header lines
            data = np.loadtxt(file_path, comments='#')
            order_data[order] = data
            print(f"Loaded order {order}: {data.shape[0]} data points")
        else:
            print(f"Warning: {file_path} not found")
    
    return order_data

def plot_order_data(order_data: Dict[int, np.ndarray], 
                   target_name: str = "Unknown",
                   save_plot: bool = True,
                   output_dir: pathlib.Path = None) -> None:
    """
    Plot the order data with data and models.
    
    Parameters:
    -----------
    order_data : Dict[int, np.ndarray]
        Dictionary containing order data
    target_name : str
        Name of the target for plot titles
    save_plot : bool
        Whether to save the plot
    output_dir : pathlib.Path
        Directory to save the plot (if save_plot is True)
    """
    # Set up the plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f'Spectra for {target_name}', fontsize=14)
    
    # Colors for different components
    colors = ['black', 'orange', 'seagreen', 'red']
    labels = ['Data', 'Best-fit model', 'Model no 13CO', 'Model no C18O']
    
    # Plot each order
    for i, order in enumerate([0, 1, 2]):
        if order not in order_data:
            continue
            
        ax = axes[i]
        data = order_data[order]
        
        # Extract vectors
        wavelength = data[:, 0]  # nm
        flux_data = data[:, 1]
        best_fit = data[:, 2]
        no_13co = data[:, 3]
        no_c18o = data[:, 4]
        
        # Plot data
        ax.plot(wavelength, flux_data, color=colors[0], lw=1.0, label=labels[0], alpha=0.8)
        ax.plot(wavelength, best_fit, color=colors[1], lw=1.5, label=labels[1])
        ax.plot(wavelength, no_13co, color=colors[2], lw=1.0, label=labels[2], alpha=0.8)
        ax.plot(wavelength, no_c18o, color=colors[3], lw=1.0, label=labels[3], alpha=0.8)
        
        # Set labels and title
        ax.set_ylabel('Normalized flux')
        ax.set_title(f'Order {order}')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Set y-axis limits
        all_flux = np.concatenate([flux_data, best_fit, no_13co, no_c18o])
        y_min, y_max = np.nanmin(all_flux), np.nanmax(all_flux)
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
    
    # Set x-axis label for the bottom plot
    axes[-1].set_xlabel('Wavelength (nm)')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if requested
    if save_plot and output_dir is not None:
        output_dir.mkdir(exist_ok=True)
        plot_file = output_dir / f'{target_name}_order_plots.pdf'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {plot_file}")
    
    plt.show()

def main():
    """Main function to run the plotting script."""
    # Configuration
    target_name = "gl205"  # Change this to your target
    base_path = pathlib.Path('/home/dario/phd/retrieval_base')
    
    # Find the data directory
    outputs_dir = base_path / target_name / 'retrieval_outputs'
    dirs = [d for d in outputs_dir.iterdir() if d.is_dir() and 'fc' in d.name and '_' not in d.name]
    
    if len(dirs) == 0:
        print(f"No runs found in {outputs_dir}")
        return
    
    # Use the latest run
    latest_run = max(dirs, key=lambda x: int(x.name.split('fc')[-1]))
    data_dir = latest_run / 'test_output' / 'order_data'
    
    if not data_dir.exists():
        print(f"Data directory {data_dir} not found.")
        print("Please run save_order_data.py first to generate the text files.")
        return
    
    print(f"Loading data from {data_dir}")
    
    # Load the data
    order_data = load_order_data(data_dir)
    
    if not order_data:
        print("No order data found.")
        return
    
    # Plot the data
    plot_order_data(
        order_data=order_data,
        target_name=target_name,
        save_plot=True,
        output_dir=data_dir
    )

if __name__ == '__main__':
    main()
