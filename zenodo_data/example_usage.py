"""Example script demonstrating how to load and use the zenodo data files"""

import numpy as np
import matplotlib.pyplot as plt
import pathlib
from typing import Dict, Tuple

# Set up the data directory
data_dir = pathlib.Path(__file__).parent
colors = {'TWA27A': '#009E73', 'TWA28': '#D55E00'}

def load_grating_data(target: str, grating: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load data for a specific target and grating.
    
    Parameters
    ----------
    target : str
        Target name (TWA27A or TWA28)
    grating : str
        Grating name (G140H, G235H, or G395H)
        
    Returns
    -------
    tuple
        (wavelength, flux, flux_err, model, blackbody) arrays
    """
    filename = f"{target}_{grating}.dat"
    filepath = data_dir / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    data = np.loadtxt(filepath)
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]


def plot_grating_comparison(target: str, grating: str, ax: plt.Axes = None,
                            save_fig: bool = False, fig_name: str = None) -> plt.Axes:
    """
    Plot data for a specific target and grating.
    
    Parameters
    ----------
    target : str
        Target name
    grating : str
        Grating name
    ax : plt.Axes, optional
        Matplotlib axes to plot on
        
    Returns
    -------
    plt.Axes
        The axes with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    color = colors[target]
    # Load data
    wavelength, flux, flux_err, model, blackbody = load_grating_data(target, grating)
    
    # Plot data
    ax[0].errorbar(wavelength, flux, yerr=flux_err, 
               label='Observed', alpha=0.7, fmt='o', markersize=2, capsize=1, color='black',
               markerfacecolor='none', zorder=-1)
    ax[0].plot(wavelength, model, label='Best Fit Model', linewidth=1.0, alpha=0.8, color=color)
    ax[0].plot(wavelength, blackbody, label='Blackbody Model', linewidth=1.0, 
            linestyle='--', alpha=0.8, color=color)
    
    # Customize plot
    ax[0].set_xlabel('Wavelength (nm)')
    ax[0].set_ylabel('Flux (erg/s/cm²/nm)')
    ax[0].set_title(f'{target} - {grating}')
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    
    ax[1].scatter(wavelength, (flux - model), label='Residuals', linewidth=1.0, alpha=0.3, color='black',s=2)
    ax[1].axhline(0, color=color, linestyle='-', alpha=0.8, linewidth=0.8)
    ax[1].set_xlabel('Wavelength (nm)')
    ax[1].set_xlim(wavelength.min(), wavelength.max())
    ax[1].set_ylabel('Flux Residuals (erg/s/cm²/nm)')
    ax[1].set_title(f'{target} - {grating} Flux Residuals')
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    
    if save_fig:
        plt.savefig(data_dir / fig_name, dpi=150, bbox_inches='tight')
        print(f"Saved example plot to: {data_dir / fig_name}")
        plt.close()
    
    return ax


def plot_all_targets_grating(grating: str):
    """
    Plot comparison of both targets for a specific grating.
    
    Parameters
    ----------
    grating : str
        Grating name to plot
    """
    
    plot_grating_comparison('TWA27A', grating, save_fig=True, fig_name=f'TWA27A_{grating}.png')
    plot_grating_comparison('TWA28', grating, save_fig=True, fig_name=f'TWA28_{grating}.png')



def calculate_statistics(target: str, grating: str) -> Dict:
    """
    Calculate basic statistics for a target and grating.
    
    Parameters
    ----------
    target : str
        Target name
    grating : str
        Grating name
        
    Returns
    -------
    dict
        Dictionary with calculated statistics
    """
    wavelength, flux, flux_err, model, blackbody = load_grating_data(target, grating)
    
    # Calculate residuals
    residuals = (flux - model) / flux
    
    stats = {
        'n_points': len(wavelength),
        'wavelength_range': (np.nanmin(wavelength), np.nanmax(wavelength)),
        'mean_flux': np.nanmean(flux),
        'mean_error': np.nanmean(flux_err),
        'mean_residual': np.nanmean(residuals),
        'std_residual': np.nanstd(residuals),
        'snr_mean': np.nanmean(flux / flux_err),
        'model_blackbody_ratio': np.nanmean(model / blackbody),
    }
    
    return stats


def main():
    """Main function demonstrating data usage."""
    print("Zenodo Data Usage Example")
    print("=" * 50)
    
    # List available files
    data_files = list(data_dir.glob("*.dat"))
    print(f"Found {len(data_files)} data files:")
    for f in sorted(data_files):
        print(f"  - {f.name}")
    
    print("\n" + "=" * 50)
    print("Statistics Summary")
    print("=" * 50)
    
    # Calculate and display statistics for all files
    for target in ['TWA27A', 'TWA28']:
        for grating in ['G140H', 'G235H', 'G395H']:
            try:
                stats = calculate_statistics(target, grating)
                print(f"\n{target} {grating}:")
                print(f"  Data points: {stats['n_points']:,}")
                print(f"  Wavelength range: {stats['wavelength_range'][0]:.1f} - {stats['wavelength_range'][1]:.1f} nm")
                print(f"  Mean flux: {stats['mean_flux']:.2e} erg/s/cm²/nm")
                print(f"  Mean S/N: {stats['snr_mean']:.1f}")
                print(f"  Mean residual: {stats['mean_residual']:.3f} ± {stats['std_residual']:.3f}")
                print(f"  Model/BB ratio: {stats['model_blackbody_ratio']:.1f}")
            except FileNotFoundError:
                print(f"\n{target} {grating}: File not found")
                
            try:    
                plot_grating_comparison(target, grating, save_fig=True, fig_name=f'{target}_{grating}.png')
            except FileNotFoundError:
                print(f"\n{target} {grating}: File not found")
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main() 