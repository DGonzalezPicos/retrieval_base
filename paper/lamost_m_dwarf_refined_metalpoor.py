"""Generate publication-quality corner plot of the LAMOST M dwarf catalog.

This module creates a 2D density plot of Teff vs [M/H] with marginal histograms
and distance-based coloring, suitable for publication.
"""

from typing import Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import BoundaryNorm
import seaborn as sns
from pathlib import Path
import pandas as pd
from scipy.stats import gaussian_kde
def setup_publication_plot() -> None:
    """Configure matplotlib settings for publication-quality plots."""
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': 300,
    })

def get_subset(df: pd.DataFrame, teff_range: Tuple[float, float], metallicity_range: Tuple[float, float], distance_range: Tuple[float, float]) -> pd.DataFrame:
    """
    Get a subset of the dataframe based on the given ranges.
    """
    return df[(df['Teff'] >= teff_range[0]) & (df['Teff'] <= teff_range[1]) &
              (df['[M/H]'] >= metallicity_range[0]) & (df['[M/H]'] <= metallicity_range[1]) &
              (df['Plx'] >= distance_range[0]) & (df['Plx'] <= distance_range[1])]
    
def nearby_low_metallicity_dwarfs(df: pd.DataFrame, 
                                  distance_range: Tuple[float, float],
                                  metallicity_range: Tuple[float, float]) -> pd.DataFrame:
    """
    Get a subset of the dataframe based on the given ranges.
    
    Example:
    >>> df = pd.read_csv('paper/data/DR10_Cycle-SN_M_dwarfs.csv')
    >>> df = nearby_low_metallicity_dwarfs(df, (0, 100), (-0.5, 0.5))
    """
    condition = (df['distance'] >= distance_range[0]) & (df['distance'] <= distance_range[1])
    condition &= (df['[M/H]'] >= metallicity_range[0]) & (df['[M/H]'] <= metallicity_range[1])
    return df[condition]

def create_diagnostic_plots(
    df: pd.DataFrame,
    color_by: str = 'Teff',
    northern_hemisphere: bool = True,
    save_path: Optional[Path] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create diagnostic plots for target selection and observing proposals.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the catalog data with columns: Teff, Logg, [M/H], Gmag, RA, Dec
    color_by : str
        Column to use for colorbar ('Teff' or 'Gmag')
    northern_hemisphere : bool
        If True, only show stars with Dec > -20 degrees
    save_path : Optional[Path]
        Path to save the figure, if provided
        
    Returns
    -------
    fig : plt.Figure
        The matplotlib figure object
    axes : np.ndarray
        Array of the figure's axes
    """
    setup_publication_plot()
    
    # Filter for northern hemisphere if requested
    if northern_hemisphere:
        df_plot = df[df['Dec'] > -20].copy()
        title_suffix = ' (Northern Hemisphere, Dec > -20°)'
    else:
        df_plot = df.copy()
        title_suffix = ' (All Sky)'
    
    # Get color values
    if color_by == 'Teff':
        color_values = df_plot['Teff'].values
        color_label = r'T$_{\mathrm{eff}}$ [K]'
        cmap = 'coolwarm'
    elif color_by == 'Gmag':
        color_values = df_plot['Gmag'].values
        color_label = 'G [mag]'
        cmap = 'viridis_r'  # Reversed so brighter = darker color
    else:
        raise ValueError(f"color_by must be 'Teff' or 'Gmag', got {color_by}")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Panel 1: Teff vs Logg
    ax1 = fig.add_subplot(gs[0, 0])
    scatter1 = ax1.scatter(
        df_plot['Teff'].values,
        df_plot['Logg'].values,
        c=color_values,
        cmap=cmap,
        alpha=0.6,
        s=15,
        rasterized=True,
        edgecolors='none'
    )
    ax1.set_xlabel(r'T$_{\mathrm{eff}}$ [K]')
    ax1.set_ylabel(r'log g [dex]')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', which='both', top=True)
    ax1.tick_params(axis='y', which='both', right=True)
    ax1.minorticks_on()
    
    # Panel 2: Teff vs [M/H]
    ax2 = fig.add_subplot(gs[0, 1])
    scatter2 = ax2.scatter(
        df_plot['Teff'].values,
        df_plot['[M/H]'].values,
        c=color_values,
        cmap=cmap,
        alpha=0.6,
        s=15,
        rasterized=True,
        edgecolors='none'
    )
    ax2.set_xlabel(r'T$_{\mathrm{eff}}$ [K]')
    ax2.set_ylabel('[M/H] [dex]')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', which='both', top=True)
    ax2.tick_params(axis='y', which='both', right=True)
    ax2.minorticks_on()
    
    # Panel 3: Gmag vs [M/H]
    ax3 = fig.add_subplot(gs[0, 2])
    scatter3 = ax3.scatter(
        df_plot['Gmag'].values,
        df_plot['[M/H]'].values,
        c=color_values,
        cmap=cmap,
        alpha=0.6,
        s=15,
        rasterized=True,
        edgecolors='none'
    )
    ax3.set_xlabel('G [mag]')
    ax3.set_ylabel('[M/H] [dex]')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', which='both', top=True)
    ax3.tick_params(axis='y', which='both', right=True)
    ax3.minorticks_on()
    ax3.invert_xaxis()  # Brighter stars on the left
    
    # Panel 4: RA vs Dec (sky plot)
    ax4 = fig.add_subplot(gs[1, :])
    scatter4 = ax4.scatter(
        df_plot['RA'].values,
        df_plot['Dec'].values,
        c=color_values,
        cmap=cmap,
        alpha=0.6,
        s=15,
        rasterized=True,
        edgecolors='none'
    )
    ax4.set_xlabel('RA [deg]')
    ax4.set_ylabel('Dec [deg]')
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', which='both', top=True)
    ax4.tick_params(axis='y', which='both', right=True)
    ax4.minorticks_on()
    if northern_hemisphere:
        ax4.axhline(y=-20, color='red', linestyle='--', linewidth=1.5, 
                   label='Northern Hemisphere Threshold')
        ax4.legend(loc='upper right', fontsize=10)
    
    # Panel 5: Distance vs [M/H]
    ax5 = fig.add_subplot(gs[2, 0])
    scatter5 = ax5.scatter(
        df_plot['distance'].values,
        df_plot['[M/H]'].values,
        c=color_values,
        cmap=cmap,
        alpha=0.6,
        s=15,
        rasterized=True,
        edgecolors='none'
    )
    ax5.set_xlabel('Distance [pc]')
    ax5.set_ylabel('[M/H] [dex]')
    ax5.grid(True, alpha=0.3)
    ax5.tick_params(axis='x', which='both', top=True)
    ax5.tick_params(axis='y', which='both', right=True)
    ax5.minorticks_on()
    ax5.set_xscale('log')
    
    # Panel 6: Gmag vs Distance
    ax6 = fig.add_subplot(gs[2, 1])
    scatter6 = ax6.scatter(
        df_plot['Gmag'].values,
        df_plot['distance'].values,
        c=color_values,
        cmap=cmap,
        alpha=0.6,
        s=15,
        rasterized=True,
        edgecolors='none'
    )
    ax6.set_xlabel('G [mag]')
    ax6.set_ylabel('Distance [pc]')
    ax6.grid(True, alpha=0.3)
    ax6.tick_params(axis='x', which='both', top=True)
    ax6.tick_params(axis='y', which='both', right=True)
    ax6.minorticks_on()
    ax6.invert_xaxis()
    ax6.set_yscale('log')
    
    # Panel 7: Teff vs Distance
    ax7 = fig.add_subplot(gs[2, 2])
    scatter7 = ax7.scatter(
        df_plot['Teff'].values,
        df_plot['distance'].values,
        c=color_values,
        cmap=cmap,
        alpha=0.6,
        s=15,
        rasterized=True,
        edgecolors='none'
    )
    ax7.set_xlabel(r'T$_{\mathrm{eff}}$ [K]')
    ax7.set_ylabel('Distance [pc]')
    ax7.grid(True, alpha=0.3)
    ax7.tick_params(axis='x', which='both', top=True)
    ax7.tick_params(axis='y', which='both', right=True)
    ax7.minorticks_on()
    ax7.set_yscale('log')
    
    # Add colorbar at the bottom
    # Create a new axis for the colorbar
    cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.015])
    cbar = plt.colorbar(scatter4, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(color_label, fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    # Add overall title
    fig.suptitle(f'LAMOST M Dwarf Catalog Diagnostic Plots{title_suffix}', 
                fontsize=16, y=0.995)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f'Saved diagnostic plots to {save_path}')
        plt.savefig(save_path.with_suffix('.png'), bbox_inches='tight', dpi=300)
    
    return fig, np.array([ax1, ax2, ax3, ax4, ax5, ax6, ax7])

if __name__ == '__main__':
    # Read data from catalog
    df = pd.read_csv('paper/data/DR10_Cycle-SN_M_dwarfs.csv')
    teff = df['Teff'].values
    metallicity = df['[M/H]'].values
    parallax = df['Plx'].values
    df['distance'] = 1000 / parallax # in pc
    n_all = len(df)
    # Create output directory if it doesn't exist
    output_dir = Path('paper/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # diagnostic plot with panels: Teff, Logg, [M/H], Gmag, Ra, Dec
    # northern hemisphere threshold at dec > -20
    
    # apply threshold
    threshold = {
        'Gmag':(10,12.5),
        '[M/H]':(-2.0,-0.6),
    }
    
    suffix = ''
    if len(threshold) > 0:
        df = df[(df['Gmag'] >= threshold['Gmag'][0]) & (df['Gmag'] <= threshold['Gmag'][1]) &
                (df['[M/H]'] >= threshold['[M/H]'][0]) & (df['[M/H]'] <= threshold['[M/H]'][1])]
        suffix = f"_gmag{threshold['Gmag'][0]:.1f}-{threshold['Gmag'][1]:.1f}_mhm{threshold['[M/H]'][0]:.1f}-{threshold['[M/H]'][1]:.1f}"
    n_sources = len(df)
    if n_sources == 0:
        print('No sources found within the threshold')
        exit()
        
    print(f'Number of sources within the threshold: {n_sources}/{n_all}')
    
    # Generate diagnostic plots with Teff as colorbar
    fig_teff, axes_teff = create_diagnostic_plots(
        df=df,
        color_by='Teff',
        northern_hemisphere=True,
        save_path=output_dir / f'lamost_mdwarf_diagnostic_teff{suffix}.pdf'
    )
    plt.close(fig_teff)
    
    # Generate diagnostic plots with Gmag as colorbar
    fig_gmag, axes_gmag = create_diagnostic_plots(
        df=df,
        color_by='Gmag',
        northern_hemisphere=True,
        save_path=output_dir / f'lamost_mdwarf_diagnostic_gmag{suffix}.pdf'
    )
    plt.close(fig_gmag)
    