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

def create_corner_plot(
    teff: np.ndarray,
    metallicity: np.ndarray,
    distance: np.ndarray,
    save_path: Optional[Path] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create a corner plot showing Teff vs [M/H] with marginal distributions.
    
    Parameters
    ----------
    teff : np.ndarray
        Effective temperature values in Kelvin
    metallicity : np.ndarray
        Metallicity values [M/H] (dimensionless)
    distance : np.ndarray
        Distance values in parsecs
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
    
    # Create figure with custom gridspec
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(3, 3)
    
    # Create discrete distance bins every 50 pc
    max_dist = np.ceil(np.nanmax(distance) / 50) * 50
    print(f'Max distance: {max_dist} pc')
    bins = np.arange(0, max_dist + 50, 50)
    norm = BoundaryNorm(bins, plt.cm.viridis.N)
    
    # Main scatter plot
    ax_main = fig.add_subplot(gs[1:, :-1])
    scatter = ax_main.scatter(
        teff, 
        metallicity,
        c=distance,
        cmap='viridis',
        norm=norm,
        alpha=0.5,
        s=20,
        rasterized=True  # Better for PDF output
    )
    
    # Top histogram
    ax_hist_x = fig.add_subplot(gs[0, :-1], sharex=ax_main)
    ax_hist_x.hist(teff, bins=50, color='darkblue', alpha=0.6, density=True)
    ax_hist_x.set_ylabel('Density')
    
    # Right histogram
    ax_hist_y = fig.add_subplot(gs[1:, -1], sharey=ax_main)
    ax_hist_y.hist(metallicity, bins=50, color='darkblue', alpha=0.6,
                   orientation='horizontal', density=True)
    ax_hist_y.set_xlabel('Density')
    
    # Colorbar with discrete bins
    ax_cbar = fig.add_subplot(gs[0, -1])
    ax_cbar.set_position([ax_cbar.get_position().x0 + 0.05,
                         ax_cbar.get_position().y0,
                         ax_cbar.get_position().width * 0.6,
                         ax_cbar.get_position().height])
    cbar = plt.colorbar(scatter, cax=ax_cbar, orientation='horizontal',
                       label='Distance [pc]', ticks=bins[::4])
    
    # Labels and formatting
    ax_main.set_xlabel(r'T$_{\mathrm{eff}}$ [K]')
    ax_main.set_ylabel('[M/H]')
    
    # Remove top and right spines from histograms
    ax_hist_x.spines['top'].set_visible(False)
    ax_hist_x.spines['right'].set_visible(False)
    ax_hist_y.spines['top'].set_visible(False)
    ax_hist_y.spines['right'].set_visible(False)
    
    # Remove histogram labels
    ax_hist_x.set_title('')
    ax_hist_x.set_xlabel('')
    ax_hist_y.set_ylabel('')
    
    # Adjust layout
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f'Saved figure to {save_path}')
        plt.savefig(save_path.with_suffix('.png'), bbox_inches='tight', dpi=300)
        
    return fig, np.array([ax_main, ax_hist_x, ax_hist_y])

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

if __name__ == '__main__':
    # Read data from catalog
    df = pd.read_csv('paper/data/DR10_Cycle-SN_M_dwarfs.csv')
    teff = df['Teff'].values
    metallicity = df['[M/H]'].values
    parallax = df['Plx'].values
    df['distance'] = 1000 / parallax # in pc
    
    # Create output directory if it doesn't exist
    output_dir = Path('paper/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate and save the plot
    fig, axes = create_corner_plot(
        teff=teff,
        metallicity=metallicity,
        distance=df['distance'].values,
        save_path=output_dir / 'lamost_mdwarf_corner.pdf'
    )
    plt.close()
    
    # get subset of dwarfs with low metallicity and distance < 200 pc
    from matplotlib.colors import ListedColormap, BoundaryNorm
    
    distance_bins = np.arange(100, 1000+100, 100)
    metallicity_thresholds = [-0.5, -0.6, -0.7, -0.8, -0.9, -1.0]
    
    # Create discrete colors from viridis colormap
    n_colors = len(metallicity_thresholds)
    colors = plt.cm.viridis(np.linspace(0, 1, n_colors))
    discrete_cmap = ListedColormap(colors)
    
    # Create boundaries for the colorbar
    bounds = metallicity_thresholds + [metallicity_thresholds[-1] - 0.1]  # Add one more boundary
    norm = BoundaryNorm(bounds, discrete_cmap.N)
    
    numbers = np.zeros((len(metallicity_thresholds), len(distance_bins)))
    for i, metallicity_threshold in enumerate(metallicity_thresholds):
        for j in range(len(distance_bins)):
            df_subset = nearby_low_metallicity_dwarfs(df, (0, distance_bins[j]), (-4.0, metallicity_threshold))
            numbers[i, j] = len(df_subset)
    
    # Create cumulative plot of metal-poor dwarfs vs distance
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create colormap and normalize for metallicity thresholds
    sm = plt.cm.ScalarMappable(cmap=discrete_cmap, norm=norm)
    sm.set_array([])  # Required for older matplotlib versions
    
    cumulative_numbers = np.array(numbers)
    for i, threshold in enumerate(metallicity_thresholds):
        cumulative_numbers[i, :] = np.cumsum(cumulative_numbers[i, :])
        color = colors[i]  # Use the discrete colors directly
        line = ax.plot(distance_bins, cumulative_numbers[i, :], 
                      color=color, linewidth=2, 
                      label=f'[M/H] < {threshold:.1f}')
        ax.scatter(distance_bins, cumulative_numbers[i, :], color=color, s=50)
    
    # Configure axes
    ax.set_xlabel('Distance (pc)')
    ax.set_ylabel('Number of Metal-Poor M Dwarfs')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar with discrete colors
    cbar = plt.colorbar(sm, ax=ax, 
                       ticks=metallicity_thresholds,
                       boundaries=bounds,
                       spacing='proportional',
                       format='%.1f')
    cbar.set_label('[M/H] Threshold', fontsize=12)
    
    # Adjust colorbar label positions to center them between boundaries
    cbar.ax.set_yticklabels([f'{x:.1f}' for x in metallicity_thresholds])
    
    # Add title
    ax.set_title('Cumulative Distribution of Metal-Poor M Dwarfs with Distance', pad=15)
    
    # Adjust layout to accommodate colorbar
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_dir / 'metal_poor_cumulative.pdf', 
                bbox_inches='tight', dpi=300)
    plt.savefig(output_dir / 'metal_poor_cumulative.png', 
                bbox_inches='tight', dpi=300)
    print(f'Saved figure to {output_dir / "metal_poor_cumulative.pdf"}')
    plt.close()

    ## Plot the cumulative distribution of metal-poor dwarfs as a of metallicity
    # consider different tracks for distance ranges
    # distance_range = [50, 100, 200, 400, 800]
    distance_range = np.arange(100, 1000+100, 100)
    
    # Create metallicity bins for analysis
    metallicity_bins = np.linspace(-1.2, 0.5, 50)  # Adjust range based on your data
    numbers_cumulative = np.zeros((len(distance_range), len(metallicity_bins)-1))
    numbers = np.zeros((len(distance_range), len(metallicity_bins)-1))
    
    # Calculate number of sources for each distance and metallicity bin
    for i, distance_threshold in enumerate(distance_range):
        df_subset = df[df['distance'] <= distance_threshold]
        print(f'Number of sources in distance range {distance_threshold} pc: {len(df_subset)}')
        hist, _ = np.histogram(df_subset['[M/H]'], bins=metallicity_bins)
        numbers[i, :] = hist
        numbers_cumulative[i, :] = np.cumsum(hist)
    # save numbers to file as txt with useful header 
    # header = f'Cumulative number of M dwarfs with metallicity < [M/H] for different distance ranges\n'
    # header += f'Distance ranges: {distance_range}\n'
    # header += f'Metallicity bins: {metallicity_bins}\n'
    # np.savetxt(output_dir / 'metal_poor_cumulative_distance.txt', numbers.T, header=header)
    # print(f'Saved cumulative number of M dwarfs to {output_dir / "metal_poor_cumulative_distance.txt"}')
    
    # also save as h5, save also the distance range and metallicity bins
    import h5py
    h5_file = h5py.File(output_dir / 'metal_poor_cumulative_distance.h5', 'w')
    h5_file.create_dataset('numbers', data=numbers)
    h5_file.create_dataset('numbers_cumulative', data=numbers_cumulative)
    h5_file.create_dataset('distance_range', data=distance_range)
    h5_file.create_dataset('metallicity_bins', data=metallicity_bins)
    h5_file.close()
    print(f'Saved cumulative number of M dwarfs to {output_dir / "metal_poor_cumulative_distance.h5"}')
    
    # example on how to load the h5 file
    """
    h5_file = h5py.File(output_dir / 'metal_poor_cumulative_distance.h5', 'r')
    numbers = h5_file['numbers'][:]
    numbers_cumulative = h5_file['numbers_cumulative'][:]
    distance_range = h5_file['distance_range'][:]
    metallicity_bins = h5_file['metallicity_bins'][:]
    """
    
    # Create figure with discrete colors for distance ranges
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create color map for distance ranges
    colors = plt.cm.viridis(np.linspace(0, 1, len(distance_range)))
    
    # Plot cumulative tracks for each distance bin
    for i, distance_threshold in enumerate(distance_range):
        color = colors[i]
        line = ax.plot(metallicity_bins[:-1], numbers_cumulative[i, :], 
                      color=color, linewidth=2, 
                      label=f'd < {distance_threshold:.0f} pc')
        ax.scatter(metallicity_bins[:-1], numbers_cumulative[i, :], color=color, s=20, alpha=0.5)
    
    # Configure axes
    ax.set_xlabel('[M/H]')
    ax.set_ylabel('Cumulative Number of M Dwarfs')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Add legend with two columns
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', ncol=1)
    ax.set_ylim(1,None)
    
    # Add title
    ax.set_title('Cumulative Distribution of M Dwarfs by Metallicity and Distance', pad=15)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_dir / 'metallicity_distribution.pdf', 
                bbox_inches='tight', dpi=300)
    plt.savefig(output_dir / 'metallicity_distribution.png', 
                bbox_inches='tight', dpi=300)
    print(f'Saved figure to {output_dir / "metallicity_distribution.pdf"}')
    plt.close()
    
    
    
    ## plot histograms of metallicity density for different distance ranges
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate bin width for proper normalization
    bin_width = np.diff(metallicity_bins)[0]
    
    kde_scaled_values = []
    
    distance_range = np.arange(100, 1000+100, 100)[::-1]
    for i, distance_threshold in enumerate(distance_range):
        df_subset = df[df['distance'] <= distance_threshold]
        
        # Calculate histogram values for proper scaling of KDE
        hist_values, _ = np.histogram(df_subset['[M/H]'], bins=metallicity_bins)
        
        # Plot histogram
        ax.hist(df_subset['[M/H]'], bins=metallicity_bins, 
                color=colors[i], alpha=0.5, 
                label=f'd < {distance_threshold:.0f} pc',
                density=False)  # Use density for proper normalization
        
        # Compute KDE with optimal bandwidth using Scott's rule
        kde = gaussian_kde(df_subset['[M/H]'], bw_method='scott')
        
        # Generate points for smooth KDE curve
        x_kde = np.linspace(metallicity_bins[0], metallicity_bins[-1], 200)
        
        # Scale KDE to match histogram height
        kde_values = kde(x_kde)
        scaling_factor = len(df_subset) * bin_width  # Scale factor for count-based normalization
        
        # Plot KDE
        ax.plot(x_kde, kde_values * scaling_factor, 
                color=colors[i], linewidth=2,
                label=f'KDE (d < {distance_threshold:.0f} pc)')
        kde_scaled_values.append(kde_values * scaling_factor)
        
    # save relevant variables to h5 file
    h5_file = h5py.File(output_dir / 'metallicity_density_distance_histogram.h5', 'w')
    h5_file.create_dataset('kde_scaled_values', data=np.array(kde_scaled_values))
    h5_file.create_dataset('x_kde', data=x_kde)
    h5_file.create_dataset('metallicity_bins', data=metallicity_bins)
    h5_file.create_dataset('distance_range', data=distance_range)
    h5_file.close()
    print(f'Saved relevant variables to {output_dir / "metallicity_density_distance_histogram.h5"}')
    
    ax.set_xlabel('[M/H]')
    ax.set_ylabel('Number of stars')
    # ax.legend(fontsize=10, title='Distance ranges')
    ax.grid(True, alpha=0.3)
    # plt.show()
    fig_name = output_dir / 'metallicity_density_distance_histogram.pdf'
    # plt.savefig(fig_name, bbox_inches='tight', dpi=300)
    # print(f'Saved figure to {fig_name}')
    plt.close()
