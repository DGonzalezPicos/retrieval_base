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

def read_lamost_refined_catalog(file_path: Path) -> pd.DataFrame:
    """
    Read the LAMOST refined catalog from fixed-width format text file.
    
    Parameters
    ----------
    file_path : Path
        Path to the lamost_mdwarfs_refined2025.txt file
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns: obsid, RA, Dec, Teff, Logg, [M/H], Gmag, distance
    """
    # Fixed-width column specifications based on the file format
    # Note: positions are 0-indexed in Python, but file format is 1-indexed
    colspecs = [
        (0, 10),    # obsid (bytes 1-10)
        (11, 22),   # RAdeg (bytes 12-22)
        (23, 33),   # DEdeg (bytes 24-33)
        (34, 42),   # snr-u (bytes 35-42)
        (43, 51),   # snr-g (bytes 44-51)
        (52, 60),   # snr-r (bytes 53-60)
        (61, 69),   # snr-i (bytes 62-69)
        (70, 78),   # snr-z (bytes 71-78)
        (79, 85),   # orp-u (bytes 80-85)
        (86, 92),   # orp-g (bytes 87-92)
        (93, 99),   # orp-r (bytes 94-99)
        (100, 106), # orp-i (bytes 101-106)
        (107, 113), # orp-z (bytes 108-113)
        (114, 118), # subclass (bytes 115-118)
        (119, 129), # LClass (bytes 120-129)
        (130, 137), # RVel (bytes 131-137)
        (138, 146), # errRVel (bytes 139-146)
        (147, 162), # r_RVel (bytes 148-162)
        (163, 167), # Teff (bytes 164-167)
        (168, 171), # e_Teff (bytes 169-171)
        (172, 178), # logg (bytes 173-178)
        (179, 184), # e_logg (bytes 180-184)
        (185, 191), # Z ([M/H]) (bytes 186-191)
        (192, 197), # e_Z (bytes 193-197)
        (198, 204), # alpha (bytes 199-204)
        (205, 210), # e_alpha (bytes 206-210)
        (211, 212), # n_RVel (bytes 212-212)
        (213, 214), # Param (bytes 214-214)
        (215, 234), # Gaia (bytes 216-234)
        (235, 243), # Rgeo (distance in pc) (bytes 236-243)
        (244, 249), # E(B-V) (bytes 245-249)
        (250, 256), # Bp-Rp (bytes 251-256)
        (257, 263), # GMag (absolute magnitude) (bytes 258-263)
        (264, 265), # CMDFlag (bytes 265-265)
        (266, 271), # CMDPos (bytes 267-271)
        (272, 278), # W1-W2 (bytes 273-278)
        (279, 284), # errW1 (bytes 280-284)
        (285, 290), # errW2 (bytes 286-290)
    ]
    
    column_names = [
        'obsid', 'RAdeg', 'DEdeg', 'snr_u', 'snr_g', 'snr_r', 'snr_i', 'snr_z',
        'orp_u', 'orp_g', 'orp_r', 'orp_i', 'orp_z', 'subclass', 'LClass',
        'RVel', 'errRVel', 'r_RVel', 'Teff', 'e_Teff', 'logg', 'e_logg',
        'Z', 'e_Z', 'alpha', 'e_alpha', 'n_RVel', 'Param', 'Gaia', 'Rgeo',
        'E_BV', 'Bp_Rp', 'GMag', 'CMDFlag', 'CMDPos', 'W1_W2', 'errW1', 'errW2'
    ]
    
    # Read the file, skipping header lines
    # Find where the data starts (after the column description line with '---')
    with open(file_path, 'r') as f:
        lines = f.readlines()
        data_start = 0
        for i, line in enumerate(lines):
            if '---' in line and i > 30:  # Look for separator line after header
                data_start = i + 1
                break
    
    # Read the data using fixed-width format
    try:
        df = pd.read_fwf(
            file_path,
            colspecs=colspecs,
            names=column_names,
            skiprows=data_start,
            na_values=['', ' ', 'NaN', 'nan', 'NULL', 'null', '...'],
            skip_blank_lines=True
        )
    except Exception as e:
        # Fallback: try reading as whitespace-separated
        print(f"Warning: Fixed-width reading failed: {e}")
        print("Attempting to read as whitespace-separated...")
        df = pd.read_csv(
            file_path,
            sep=r'\s+',
            skiprows=data_start,
            names=column_names,
            na_values=['', ' ', 'NaN', 'nan', 'NULL', 'null', '...']
        )
    
    # Rename columns to match expected format
    df = df.rename(columns={
        'RAdeg': 'RA',
        'DEdeg': 'Dec',
        'logg': 'Logg',
        'Z': '[M/H]',
        'Rgeo': 'distance'
    })
    
    # Convert to numeric, handling any string values
    numeric_cols = ['RA', 'Dec', 'Teff', 'Logg', '[M/H]', 'GMag', 'distance']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calculate apparent G magnitude from absolute magnitude and distance
    # m = M + 5*log10(d/10) where d is in parsecs
    valid_distance = (df['distance'] > 0) & df['distance'].notna()
    valid_gmag = df['GMag'].notna()
    valid_mask = valid_distance & valid_gmag
    
    df['Gmag'] = np.nan
    df.loc[valid_mask, 'Gmag'] = (
        df.loc[valid_mask, 'GMag'] + 
        5 * np.log10(df.loc[valid_mask, 'distance'] / 10.0)
    )
    
    return df

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
    refined = True
    ignore_giants = True
    if refined:
        df = read_lamost_refined_catalog(Path('paper/data/lamost_mdwarfs_refined2025.txt'))
    else:
        df = pd.read_csv('paper/data/DR10_Cycle-SN_M_dwarfs.csv')
        teff = df['Teff'].values
        metallicity = df['[M/H]'].values
        parallax = df['Plx'].values
        df['distance'] = 1000 / parallax # in pc
        
    if ignore_giants:
        df = df[df['Logg'] > 4.0]
        
    n_all = len(df)
    # Create output directory if it doesn't exist
    output_dir = Path('paper/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # diagnostic plot with panels: Teff, Logg, [M/H], Gmag, Ra, Dec
    # northern hemisphere threshold at dec > -20
    
    # apply threshold
    threshold = {
        'Gmag':(10,12.1),
        '[M/H]':(-2.0,-0.7),
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
    
    # latex table with name, RA, DEC, Teff, Logg, [M/H], subclass,Gmag
    def generate_latex_table(df: pd.DataFrame, save_path: Path) -> None:
        """
        Generate a LaTeX table with target parameters.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the filtered catalog data
        save_path : Path
            Path to save the LaTeX table file
        """
        # Sort by Gmag (brightest first)
        df_sorted = df.sort_values('Gmag').copy()
        
        # Begin the LaTeX table
        latex_table = r'''\begin{table*}[ht]
\renewcommand{\arraystretch}{1.3}
\centering
'''
        latex_table += "\\caption{"
        latex_table += "LAMOST M dwarf targets selected for observation. "
        latex_table += "The table lists the observation ID, coordinates, "
        latex_table += "effective temperature, surface gravity, metallicity, "
        latex_table += "spectral subclass, and apparent G magnitude."
        latex_table += "}\n"
        latex_table += "\\label{tab:lamost_targets}\n"
        latex_table += "\\begin{tabular}{lccccccc}\\hline\n"
        
        # Header row
        headers = [
            'Name',
            'RA',
            'Dec',
            r'T$_{\mathrm{eff}}$' + ' [K]',
            'log g',
            '[M/H]',
            'Subclass',
            'G' + ' [mag]'
        ]
        latex_table += ' & '.join(headers) + r'\\' + '\n' + r'\hline' + '\n'
        
        # Data rows
        for index, row in df_sorted.iterrows():
            # Name (obsid) - handle both int and float formats
            if pd.notna(row['obsid']):
                try:
                    name = str(int(float(row['obsid'])))
                except (ValueError, TypeError):
                    name = str(row['obsid']).strip()
            else:
                name = '---'
            
            # RA - format as decimal degrees with appropriate precision
            ra = f"{row['RA']:.3f}" if pd.notna(row['RA']) else '---'
            
            # Dec - format as decimal degrees with appropriate precision
            dec = f"{row['Dec']:.3f}" if pd.notna(row['Dec']) else '---'
            
            # Teff - format as integer
            if pd.notna(row['Teff']):
                try:
                    teff = f"{int(float(row['Teff']))}"
                except (ValueError, TypeError):
                    teff = '---'
            else:
                teff = '---'
            
            # Logg - format with 2 decimal places
            logg = f"{row['Logg']:.2f}" if pd.notna(row['Logg']) else '---'
            
            # [M/H] - format with 2 decimal places, include sign
            mh = f"{row['[M/H]']:.2f}" if pd.notna(row['[M/H]']) else '---'
            
            # Subclass
            subclass = str(row['subclass']).strip() if pd.notna(row['subclass']) else '---'
            
            # Gmag - format with 2 decimal places
            gmag = f"{row['Gmag']:.2f}" if pd.notna(row['Gmag']) else '---'
            
            # Build row
            row_data = [name, ra, dec, teff, logg, mh, subclass, gmag]
            latex_table += ' & '.join(row_data) + r'\\' + '\n'
        
        # End the LaTeX table
        latex_table += r'''\hline
\end{tabular}
\end{table*}
'''
        
        # Write to file
        with open(save_path, 'w') as f:
            f.write(latex_table)
        
        print(f'LaTeX table written to {save_path}')
        print(f'Table contains {len(df_sorted)} targets')
    
    # Generate LaTeX table
    table_path = output_dir / f'lamost_targets_table{suffix}.tex'
    generate_latex_table(df, table_path)