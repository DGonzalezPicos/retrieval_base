"""Script for analyzing and plotting carbon and oxygen isotope ratios as a function of metallicity.

This module generates a figure with three subpanels showing time, carbon and oxygen isotope ratios
as a function of metallicity, including comparison with model values and shaded regions.
"""

import matplotlib.pyplot as plt
from retrieval_base.auxiliary_functions import load_romano_models, axvspan_gradient
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import h5py
from matplotlib.colors import ListedColormap
from matplotlib.colorbar import Colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable

single_column = True
FONTSIZE = 6 if single_column else 7  # Keep within 5-7pt range as required by Nature

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['font.size'] = FONTSIZE

# Ensure RGB color mode for Nature requirements
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# ─── ensure text stays as text ────────────────────────────────────────────────
plt.rcParams['pdf.fonttype']   = 42
plt.rcParams['ps.fonttype']    = 42
plt.rcParams['svg.fonttype']   = 'none'
plt.rcParams['text.usetex']    = False

lw = 1.5

nat_path = '/home/dario/phd/nat/'

@dataclass
class PlotConfig:
    """Configuration for plot styling and parameters."""
    y_labels: Dict[str, str] = None
    y_ticks: Dict[str, List[int]] = None
    colors: List[str] = None
    line_width: float = lw
    z_regions: List[Tuple[float, float]] = None
    
    def __post_init__(self):
        self.y_labels = {
            'time': 'Time (Gyr)',
            'carbon': r'$^{12}$C/$^{13}$C',
            'oxygen': r'$^{16}$O/$^{18}$O'
        }
        self.y_ticks = {
            'carbon': [40, 100, 300, 1000],
            'oxygen': [40, 100, 300, 1000, 4000]
        }
        self.colors = ['purple', 'black']
        self.z_regions = [(-0.40, 0.42), (-0.9, -0.40)]  # Observed and opposite regions

def setup_axis(ax: plt.Axes, y_param: str, config: PlotConfig, xlim: Tuple[float, float] = (-1.2, 0.4)) -> None:
    """Configure a single axis with proper scaling and labels."""
    if y_param in ['carbon', 'oxygen']:
        ax.set_yscale('log')
        
    if y_param == 'oxygen':
        ax.set_xlabel('[M/H]', fontsize=FONTSIZE)
    ax.set_ylabel(config.y_labels[y_param], fontsize=FONTSIZE)
    
    if y_param in ['carbon', 'oxygen']:
        y_ticks = config.y_ticks[y_param]
        y_lims = (min(y_ticks), max(y_ticks))
        ax.set_ylim(y_lims)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([str(t) for t in y_ticks])
    # ax.grid(True)
    if y_param == 'time':
        ax.set_ylim(0, 14)
        ax.set_yticks([0, 3, 6, 9, 12, 13.7])
        
    # Set x-axis limits and ticks
    ax.set_xlim(xlim)
    ax.set_xticks(np.arange(xlim[0], xlim[1]+0.1, 0.2))
    
    # set global fontsize
    ax.tick_params(labelsize=FONTSIZE)
    ax.xaxis.label.set_size(FONTSIZE)
    ax.yaxis.label.set_size(FONTSIZE)
    

def add_shaded_regions(ax: plt.Axes, z_regions: List[Tuple[float, float]], 
                      y_range: Tuple[float, float], colors: Dict[str, np.ndarray]) -> None:
    """Add shaded gradient regions to the plot."""
    for i, z_range in enumerate(z_regions):
        axvspan_gradient(ax, x_range=z_range, y_range=y_range,
                        rgb_color=colors['steelblue'] if i == 0 else colors['darkgold'],
                        gamma=3, reverse=True)

def plot_romano_models(ax: plt.Axes, mass_key: str, y_param: str, 
                      color: str, lw: float) -> None:
    """Plot Romano models for a given mass range and parameter."""
    Z, c12c13, o16o18, time = load_romano_models(
        Z_min=-1.2, mass_range=mass_key, return_time=True
    )
    
    legend_label = None
    if y_param == 'carbon':
        y_data = c12c13
        legend_labels = {'1_8': r'1 $M_\odot$', '3_8': r'3 $M_\odot$'}
        legend_label = legend_labels[mass_key]
    elif y_param == 'oxygen':
        y_data = o16o18
    else:  # time
        y_data = time[::-1]
    ax.plot(Z, y_data, color=color, lw=lw, label=legend_label)
    
def find_time_at_z(time: np.ndarray, Z: np.ndarray, z: float) -> float:
    """Find the time at a given metallicity."""
    return time[np.argmin(np.abs(Z - z))]

def plot_lamost_kde(ax_lamost: plt.Axes, data_path: Path, lamost_color: str = 'brown') -> plt.Axes:
    """Plot LAMOST survey data with distance-based envelopes."""
    
    # Create a new axis above the main plot for LAMOST data
    # divider = make_axes_locatable(ax)
    # ax_lamost = divider.append_axes("top", size="40%", pad=0.1)
    
    # Create inset axis in the upper right corner
    lamost_yscale = 'log'

    add_inset = (not lamost_yscale == 'log')
    if add_inset:
        ax_inset = ax_lamost.inset_axes([0.12, 0.35, 0.3, 0.65])
        
        # Configure inset axis
        x1, x2, y1, y2 = -1.0, -0.7, 0, 200
        ax_inset.set(xlim=(x1, x2), ylim=(y1, y2))
        inset_x_ticks = [-1.0, -0.9, -0.8, -0.7]
        ax_inset.set_xticks(inset_x_ticks)
        ax_inset.set_xticklabels([f'{x:.1f}' for x in inset_x_ticks])
    
        # Show inset zoom with minimal connectors
        rect = (x1, y1, x2-x1, y2-y1)
        box = ax_lamost.indicate_inset_zoom(ax_inset, edgecolor='black', lw=0.1, zorder=-1)
        box[1][3].set_visible(False)
        box[1][1].set_visible(False)
        
        for box_id in [0, 2]:
            box[1][box_id].set_visible(True)
            box[1][box_id].set_linewidth(0.5)
            box[1][box_id].set_zorder(-10)
            
        # Configure inset axis spines and ticks
        ax_inset.spines['top'].set_visible(False)
        ax_inset.spines['right'].set_visible(False)
    
    
    # Load data
    with h5py.File(data_path, 'r') as h5_file:
        kde_scaled_values = h5_file['kde_scaled_values'][:]
        x_kde = h5_file['x_kde'][:]
        metallicity_bins = h5_file['metallicity_bins'][:]
        distance_range = h5_file['distance_range'][:]
    
    # Plot KDE
    alphas = np.linspace(0.1, 1.0, len(distance_range)) * 0.3
    for d, distance in enumerate(distance_range):
        # if distance < 99 or distance > 1000:
            # continue
            
        ax_lamost_axes = [ax_lamost]
        if add_inset:
            ax_lamost_axes.append(ax_inset)

        for axii, axi in enumerate(ax_lamost_axes):
            edgecolor = 'w'
            axi.fill_between(x_kde, kde_scaled_values[d], color=lamost_color, 
                           alpha=alphas[d],
                           lw=2 if distance == 100 else 0.0,
                        #    lw=1.5, ls='-',
                           edgecolor=edgecolor,
                           )
            
            # axi.plot(x_kde, kde_scaled_values[d], color=edgecolor, lw=1.5, ls='-', alpha=0.4,
            #         )
            
            if axii == 0:
                x_max = -0.04
                y_factor = 0.30 if distance == 100 else 1.1
                if int(distance) in [100]:
                    axi.text(s=f'{distance} pc', x=x_max, 
                            y=np.max(kde_scaled_values[d])*y_factor,
                            ha='center', va='center', fontsize=FONTSIZE,
                            color='white' if distance == 100 else 'k', 
                            alpha=0.5,
                            zorder=10,
                            weight='bold',
                            bbox=dict(facecolor='none', alpha=0.5, edgecolor='none'))
    
    # Configure main LAMOST axis
    
    
    if lamost_yscale == 'log':
        ax_lamost.set_yscale('log')
        ax_lamost.set_ylim(1e0, 1e5)
    else:
        ax_lamost.set(ylim=(0, 30e3))

        yticks = np.array([0, 10e3, 20e3, 30e3], dtype=int)
        ax_lamost.set_yticks(yticks)
        ax_lamost.set_yticklabels([f'{y:.0f}' for y in yticks])
    
    # Remove unnecessary spines and ticks
    ax_lamost.spines['top'].set_visible(False)
    ax_lamost.spines['right'].set_visible(False)
    # ax_lamost.spines['left'].set_visible(False)
    # ax_lamost.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    
  
    ax_lamost.set_ylabel('Number of stars', fontsize=FONTSIZE)
    # ax_inset.spines['left'].set_visible(False)
    # ax_inset.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    
    # Share x-axis with main plot
    # ax_lamost.set_xlim(ax_lamost.get_xlim())
    # ax_lamost.set_xticklabels([])
    
    # Ensure tick labels follow journal requirements
    for axis in [ax_lamost.xaxis, ax_lamost.yaxis]:
        axis.set_tick_params(labelsize=FONTSIZE)
    
    return ax_lamost

def plot_lamost_survey(
    ax: plt.Axes,
    data_path: Path,
    lamost_color: str = '#CC2B41'
) -> plt.Axes:
    """Plot LAMOST survey data with distance-based envelopes.
    
    Args:
        ax: The matplotlib axis to plot on
        data_path: Path to the HDF5 file containing LAMOST data
        lamost_color: Color for the LAMOST envelopes
    
    Returns:
        Twin axis with the LAMOST plot
        
    Raises:
        FileNotFoundError: If the HDF5 file doesn't exist
        KeyError: If required datasets are missing from HDF5 file
    """
    try:
        with h5py.File(data_path, 'r') as h5_file:
            numbers = h5_file['numbers'][:]
            distance_range = h5_file['distance_range'][:]
            metallicity_bins = h5_file['metallicity_bins'][:]
    except FileNotFoundError:
        raise FileNotFoundError(f"LAMOST data file not found: {data_path}")
    except KeyError as e:
        raise KeyError(f"Required dataset missing from HDF5 file: {e}")

    # Create twin axis for LAMOST data
    ax_lamost = ax.twinx()
    
    # Find index for distance centre
    distance_centre = 1000
    idx_distance_centre = np.argmin(np.abs(distance_range - distance_centre))
    
    # Plot the 300 pc envelope with solid line
    ax_lamost.plot(
        metallicity_bins[:-1],
        numbers[idx_distance_centre,:],
        color='k',
        lw=2.0,
        label=f'{distance_centre} pc',
        linestyle='-'
    )
    
    # Plot additional envelopes in steps of 100 pc with decreasing alpha
    # alphas = np.linspace(0.6, 0.1, len(distance_range[distance_range > distance_centre]))
    # for i, d in enumerate(distance_range[distance_range != distance_centre]):
    #     idx = np.where(distance_range == d)[0][0]
    #     alpha = 0.1 if d > distance_centre else 0.25
    #     ax_lamost.fill_between(
    #         metallicity_bins[:-1],
    #         numbers[idx,:],
    #         numbers[idx_distance_centre,:],
    #         color=lamost_color,
    #         alpha=alpha,
    #         lw=1
    #     )
    
    # Configure axis
    ax_lamost.set_yscale('log')
    ax_lamost.set_ylim(1, None)
    ax_lamost.set_ylabel('Number of stars', fontsize=FONTSIZE)
    ax_lamost.tick_params(labelsize=FONTSIZE)
    
    
    # Add legend
    ax_lamost.legend(loc='upper right', fontsize=FONTSIZE, 
                    title='LAMOST survey\ndistance ranges',
                    title_fontsize=FONTSIZE)
    
    return ax_lamost

def main() -> None:
    """Main function to create and save the plot."""
    # Initialize configuration
    config = PlotConfig()
    
    # Setup colors for shaded regions
    colors = {
        'light_blue': np.array([50, 164, 168]) / 255,
        'orange_red': np.array([255, 127, 0]) / 255,
        'steelblue': np.array([70, 130, 140]) / 255,
        'lightbrown': np.array([139, 69, 19]) / 255,
        'darkgold': np.array([230, 150, 10]) / 255,
    }
    
    # Create figure and axes
    width_mm = 88.0 if single_column else 180.0
    aspect_ratio = 4/8
    height_mm = width_mm/aspect_ratio
    mm_to_inch = 0.0393701
    fig, axes_all = plt.subplots(4, 1, figsize=(width_mm*mm_to_inch, height_mm*mm_to_inch),
                                 gridspec_kw={'hspace': 0.3,
                                              'height_ratios': [0.6, 1, 1, 1]})
    ax_lamost = axes_all[0]
    axes = axes_all[1:]
    # Plot parameters
    y_params = ['time', 'carbon', 'oxygen']
    mass_keys = ['1_8', '3_8']
    xlim = (-1.2, 0.4)
    # Setup each axis and plot data
    for i, y_param in enumerate(y_params):
        setup_axis(axes[i], y_param, config, xlim=xlim)
        
        if y_param in ['oxygen']:
            y_range = (-300, 1600)
            add_shaded_regions(axes[i], config.z_regions, y_range, colors)
            
        if y_param == 'carbon':
            y_range = (-200, 840)
            add_shaded_regions(axes[i], [config.z_regions[0]], y_range, colors)
        
        if y_param == 'time':
            y_range = (-2.0, 10.0)
            add_shaded_regions(axes[i], [config.z_regions[0]], y_range, colors)
        
        # Plot Romano models
        for j, mass_key in enumerate(mass_keys):
            plot_romano_models(axes[i], mass_key, y_param, config.colors[j], lw=lw)

    Z, c12c13, o16o18, time = load_romano_models(
        Z_min=-1.2, mass_range=mass_key, return_time=True
    )
    time_at_z = find_time_at_z(time[::-1], Z, config.z_regions[0][0])
    axes[0].scatter(config.z_regions[0][0], time_at_z, color='black', marker='o', s=20)
    axes[0].text(config.z_regions[0][0]+0.1, time_at_z+0.5, f'{time_at_z:.1f} Gyr', ha='center', va='bottom', fontsize=FONTSIZE)
    
    # Add LAMOST survey data to first plot
    # lamost_data_path = Path('/home/dario/phd/retrieval_base/paper/figures/metal_poor_cumulative_distance.h5')
    # ax_lamost = plot_lamost_survey(axes[0], lamost_data_path)
    lamost_data_path = Path('/home/dario/phd/retrieval_base/paper/figures/metallicity_density_distance_histogram.h5')
    # lamost_color = 'darkolivegreen'
    # lamost_color = np.array([150, 160, 130]) / 255
    lamost_color = 'peru'
    ax_lamost = plot_lamost_kde(ax_lamost, lamost_data_path, lamost_color)
    ax_lamost.set_xlim(xlim)
    ax_lamost.set_xticks(np.arange(xlim[0], xlim[1]+0.1, 0.2))
    ax_lamost.text(
        s='LAMOST DR10',
        x=0.60, y=0.14,
        ha='left', va='top', fontsize=FONTSIZE,
        transform=ax_lamost.transAxes,
    )

    axes[1].legend(loc='lower left', fontsize=FONTSIZE, title='Nova progenitor\nminimum mass',
                   edgecolor='black', facecolor='white', handlelength=1.5, handletextpad=0.5,
                   title_fontsize=FONTSIZE)
    
    # add custom text in last plot
    s_opp = 'M dwarf\nopportunity'
    s_obs = 'Observed sample'
    s_oxygen_enrichment = r'$^{18}$O' + ' enrichment\nmassive fast-rotating stars'
    axes[2].text(0.34, 0.16, s_opp, ha='center', va='center', fontsize=FONTSIZE, transform=axes[2].transAxes)
    axes[2].text(0.75, 0.12, s_obs, ha='center', va='center', fontsize=FONTSIZE, transform=axes[2].transAxes)
    axes[2].text(0.02, 0.84, s_oxygen_enrichment, ha='left', va='center', fontsize=FONTSIZE, transform=axes[2].transAxes)
    
    # add custom arrow next to text
    axes[2].annotate('', xy=(0.01, 0.5), xytext=(0.25, 0.702), xycoords='axes fraction',
                     
                     arrowprops=dict(arrowstyle='->', color='black', lw=1.5, 
                                     connectionstyle='arc3,rad=-0.08', 
                                     shrinkA=5, shrinkB=5))
    
    # add age of the sun as scatter point 
    axes[0].scatter(0.0, 4.567, color='gold', marker='*', s=100, edgecolor='black', lw=1.0)
    axes[0].text(0.0, 4.567-3, 'Sun\n(4.6 Gyr)', ha='center', va='bottom', fontsize=FONTSIZE)
    
    # also add Barnard's star as scatter point
    gl699 = {'metallicity': -0.37, 'metallicity_err': 0.10,
             'age': 8.5, 'age_err': 1.5}
    axes[0].errorbar(gl699['metallicity'], gl699['age'], xerr=gl699['metallicity_err'], yerr=gl699['age_err'],
                     color='brown', marker='o', ms=4, lw=1.0, capsize=2, capthick=1.0,
    )
    axes[0].annotate(f'Barnard\'s star\n(7-10 Gyr)', xy=(gl699['metallicity'], gl699['age']),
                     xytext=(gl699['metallicity']-0.25, gl699['age']+0.50),
                     ha='center', va='bottom', fontsize=FONTSIZE,
                     arrowprops=dict(arrowstyle='->', color='k', lw=1.0,
                                     connectionstyle='arc3,rad=0.5', 
                                     shrinkA=5, shrinkB=5))
    panel_labels = ['a', 'b', 'c', 'd']
    
    for ax, label in zip(axes_all, panel_labels):
        ax.text(-0.14, 1.06, label, ha='center', va='center', fontsize=FONTSIZE*1.2, 
                transform=ax.transAxes, fontweight='bold')
    
    # Save figure
    save_png = False
    if save_png:
        output_dir = Path('/home/dario/phd/nat/figures/png')
        output_dir.mkdir(parents=True, exist_ok=True)
        fig_path = output_dir / 'nat_fig_4_revised.png'
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f'Saved {fig_path}')

    save_pdf = True
    if save_pdf:
        fig_path_pdf = nat_path + 'nat_fig_4_revised.pdf'
        # Save in RGB mode with editable text as required by Nature
        fig.savefig(fig_path_pdf, dpi=300, bbox_inches='tight', format='pdf',
                    facecolor='white', edgecolor='none', 
                    metadata={'Creator': 'Dario Gonzalez Picos', 
                              'Producer': 'matplotlib'})
        print(f'Saved {fig_path_pdf}')
    
    plt.close(fig)

if __name__ == '__main__':
    # increase global plt parameter for width of axes
    main()