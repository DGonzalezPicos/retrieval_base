"""Generate corner plot for selected parameters of both objects TWA 28 and TWA 27A.

This script:
* Loads posterior distributions from PyMultiNest outputs or HDF5 cache files
* Creates a custom corner plot with 1D histograms on diagonal and 2D contours on lower triangle  
* Overlays both targets on the same figure with different colors
* Shows parameter values and 1-sigma uncertainties in the titles
* Saves a publication-ready figure

Usage:
    python fig_corner.py

The script automatically handles:
- Loading/caching posteriors in HDF5 format for fast subsequent runs
- Computing proper parameter ranges based on both targets
- Robust 2D contour plotting with confidence levels
- Publication-ready formatting and color schemes

Requirements:
- retrieval_base package
- h5py, matplotlib, numpy, scipy
- Access to retrieval output directories for both targets
"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid display issues

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
import os
import pathlib
from retrieval_base.retrieval import Retrieval
import retrieval_base.auxiliary_functions as af
from retrieval_base.config import Config
import h5py
from scipy.stats import gaussian_kde
import corner
import time
from scipy.ndimage import gaussian_filter
import matplotlib.patheffects as pe
# Configuration
FAST_MODE = True  # Set to False for higher accuracy KDE contours (slower)
pe_white = [pe.Stroke(linewidth=1.2, foreground='w'), pe.Normal()]
pe_black = [pe.Stroke(linewidth=1.2, foreground='k'), pe.Normal()]
def setup_paths():
    path = af.get_path(return_pathlib=True)
    path_figures = pathlib.Path('/home/dario/phd/twa2x_paper/figures')
    return path, path_figures

def define_runs_and_colors():
    # only one run per target
    runs = {
        'TWA28': ['freeslab_lbl10_G1G2G3_1'],
        'TWA27A': ['freeslab_lbl10_G1G2G3_1'],
    }
    
    # Using colorblind-friendly palette
    colors = {        
        
        'TWA28': {
            'data': 'k',
            'model': ['#0072B2', 'gold'],  # Dark blue, Orange
            'crires': 'brown',
            'corner': '#D55E00'  # Orange-red for corner plots
        },
        'TWA27A': {
            'data': '#733b27',
            'model': ['#0072B2', 'gold'],  # Pink, Dark blue, Green
            'zhang2025': 'black',
            'corner': '#009E73'  # Bluish green for corner plots
        },
    }
    
    return runs, colors

def define_parameter_labels():
    """Define LaTeX formatted parameter labels"""
    parameter_labels = {
        'R_p': r'$R_p$ [R$_{\rm Jup}$]',
        'T_0': r'$T_0$ [K]',
        'T_d': r'$T_{\rm eff}^{\rm bb}$ [K]',
        'R_d': r'$R^{\rm bb}$ [R$_{\rm Jup}$]',
        'log_Hminus': r'$\log$ H$^-$',
        'log_g': r'$\log g$ [cgs]',
        'rv': r'$v_{\rm rad}$ [km/s]',
        'alpha_12CO': r'$\alpha_{\rm ^{12}CO}$',
        'alpha_H2O': r'$\alpha_{\rm H_2O}$',
        'log_12CO': r'$\log$ $^{12}$CO',
        'log_H2O': r'$\log$ H$_2$O',
        'log_C18O': r'$\log$ C$^{18}$O',
        'dlog_P_1': r'$\Delta\log P_1$',
        'dlog_P_3': r'$\Delta\log P_3$',
        'dlnT_dlnP_RCE': r'$\nabla_{T,\rm RCE}$',
        'log_P_RCE': r'$\log P_{\rm RCE}$',
        'log_12CO/13CO': r'$\log$ $^{12}$CO/$^{13}$CO',
        'log_N_mol': r'$\log N_{\rm mol}^{\rm slab}$',
        'log_T_ex': r'$\log T_{\rm ex}^{\rm slab}$',
        'T_ex': r'$T_{\rm ex}^{\rm slab}$ [K]',
        'log_R_jup': r'$\log R^{\rm slab}$',
        'R_jup': r'$R^{\rm slab}$ [R$_{\rm Jup}$]'
    }
    
    return parameter_labels

def save_posterior_h5(posterior_dict: dict, param_keys: list, filename: str):
    """Save posterior samples to HDF5 file for fast loading"""
    with h5py.File(filename, 'w') as f:
        # Save parameter names
        f.create_dataset('param_keys', data=[k.encode('utf-8') for k in param_keys])
        
        # Save posterior samples for each parameter
        for param, samples in posterior_dict.items():
            f.create_dataset(param, data=samples)
        
        print(f'Saved posterior to {filename}')

def load_posterior_h5(filename: str) -> tuple[dict, list]:
    """Load posterior samples from HDF5 file"""
    posterior_dict = {}
    
    with h5py.File(filename, 'r') as f:
        # Load parameter names
        param_keys = [k.decode('utf-8') for k in f['param_keys'][:]]
        
        # Load posterior samples
        for param in param_keys:
            posterior_dict[param] = f[param][:]
    
    print(f'Loaded posterior from {filename}')
    return posterior_dict, param_keys

def transform_parameters(posterior_dict):
    """Transform log parameters to linear scale where needed"""
    transformed_dict = posterior_dict.copy()
    
    # Convert log_R_d to R_d by taking 10^log_R_d
    if 'log_R_d' in posterior_dict:
        transformed_dict['R_d'] = 10**posterior_dict['log_R_d']
        # Remove the log version to avoid confusion
        if 'log_R_d' in transformed_dict:
            del transformed_dict['log_R_d']
            
    if 'log_R_jup' in posterior_dict:
        transformed_dict['R_jup'] = 10**posterior_dict['log_R_jup']
        # Remove the log version to avoid confusion
        if 'log_R_jup' in transformed_dict:
            del transformed_dict['log_R_jup']
            
    if 'log_T_ex' in posterior_dict:
        transformed_dict['T_ex'] = 10**posterior_dict['log_T_ex']
        # Remove the log version to avoid confusion
        if 'log_T_ex' in transformed_dict:
            del transformed_dict['log_T_ex']
        
            
    return transformed_dict

def load_data(path, target, run, cache=True):
    cwd = os.getcwd()
    if target not in cwd:
        os.chdir(f'{path}/{target}')
        print(f'Changed directory to {target}')
    
    config_file = 'config_jwst.txt'    
    conf = Config(path=path, target=target, run=run)(config_file)
    
    posterior_file = f'{conf.prefix}data/posteriors.h5'
    
    if not cache or not os.path.exists(posterior_file):
        ret = Retrieval(conf=conf, evaluation=False)
        _, posterior = ret.PMN_analyze()
        
        # Create posterior dictionary
        posterior_dict = {}
        for i, param in enumerate(ret.Param.param_keys):
            posterior_dict[param] = posterior[:, i]
        
        # Save to HDF5 file
        save_posterior_h5(posterior_dict, ret.Param.param_keys, posterior_file)
    else:
        # Load from HDF5 file
        posterior_dict, param_keys = load_posterior_h5(posterior_file)
    
    # Transform parameters (e.g., log_R_d to R_d)
    posterior_dict = transform_parameters(posterior_dict)
    
    return posterior_dict

def compute_quantiles(samples: np.ndarray, q: list = [0.16, 0.5, 0.84]) -> np.ndarray:
    """Compute quantiles for parameter samples"""
    return af.quantiles(samples, q=q)

def format_quantile_title(param_name: str, quantiles_dict: dict, param_labels: dict, colors: dict) -> str:
    """Format parameter title with median and 1-sigma errors for both targets"""
    param_label = param_labels.get(param_name, param_name)
    
    # Start with parameter name
    title_parts = [param_label]
    
    # Add values for each target
    for target in ['TWA27A', 'TWA28']:
        if target in quantiles_dict:
            quantiles = quantiles_dict[target]
            median = quantiles[1]
            lower = quantiles[1] - quantiles[0]
            upper = quantiles[2] - quantiles[1]
            
            # Format numbers appropriately
            if abs(median) > 100 or abs(median) < 0.01:
                value_str = f"{median:.2e}$^{{+{upper:.1e}}}_{{-{lower:.1e}}}$"
            else:
                value_str = f"{median:.2f}$^{{+{upper:.2f}}}_{{-{lower:.2f}}}$"
            
            # Simple text without color coding in LaTeX
            title_parts.append(f"{value_str}")
    
    return '\n'.join(title_parts)

def plot_2d_contours_with_scatter(ax, x_samples, y_samples, color, alpha=0.6, bins=25, n_scatter=300):
    """Plot 2D contours with overlaid scatter points using corner package approach"""
    try:
        # First plot the scatter points (underneath contours)
        n_points = min(n_scatter, len(x_samples))
        idx = np.random.choice(len(x_samples), n_points, replace=False)
        ax.scatter(x_samples[idx], y_samples[idx], c=color, alpha=0.4, s=1.5, zorder=1)
        
        # Get axis ranges for proper binning
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        # Create 2D histogram bins (linear scaling like corner package)
        bins_2d = [
            np.linspace(xlim[0], xlim[1], bins + 1),
            np.linspace(ylim[0], ylim[1], bins + 1)
        ]
        
        # Create 2D histogram
        H, X, Y = np.histogram2d(x_samples, y_samples, bins=bins_2d)
        
        if H.sum() == 0:
            return  # No data to plot
        
        # Apply Gaussian smoothing
        smooth = 0.5  # Similar to corner package default
        H = gaussian_filter(H, smooth)
        
        # Compute density levels (from corner package approach)
        levels = [0.68, 0.95]  # 68% and 95% confidence levels
        Hflat = H.flatten()
        inds = np.argsort(Hflat)[::-1]
        Hflat = Hflat[inds]
        sm = np.cumsum(Hflat)
        sm /= sm[-1]
        V = np.empty(len(levels))
        for i, v0 in enumerate(levels):
            try:
                V[i] = Hflat[sm <= v0][-1]
            except IndexError:
                V[i] = Hflat[0]
        V.sort()
        
        # Handle edge case where levels are too close
        m = np.diff(V) == 0
        while np.any(m):
            V[np.where(m)[0][0]] *= 1.0 - 1e-4
            m = np.diff(V) == 0
        V.sort()
        
        # Compute bin centers
        X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])
        
        # Extend array for contours at plot edges (from corner package)
        H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
        H2[2:-2, 2:-2] = H
        H2[2:-2, 1] = H[:, 0]
        H2[2:-2, -2] = H[:, -1]
        H2[1, 2:-2] = H[0]
        H2[-2, 2:-2] = H[-1]
        H2[1, 1] = H[0, 0]
        H2[1, -2] = H[0, -1]
        H2[-2, 1] = H[-1, 0]
        H2[-2, -2] = H[-1, -1]
        
        X2 = np.concatenate([
            X1[0] + np.array([-2, -1]) * np.diff(X1[:2]),
            X1,
            X1[-1] + np.array([1, 2]) * np.diff(X1[-2:]),
        ])
        Y2 = np.concatenate([
            Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]),
            Y1,
            Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:]),
        ])
        
        # Plot filled contours and contour lines
        if len(V) > 0 and np.max(V) > 0:
            ax.contourf(X2, Y2, H2.T, levels=V, colors=[color], alpha=alpha*0.3, 
                       extend='max', zorder=2)
            ax.contour(X2, Y2, H2.T, levels=V, colors=[color], alpha=alpha*0.9, 
                      linewidths=1.2, zorder=3)
                           
    except Exception as e:
        print(f"Warning: Could not plot 2D contours ({e}), using scatter plot fallback")
        # Fallback to scatter plot only
        n_points = min(800, len(x_samples))
        idx = np.random.choice(len(x_samples), n_points, replace=False)
        ax.scatter(x_samples[idx], y_samples[idx], c=color, alpha=0.4, s=2)

def plot_1d_histogram(ax, samples, color, alpha=0.65, bins=30, density=True, 
                     label=None, quantiles=None):
    """Plot 1D histogram with quantile lines"""
    # Plot histogram
    counts, bin_edges, patches = ax.hist(samples, bins=bins, alpha=alpha, 
                                        color=color, density=density,
                                        histtype='stepfilled', edgecolor='black',
                                        linewidth=0.5, label=label)
    
    # Add quantile lines if provided
    if quantiles is not None:
        ymax = ax.get_ylim()[1]
        for i, q_val in enumerate(quantiles):
            linestyle = '-' if i == 1 else '--'  # Solid for median, dashed for bounds
            alpha_line = 0.8 if i == 1 else 0.6
            ax.axvline(q_val, color=color, linestyle=linestyle, 
                      alpha=alpha_line, linewidth=1.5)

def create_custom_corner_plot(data_dict: dict, param_order: list, colors: dict, 
                             path_figures: pathlib.Path):
    """Create custom corner plot with overlaid data from both targets"""
    
    n_params = len(param_order)
    param_labels = define_parameter_labels()
    
    # Create figure
    fig, axes = plt.subplots(n_params, n_params, figsize=(14, 14))
    fig.subplots_adjust(hspace=0.08, wspace=0.08)
    
    # Calculate ranges accounting for both targets with extra margin
    ranges = {}
    for param in param_order:
        all_samples = []
        for target in data_dict.keys():
            if param in data_dict[target]:
                all_samples.extend(data_dict[target][param])
        
        if all_samples:
            all_samples = np.array(all_samples)
            q = af.quantiles(all_samples, q=[0.005, 0.995])  # Use wider range
            margin = 0.15 * (q[1] - q[0])  # Increase margin to ensure no cutoff
            ranges[param] = [q[0] - margin, q[1] + margin]
    
    # Plot each subplot
    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: 1D histograms
                param = param_order[i]
                quantiles_dict = {}
                
                for target in ['TWA28', 'TWA27A']:
                    if target in data_dict and param in data_dict[target]:
                        samples = data_dict[target][param]
                        color = colors[target]['corner']
                        
                        # Compute quantiles
                        quantiles = compute_quantiles(samples)
                        quantiles_dict[target] = quantiles
                        
                        # Plot histogram
                        plot_1d_histogram(ax, samples, color, alpha=0.6, 
                                        label=target, quantiles=quantiles)
                
                # Set title with formatted parameter name and values
                if quantiles_dict:
                    param_label = param_labels.get(param, param)
                    
                    # Set the parameter name as the main title
                    ax.set_title(param_label, fontsize=12, pad=42)
                    
                    # Add colored value text below
                    text_y = 1.55
                    for target in ['TWA27A', 'TWA28']:
                        if target in quantiles_dict:
                            quantiles = quantiles_dict[target]
                            median = quantiles[1]
                            lower = quantiles[1] - quantiles[0]
                            upper = quantiles[2] - quantiles[1]
                            
                            # Format numbers with appropriate decimal precision
                            if abs(median) >= 1000:
                                # Large numbers: use 0 decimal places
                                value_str = f"{median:.0f}$^{{+{upper:.0f}}}_{{-{lower:.0f}}}$"
                            elif abs(median) >= 100:
                                # Medium-large numbers: use 1 decimal place
                                value_str = f"{median:.1f}$^{{+{upper:.1f}}}_{{-{lower:.1f}}}$"
                            elif abs(median) >= 1:
                                # Numbers >= 1: use 2 decimal places
                                value_str = f"{median:.2f}$^{{+{upper:.2f}}}_{{-{lower:.2f}}}$"
                            elif abs(median) >= 0.1:
                                # Numbers 0.1-1: use 3 decimal places
                                value_str = f"{median:.3f}$^{{+{upper:.3f}}}_{{-{lower:.3f}}}$"
                            else:
                                # Small numbers < 0.1: use 4 decimal places
                                value_str = f"{median:.4f}$^{{+{upper:.4f}}}_{{-{lower:.4f}}}$"
                            
                            # Add colored text
                            color = colors[target]['corner']
                            ax.text(0.5, text_y, f"{value_str}", 
                                   transform=ax.transAxes, ha='center', va='top',
                                   color=color, fontsize=12, weight='bold', path_effects=pe_white)
                            text_y -= 0.24
                # Set range and appearance
                if param in ranges:
                    ax.set_xlim(ranges[param])
                ax.set_ylabel('')
                ax.tick_params(axis='y', labelleft=False)
                
                # Remove top and right spines
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
            elif i > j:
                # Lower triangle: 2D contours with scatter
                param_x = param_order[j]
                param_y = param_order[i]
                
                for target in ['TWA28', 'TWA27A']:
                    if (target in data_dict and 
                        param_x in data_dict[target] and 
                        param_y in data_dict[target]):
                        
                        x_samples = data_dict[target][param_x]
                        y_samples = data_dict[target][param_y]
                        color = colors[target]['corner']
                        
                        # Plot 2D contours with scatter overlay
                        plot_2d_contours_with_scatter(ax, x_samples, y_samples, color)
                
                # Set ranges with extra margin to avoid cutoff
                if param_x in ranges:
                    ax.set_xlim(ranges[param_x])
                if param_y in ranges:
                    ax.set_ylim(ranges[param_y])
                    
            else:
                # Upper triangle: hide
                ax.set_visible(False)
    
    # Set axis labels with LaTeX formatting
    for i in range(n_params):
        # Bottom row x-labels
        if i == n_params - 1:
            for j in range(n_params):
                if j <= i:
                    param_label = param_labels.get(param_order[j], param_order[j])
                    axes[i, j].set_xlabel(param_label, fontsize=13)
        else:
            for j in range(n_params):
                if j <= i:
                    axes[i, j].tick_params(axis='x', labelbottom=False)
        
        # Left column y-labels
        for j in range(n_params):
            if j == 0 and i > 0:
                param_label = param_labels.get(param_order[i], param_order[i])
                axes[i, j].set_ylabel(param_label, fontsize=13)
            elif j > 0 or i == 0:
                axes[i, j].tick_params(axis='y', labelleft=False)
    
    # Add legend
    handles = []
    labels = []
    for target in ['TWA28', 'TWA27A'][::-1]:
        if target in data_dict:
            color = colors[target]['corner']
            # Create patch with edge color
            handle = patches.Patch(facecolor=color, linewidth=1.0, edgecolor='black',alpha=0.9)
            handles.append(handle)
            labels.append(target.replace('TWA','TWA '))
    
    if handles:
        legend = fig.legend(handles=handles, labels=labels, loc='upper right', 
                           bbox_to_anchor=(0.68, 0.68), fontsize=16,
                           framealpha=0.9, edgecolor='black', frameon=True)
        
        # Apply bold font and path effects to legend text
        for text in legend.get_texts():
            text.set_fontweight('bold')
            text.set_path_effects(pe_white)
                   
    
    # Save figure
    fig_name = path_figures / 'corner_plot_TWA27A_TWA28.pdf'
    fig.savefig(fig_name, dpi=300, bbox_inches='tight')
    print(f'Saved figure: {fig_name}')
    
    plt.close(fig)  # Close figure to free memory
    return fig

def main():
    start_time = time.time()
    path, path_figures = setup_paths()
    runs, colors = define_runs_and_colors()
    
    # Define the parameters to plot (using available parameters)
    # Common parameters available in both datasets
    param_order = ['R_p', 'T_0', 'log_g', 'log_Hminus', 'alpha_H2O',
                   'log_12CO/13CO',
                   'T_d', 'R_d', 'log_N_mol',
                   'T_ex', 'R_jup']
        
    # Load data for both targets
    data_dict = {}
    
    for target, run_list in runs.items():
        run = run_list[0]  # Use first (and only) run
        print(f'\nLoading data for {target}, run: {run}')
        
        try:
            posterior_dict = load_data(path, target, run, cache=True)
            
            # Filter to only include parameters we want to plot
            filtered_dict = {}
            for param in param_order:
                if param in posterior_dict:
                    filtered_dict[param] = posterior_dict[param]
                else:
                    print(f'Warning: Parameter {param} not found for {target}')
            
            data_dict[target] = filtered_dict
            print(f'Loaded {len(filtered_dict)} parameters for {target}')
            
        except Exception as e:
            print(f'Error loading data for {target}: {e}')
            continue
    
    if not data_dict:
        print('No data loaded successfully. Exiting.')
        return
    
    # Create corner plot
    print('\nCreating corner plot...')
    plot_start_time = time.time()
    fig = create_custom_corner_plot(data_dict, param_order, colors, path_figures)
    plot_time = time.time() - plot_start_time
    
    total_time = time.time() - start_time
    print(f'Plot generation time: {plot_time:.1f}s')
    print(f'Total execution time: {total_time:.1f}s')
    print('Corner plot generation complete!')

if __name__ == '__main__':
    main()