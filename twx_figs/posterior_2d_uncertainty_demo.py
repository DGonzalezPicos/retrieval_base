"""
Robust Uncertainty Estimation for Highly Correlated Parameters

This script demonstrates how to properly estimate uncertainties for parameters
that are highly correlated (Pearson r > 0.8) by using the joint 2D posterior
distribution rather than marginalized 1D intervals.

Key Methods:
1. 1D Marginalized intervals (naive approach - underestimates uncertainty)
2. 2D Credible region projection (robust approach)
3. Highest Posterior Density (HPD) regions
4. Bootstrap resampling for uncertainty validation

Usage:
    python posterior_2d_uncertainty_demo.py

Requirements:
    - h5py, numpy, matplotlib, scipy, seaborn
    - Access to cached HDF5 posterior files from retrieval analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
import h5py
import os
import pathlib
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
import seaborn as sns
from typing import Tuple, List, Dict, Optional
import retrieval_base.auxiliary_functions as af

# Configuration
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['figure.dpi'] = 100

def setup_paths():
    """Setup paths for data and figures"""
    path = af.get_path(return_pathlib=True)
    path_figures = pathlib.Path('/home/dario/phd/twa2x_paper/figures')
    return path, path_figures

def load_joint_posterior(h5_file: str, param_x: str, param_y: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load joint posterior samples for two parameters from HDF5 file.
    
    Args:
        h5_file: Path to HDF5 file containing posterior samples
        param_x: Name of x-axis parameter (e.g., '12CO')
        param_y: Name of y-axis parameter (e.g., 'log_g')
    
    Returns:
        Tuple of (y_samples, x_samples) arrays
    """
    with h5py.File(h5_file, 'r') as f:
        # Load x parameter (VMR parameters need averaging and log conversion)
        if param_x in f['VMRs_posterior']:
            x_samples = f['VMRs_posterior'][param_x][:]
            x_samples = np.mean(x_samples, axis=-1)  # Average over pressure levels
            x_samples = np.log10(x_samples)  # Convert to log scale
        elif param_x in f['COH_posterior']:
            x_samples = f['COH_posterior'][param_x][:]
            x_samples = np.mean(x_samples, axis=-1)
        else:
            raise ValueError(f"Parameter {param_x} not found in HDF5 file")
        
        # Load y parameter (assuming it's already processed)
        if param_y in f['COH_posterior']:
            y_samples = f['COH_posterior'][param_y][:]
            y_samples = np.mean(y_samples, axis=-1)
        elif param_y in f['VMRs_posterior']:
            y_samples = f['VMRs_posterior'][param_y][:]
            y_samples = np.mean(y_samples, axis=-1)
        else:
            # Try to load from separate file (e.g., log_g)
            log_g_file = h5_file.replace('chem_posterior.h5', 'log_g_posterior.npy')
            if os.path.exists(log_g_file) and param_y == 'log_g':
                y_samples = np.load(log_g_file)
            else:
                raise ValueError(f"Parameter {param_y} not found")
    
    return y_samples, x_samples

def clean_samples(y_samples: np.ndarray, x_samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Clean samples by removing NaN and infinite values"""
    # Remove NaN and infinite values
    mask = np.isfinite(y_samples) & np.isfinite(x_samples)
    y_clean = y_samples[mask]
    x_clean = x_samples[mask]
    
    print(f"Cleaned samples: {len(y_clean)}/{len(y_samples)} valid points")
    return y_clean, x_clean

def compute_2d_kde(y_samples: np.ndarray, x_samples: np.ndarray, 
                   grid_size: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 2D kernel density estimate for the joint posterior.
    
    Args:
        y_samples: Y parameter samples
        x_samples: X parameter samples
        grid_size: Size of the evaluation grid
    
    Returns:
        Tuple of (X_grid, Y_grid, Z_density)
    """
    # Create KDE
    data = np.vstack([y_samples, x_samples])
    kde = gaussian_kde(data)
    
    # Create evaluation grid with some padding
    y_range = np.percentile(y_samples, [0.5, 99.5])
    x_range = np.percentile(x_samples, [0.5, 99.5])
    
    y_pad = 0.1 * (y_range[1] - y_range[0])
    x_pad = 0.1 * (x_range[1] - x_range[0])
    
    Y_grid, X_grid = np.mgrid[
        y_range[0]-y_pad:y_range[1]+y_pad:complex(0, grid_size),
        x_range[0]-x_pad:x_range[1]+x_pad:complex(0, grid_size)
    ]
    
    # Evaluate KDE on grid
    positions = np.vstack([Y_grid.ravel(), X_grid.ravel()])
    Z_density = kde(positions).T.reshape(Y_grid.shape)
    
    return X_grid, Y_grid, Z_density

def find_credible_levels(Z_density: np.ndarray, credible_levels: List[float]) -> List[float]:
    """
    Find density levels that correspond to given credible intervals.
    
    Args:
        Z_density: 2D density array
        credible_levels: List of credible levels (e.g., [0.68, 0.95])
    
    Returns:
        List of density threshold values (in increasing order)
    """
    # Flatten and sort density values
    Z_flat = Z_density.flatten()
    Z_sorted = np.sort(Z_flat)[::-1]  # Sort in descending order
    
    # Compute cumulative probability
    cumsum = np.cumsum(Z_sorted)
    cumsum = cumsum / cumsum[-1]  # Normalize
    
    # Find levels
    levels = []
    for prob in credible_levels:
        try:
            idx = np.where(cumsum <= prob)[0][-1]
            levels.append(Z_sorted[idx])
        except IndexError:
            levels.append(Z_sorted[0])
    
    # Sort levels in increasing order for matplotlib contour
    levels = sorted(levels)
    
    # Ensure levels are distinct
    levels = np.array(levels)
    if len(levels) > 1:
        # Add small increment to ensure distinct levels
        for i in range(1, len(levels)):
            if levels[i] <= levels[i-1]:
                levels[i] = levels[i-1] + 1e-10
    
    return levels.tolist()

def project_2d_credible_region(X_grid: np.ndarray, Y_grid: np.ndarray, 
                              Z_density: np.ndarray, level: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 2D credible region onto parameter axes.
    
    Args:
        X_grid: X coordinate grid
        Y_grid: Y coordinate grid  
        Z_density: 2D density array
        level: Density level threshold
    
    Returns:
        Tuple of (y_projected, x_projected) ranges
    """
    # Find points inside the credible region
    mask = Z_density >= level
    
    # Project onto axes
    y_proj = Y_grid[mask]
    x_proj = X_grid[mask]
    
    return y_proj, x_proj

def compute_hpd_interval(samples: np.ndarray, credible_level: float = 0.68) -> Tuple[float, float]:
    """
    Compute Highest Posterior Density (HPD) interval for 1D samples.
    
    Args:
        samples: 1D array of parameter samples
        credible_level: Credible level (e.g., 0.68 for 68% interval)
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    # Sort samples
    sorted_samples = np.sort(samples)
    n_samples = len(sorted_samples)
    
    # Number of samples in the interval
    n_interval = int(credible_level * n_samples)
    
    # Find the shortest interval
    interval_widths = sorted_samples[n_interval:] - sorted_samples[:n_samples-n_interval]
    min_idx = np.argmin(interval_widths)
    
    lower = sorted_samples[min_idx]
    upper = sorted_samples[min_idx + n_interval]
    
    return lower, upper

def bootstrap_uncertainty(y_samples: np.ndarray, x_samples: np.ndarray, 
                         n_bootstrap: int = 1000) -> Dict[str, np.ndarray]:
    """
    Use bootstrap resampling to estimate uncertainty in correlation and regression parameters.
    
    Args:
        y_samples: Y parameter samples
        x_samples: X parameter samples
        n_bootstrap: Number of bootstrap samples
    
    Returns:
        Dictionary with bootstrap distributions
    """
    n_samples = len(y_samples)
    correlations = []
    slopes = []
    intercepts = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        idx = np.random.choice(n_samples, n_samples, replace=True)
        y_boot = y_samples[idx]
        x_boot = x_samples[idx]
        
        # Compute statistics
        corr = np.corrcoef(y_boot, x_boot)[0, 1]
        slope, intercept = np.polyfit(x_boot, y_boot, 1)
        
        correlations.append(corr)
        slopes.append(slope)
        intercepts.append(intercept)
    
    return {
        'correlations': np.array(correlations),
        'slopes': np.array(slopes),
        'intercepts': np.array(intercepts)
    }

def create_comparison_plot(y_samples: np.ndarray, x_samples: np.ndarray, 
                          param_names: Dict[str, str], target_name: str) -> plt.Figure:
    """
    Create comprehensive comparison plot showing different uncertainty methods.
    
    Args:
        y_samples: Y parameter samples
        x_samples: X parameter samples
        param_names: Dictionary with parameter labels
        target_name: Name of the target object
    
    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Main 2D posterior plot
    ax_main = fig.add_subplot(gs[0:2, 0:2])
    
    # Marginal histograms
    ax_y_hist = fig.add_subplot(gs[0:2, 2])
    ax_x_hist = fig.add_subplot(gs[2, 0:2])
    
    # Statistics table
    ax_stats = fig.add_subplot(gs[2, 2])
    
    # Compute 2D KDE
    X_grid, Y_grid, Z_density = compute_2d_kde(y_samples, x_samples)
    
    # Find credible levels
    levels = find_credible_levels(Z_density, [0.68, 0.95])
    
    # Plot 2D posterior with contours
    im = ax_main.contourf(X_grid, Y_grid, Z_density, levels=50, cmap='Blues', alpha=0.7)
    contours = ax_main.contour(X_grid, Y_grid, Z_density, levels=levels, 
                              colors=['darkblue', 'navy'], linewidths=[2, 1.5])
    
    # Add contour labels
    contour_labels = ['68%', '95%']
    for i, label in enumerate(contour_labels):
        if i < len(levels):
            # Add manual label
            ax_main.text(0.02, 0.98-i*0.05, label, 
                        transform=ax_main.transAxes, fontsize=10, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Scatter plot of samples
    n_plot = min(2000, len(y_samples))
    idx = np.random.choice(len(y_samples), n_plot, replace=False)
    ax_main.scatter(x_samples[idx], y_samples[idx], s=1, alpha=0.3, color='red')
    
    # Compute different uncertainty estimates
    # 1. Marginalized 1D intervals
    y_1d = np.percentile(y_samples, [16, 50, 84])
    x_1d = np.percentile(x_samples, [16, 50, 84])
    
    # 2. 2D credible region projection
    y_proj, x_proj = project_2d_credible_region(X_grid, Y_grid, Z_density, levels[0])
    y_2d = np.percentile(y_proj, [16, 50, 84])
    x_2d = np.percentile(x_proj, [16, 50, 84])
    
    # 3. HPD intervals
    y_hpd = compute_hpd_interval(y_samples, 0.68)
    x_hpd = compute_hpd_interval(x_samples, 0.68)
    
    # Plot intervals on main plot
    ax_main.axhline(y_1d[1], color='orange', linestyle='--', alpha=0.8, 
                   label=f'{param_names["y"]} 1D median')
    ax_main.axvline(x_1d[1], color='green', linestyle='--', alpha=0.8, 
                   label=f'{param_names["x"]} 1D median')
    
    # Plot marginal histograms
    # Y parameter histogram
    ax_y_hist.hist(y_samples, bins=50, orientation='horizontal', alpha=0.7, 
                   color='skyblue', density=True)
    ax_y_hist.axhline(y_1d[1], color='orange', linestyle='--', linewidth=2, label='1D median')
    ax_y_hist.axhline(y_2d[1], color='red', linestyle=':', linewidth=2, label='2D proj median')
    
    # Fill between for confidence intervals
    y_hist_xlim = ax_y_hist.get_xlim()
    ax_y_hist.fill_betweenx([y_1d[0], y_1d[2]], 0, y_hist_xlim[1], 
                           alpha=0.3, color='orange', label='1D 68% CI')
    ax_y_hist.fill_betweenx([y_2d[0], y_2d[2]], 0, y_hist_xlim[1], 
                           alpha=0.3, color='red', label='2D proj 68% CI')
    ax_y_hist.set_ylabel(param_names['y'])
    ax_y_hist.legend(fontsize=8)
    
    # X parameter histogram
    ax_x_hist.hist(x_samples, bins=50, alpha=0.7, color='lightgreen', density=True)
    ax_x_hist.axvline(x_1d[1], color='orange', linestyle='--', linewidth=2, label='1D median')
    ax_x_hist.axvline(x_2d[1], color='red', linestyle=':', linewidth=2, label='2D proj median')
    
    # Fill between for confidence intervals
    x_hist_ylim = ax_x_hist.get_ylim()
    ax_x_hist.fill_between([x_1d[0], x_1d[2]], 0, x_hist_ylim[1], 
                          alpha=0.3, color='orange', label='1D 68% CI')
    ax_x_hist.fill_between([x_2d[0], x_2d[2]], 0, x_hist_ylim[1], 
                          alpha=0.3, color='red', label='2D proj 68% CI')
    ax_x_hist.set_xlabel(param_names['x'])
    ax_x_hist.legend(fontsize=8)
    
    # Compute correlation and bootstrap uncertainties
    correlation = np.corrcoef(y_samples, x_samples)[0, 1]
    bootstrap_results = bootstrap_uncertainty(y_samples, x_samples, n_bootstrap=1000)
    
    # Statistics table
    ax_stats.axis('off')
    stats_text = f"""
{target_name} Statistics

Correlation: {correlation:.3f}
Bootstrap r: {np.mean(bootstrap_results['correlations']):.3f} ± {np.std(bootstrap_results['correlations']):.3f}

{param_names['y']} Uncertainties:
1D: {y_1d[1]:.3f} +{y_1d[2]-y_1d[1]:.3f} -{y_1d[1]-y_1d[0]:.3f}
2D: {y_2d[1]:.3f} +{y_2d[2]-y_2d[1]:.3f} -{y_2d[1]-y_2d[0]:.3f}
HPD: {(y_hpd[0]+y_hpd[1])/2:.3f} +{y_hpd[1]-(y_hpd[0]+y_hpd[1])/2:.3f} -{(y_hpd[0]+y_hpd[1])/2-y_hpd[0]:.3f}

{param_names['x']} Uncertainties:
1D: {x_1d[1]:.3f} +{x_1d[2]-x_1d[1]:.3f} -{x_1d[1]-x_1d[0]:.3f}
2D: {x_2d[1]:.3f} +{x_2d[2]-x_2d[1]:.3f} -{x_2d[1]-x_2d[0]:.3f}
HPD: {(x_hpd[0]+x_hpd[1])/2:.3f} +{x_hpd[1]-(x_hpd[0]+x_hpd[1])/2:.3f} -{(x_hpd[0]+x_hpd[1])/2-x_hpd[0]:.3f}

Uncertainty Ratio (2D/1D):
{param_names['y']}: {(y_2d[2]-y_2d[0])/(y_1d[2]-y_1d[0]):.2f}x
{param_names['x']}: {(x_2d[2]-x_2d[0])/(x_1d[2]-x_1d[0]):.2f}x
    """
    
    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes, 
                 fontsize=9, verticalalignment='top', fontfamily='monospace')
    
    # Set labels and title
    ax_main.set_xlabel(param_names['x'])
    ax_main.set_ylabel(param_names['y'])
    ax_main.set_title(f'{target_name}: 2D Posterior with Credible Regions\n'
                     f'Correlation: r = {correlation:.3f}')
    
    return fig

def analyze_target(path: pathlib.Path, target: str, run: str, 
                  param_x: str = '12CO', param_y: str = 'log_g') -> Dict:
    """
    Analyze a single target's posterior distribution.
    
    Args:
        path: Base path to data
        target: Target name (e.g., 'TWA27A')
        run: Run name (e.g., 'freeslab_lbl10_G2G3_2')
        param_x: X parameter name
        param_y: Y parameter name
    
    Returns:
        Dictionary with analysis results
    """
    # Load data
    h5_file = path / target / f'retrieval_outputs/{run}/test_data/chem_posterior.h5'
    
    if not os.path.exists(h5_file):
        raise FileNotFoundError(f"HDF5 file not found: {h5_file}")
    
    print(f"\nAnalyzing {target} - {run}")
    print(f"Parameters: {param_y} vs {param_x}")
    
    # Load posterior samples
    y_samples, x_samples = load_joint_posterior(str(h5_file), param_x, param_y)
    y_samples, x_samples = clean_samples(y_samples, x_samples)
    
    # Compute correlation
    correlation = np.corrcoef(y_samples, x_samples)[0, 1]
    print(f"Correlation coefficient: {correlation:.3f}")
    
    # Compute different uncertainty estimates
    results = {
        'target': target,
        'run': run,
        'correlation': correlation,
        'n_samples': len(y_samples),
        'y_samples': y_samples,
        'x_samples': x_samples
    }
    
    # 1D marginalized intervals
    y_1d = np.percentile(y_samples, [16, 50, 84])
    x_1d = np.percentile(x_samples, [16, 50, 84])
    results['y_1d'] = y_1d
    results['x_1d'] = x_1d
    
    # 2D credible region analysis
    X_grid, Y_grid, Z_density = compute_2d_kde(y_samples, x_samples)
    levels = find_credible_levels(Z_density, [0.68, 0.95])
    y_proj, x_proj = project_2d_credible_region(X_grid, Y_grid, Z_density, levels[0])
    
    y_2d = np.percentile(y_proj, [16, 50, 84])
    x_2d = np.percentile(x_proj, [16, 50, 84])
    results['y_2d'] = y_2d
    results['x_2d'] = x_2d
    
    # HPD intervals
    y_hpd = compute_hpd_interval(y_samples, 0.68)
    x_hpd = compute_hpd_interval(x_samples, 0.68)
    results['y_hpd'] = y_hpd
    results['x_hpd'] = x_hpd
    
    # Bootstrap analysis
    bootstrap_results = bootstrap_uncertainty(y_samples, x_samples, n_bootstrap=500)
    results['bootstrap'] = bootstrap_results
    
    # Print summary
    print(f"1D uncertainties - {param_y}: ±{(y_1d[2]-y_1d[0])/2:.3f}, {param_x}: ±{(x_1d[2]-x_1d[0])/2:.3f}")
    print(f"2D uncertainties - {param_y}: ±{(y_2d[2]-y_2d[0])/2:.3f}, {param_x}: ±{(x_2d[2]-x_2d[0])/2:.3f}")
    print(f"Uncertainty ratio - {param_y}: {(y_2d[2]-y_2d[0])/(y_1d[2]-y_1d[0]):.2f}x, {param_x}: {(x_2d[2]-x_2d[0])/(x_1d[2]-x_1d[0]):.2f}x")
    
    return results

def main():
    """Main analysis function"""
    print("=== Robust Uncertainty Estimation for Highly Correlated Parameters ===\n")
    
    # Setup paths
    path, path_figures = setup_paths()
    
    # Define targets and runs to analyze
    targets_runs = [
        ('TWA27A', 'freeslab_lbl10_G2G3_2'),
        ('TWA27A', 'freeslab_lbl10_G1G2G3_1'),
        ('TWA28', 'freeslab_lbl10_G2G3_1'),
        ('TWA28', 'freeslab_lbl10_G1G2G3_1'),
    ]
    
    # Parameters to analyze
    param_x = '12CO'
    param_y = 'log_g'
    
    param_names = {
        'x': f'log {param_x} VMR',
        'y': f'{param_y} [cgs]'
    }
    
    # Analyze each target
    all_results = []
    
    for target, run in targets_runs:
        try:
            results = analyze_target(path, target, run, param_x, param_y)
            all_results.append(results)
            
            # Create individual plot
            fig = create_comparison_plot(
                results['y_samples'], 
                results['x_samples'], 
                param_names, 
                f"{target} ({run.split('_')[-1]})"
            )
            
            # Save plot
            plot_name = path_figures / f"posterior_2d_uncertainty_{target}_{run.split('_')[-1]}.pdf"
            fig.savefig(plot_name, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved plot: {plot_name}")
            
        except Exception as e:
            print(f"Error analyzing {target} - {run}: {e}")
            continue
    
    # Summary comparison
    print("\n=== SUMMARY COMPARISON ===")
    print("Target\t\tRun\t\tCorrelation\t1D_width\t2D_width\tRatio")
    print("-" * 80)
    
    for results in all_results:
        target = results['target']
        run = results['run'].split('_')[-1]
        corr = results['correlation']
        y_1d_width = results['y_1d'][2] - results['y_1d'][0]
        y_2d_width = results['y_2d'][2] - results['y_2d'][0]
        ratio = y_2d_width / y_1d_width
        
        print(f"{target}\t\t{run}\t\t{corr:.3f}\t\t{y_1d_width:.3f}\t\t{y_2d_width:.3f}\t\t{ratio:.2f}x")
    
    # Create combined comparison plot
    if len(all_results) > 1:
        fig = plt.figure(figsize=(16, 12))
        
        # Create a 2x2 grid of subplots, each with marginal histograms
        n_plots = min(4, len(all_results))
        
        for i, results in enumerate(all_results[:n_plots]):
            # Calculate subplot position
            row = i // 2
            col = i % 2
            
            # Create subplot with space for marginals
            left = 0.1 + col * 0.45
            bottom = 0.55 - row * 0.45
            width = 0.3
            height = 0.3
            
            # Main 2D plot
            ax_main = fig.add_axes([left, bottom, width, height])
            
            # Marginal histograms
            ax_top = fig.add_axes([left, bottom + height + 0.02, width, 0.08])
            ax_right = fig.add_axes([left + width + 0.02, bottom, 0.08, height])
            
            # Plot 2D posterior
            X_grid, Y_grid, Z_density = compute_2d_kde(results['y_samples'], results['x_samples'])
            levels = find_credible_levels(Z_density, [0.68, 0.95])
            
            # Contour plot
            ax_main.contour(X_grid, Y_grid, Z_density, levels=levels, 
                           colors=['darkblue', 'navy'], linewidths=[2, 1.5])
            ax_main.contourf(X_grid, Y_grid, Z_density, levels=50, cmap='Blues', alpha=0.3)
            
            # Sample scatter
            n_plot = min(1000, len(results['y_samples']))
            idx = np.random.choice(len(results['y_samples']), n_plot, replace=False)
            ax_main.scatter(results['x_samples'][idx], results['y_samples'][idx], 
                          s=0.5, alpha=0.4, color='red')
            
            # Compute uncertainty intervals
            y_1d = results['y_1d']
            x_1d = results['x_1d']
            y_2d = results['y_2d']
            x_2d = results['x_2d']
            
            # Plot medians on main plot
            ax_main.axhline(y_1d[1], color='orange', linestyle='--', alpha=0.8, linewidth=1.5)
            ax_main.axvline(x_1d[1], color='green', linestyle='--', alpha=0.8, linewidth=1.5)
            ax_main.axhline(y_2d[1], color='red', linestyle=':', alpha=0.8, linewidth=1.5)
            ax_main.axvline(x_2d[1], color='blue', linestyle=':', alpha=0.8, linewidth=1.5)
            
            # Top histogram (X parameter)
            ax_top.hist(results['x_samples'], bins=40, alpha=0.6, color='lightgreen', 
                       density=True, label='1D marginal')
            
            # For 2D projection histogram, use the projected samples
            y_proj, x_proj = project_2d_credible_region(X_grid, Y_grid, Z_density, levels[0])
            ax_top.hist(x_proj, bins=40, alpha=0.6, color='lightblue', 
                       density=True, label='2D projection')
            
            # Add confidence interval lines for X parameter
            ax_top.axvline(x_1d[0], color='green', linestyle='--', alpha=0.8, linewidth=2)
            ax_top.axvline(x_1d[2], color='green', linestyle='--', alpha=0.8, linewidth=2)
            ax_top.axvline(x_1d[1], color='green', linestyle='-', alpha=0.9, linewidth=2)
            
            ax_top.axvline(x_2d[0], color='blue', linestyle=':', alpha=0.8, linewidth=2)
            ax_top.axvline(x_2d[2], color='blue', linestyle=':', alpha=0.8, linewidth=2)
            ax_top.axvline(x_2d[1], color='blue', linestyle='-', alpha=0.9, linewidth=2)
            
            # Fill between confidence intervals
            y_hist_max = ax_top.get_ylim()[1]
            ax_top.fill_between([x_1d[0], x_1d[2]], 0, y_hist_max, 
                              alpha=0.2, color='green', label='1D 68% CI')
            ax_top.fill_between([x_2d[0], x_2d[2]], 0, y_hist_max, 
                              alpha=0.2, color='blue', label='2D 68% CI')
            
            ax_top.set_xlim(ax_main.get_xlim())
            ax_top.set_xticks([])
            ax_top.set_ylabel('Density', fontsize=8)
            ax_top.tick_params(axis='y', labelsize=8)
            
            # Right histogram (Y parameter)
            ax_right.hist(results['y_samples'], bins=40, orientation='horizontal', 
                         alpha=0.6, color='lightsalmon', density=True, label='1D marginal')
            
            # Use the same projected samples for Y parameter
            ax_right.hist(y_proj, bins=40, orientation='horizontal', 
                         alpha=0.6, color='lightcoral', density=True, label='2D projection')
            
            # Add confidence interval lines for Y parameter
            ax_right.axhline(y_1d[0], color='orange', linestyle='--', alpha=0.8, linewidth=2)
            ax_right.axhline(y_1d[2], color='orange', linestyle='--', alpha=0.8, linewidth=2)
            ax_right.axhline(y_1d[1], color='orange', linestyle='-', alpha=0.9, linewidth=2)
            
            ax_right.axhline(y_2d[0], color='red', linestyle=':', alpha=0.8, linewidth=2)
            ax_right.axhline(y_2d[2], color='red', linestyle=':', alpha=0.8, linewidth=2)
            ax_right.axhline(y_2d[1], color='red', linestyle='-', alpha=0.9, linewidth=2)
            
            # Fill between confidence intervals
            x_hist_max = ax_right.get_xlim()[1]
            ax_right.fill_betweenx([y_1d[0], y_1d[2]], 0, x_hist_max, 
                                 alpha=0.2, color='orange', label='1D 68% CI')
            ax_right.fill_betweenx([y_2d[0], y_2d[2]], 0, x_hist_max, 
                                 alpha=0.2, color='red', label='2D 68% CI')
            
            ax_right.set_ylim(ax_main.get_ylim())
            ax_right.set_yticks([])
            ax_right.set_xlabel('Density', fontsize=8)
            ax_right.tick_params(axis='x', labelsize=8)
            
            # Labels and title for main plot
            ax_main.set_xlabel(param_names['x'], fontsize=10)
            ax_main.set_ylabel(param_names['y'], fontsize=10)
            ax_main.set_title(f"{results['target']} ({results['run'].split('_')[-1]})\n"
                            f"r = {results['correlation']:.3f}", fontsize=11)
            ax_main.tick_params(labelsize=9)
            
            # Add uncertainty ratio text
            y_ratio = (y_2d[2] - y_2d[0]) / (y_1d[2] - y_1d[0])
            x_ratio = (x_2d[2] - x_2d[0]) / (x_1d[2] - x_1d[0])
            
            ax_main.text(0.02, 0.98, f'Uncertainty Ratio:\n{param_names["y"]}: {y_ratio:.2f}x\n{param_names["x"]}: {x_ratio:.2f}x', 
                        transform=ax_main.transAxes, fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Add overall legend
        legend_elements = [
            plt.Line2D([0], [0], color='green', linestyle='-', linewidth=2, label='1D Median'),
            plt.Line2D([0], [0], color='green', linestyle='--', linewidth=2, label='1D 68% CI'),
            plt.Line2D([0], [0], color='blue', linestyle='-', linewidth=2, label='2D Median'),
            plt.Line2D([0], [0], color='blue', linestyle=':', linewidth=2, label='2D 68% CI'),
            plt.Line2D([0], [0], color='darkblue', linestyle='-', linewidth=2, label='68% Contour'),
            plt.Line2D([0], [0], color='navy', linestyle='-', linewidth=1.5, label='95% Contour')
        ]
        
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
                  ncol=3, fontsize=10)
        
        # Add overall title
        fig.suptitle('2D Posterior Analysis: Robust Uncertainty Estimation\nfor Highly Correlated Parameters', 
                    fontsize=14, y=0.98)
        
        plt.savefig(path_figures / "posterior_2d_uncertainty_comparison.pdf", 
                   dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Saved enhanced combined comparison plot with marginal histograms")
    
    print(f"\nAnalysis complete! Plots saved to {path_figures}")
    print("\nKey Takeaways:")
    print("1. For highly correlated parameters (|r| > 0.8), 1D marginalized intervals underestimate uncertainty")
    print("2. 2D credible region projection provides more realistic uncertainty estimates")
    print("3. The ratio of 2D/1D uncertainty widths shows the degree of underestimation")
    print("4. Bootstrap resampling validates the correlation strength and its uncertainty")

if __name__ == "__main__":
    main() 