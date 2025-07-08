#!/usr/bin/env python3
"""
Correlation Comparison Demo: Strong vs Weak Parameter Correlations

This script demonstrates the effect of parameter correlation on uncertainty estimation
by comparing two cases for TWA28 G2G3:
1. log_g vs 12CO VMR (strong correlation, r ~ 0.91)
2. log_g vs 12C/13C ratio (weak correlation, r ~ 0.1-0.3)

The visualization shows how uncertainty underestimation depends on correlation strength.

Usage:
    python correlation_comparison_demo.py

Requirements:
    - h5py, numpy, matplotlib, scipy
    - Access to TWA28 G2G3 posterior data
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
import h5py
import os
import pathlib
from scipy.stats import gaussian_kde
import seaborn as sns
from typing import Tuple, List, Dict, Optional
import retrieval_base.auxiliary_functions as af

# Import functions from the main uncertainty demo script
import sys
sys.path.append('/home/dario/phd/retrieval_base/twx_figs')
from posterior_2d_uncertainty_demo import (
    load_joint_posterior, clean_samples, compute_2d_kde, 
    find_credible_levels, project_2d_credible_region, 
    compute_hpd_interval, bootstrap_uncertainty
)

# Configuration
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['figure.dpi'] = 100

def compute_covariance_ellipse(x_samples: np.ndarray, y_samples: np.ndarray, 
                              confidence_level: float = 0.68) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    Compute confidence ellipse parameters using covariance matrix eigendecomposition.
    
    This method finds the principal axes of the posterior distribution and computes
    the semi-major and semi-minor axes that contain the specified confidence level.
    
    Args:
        x_samples: X parameter samples
        y_samples: Y parameter samples  
        confidence_level: Confidence level (e.g., 0.68 for 68%)
    
    Returns:
        Tuple of (ellipse_x, ellipse_y, semi_major, semi_minor, angle)
    """
    from scipy.stats import chi2
    
    # Compute sample means
    x_mean = np.mean(x_samples)
    y_mean = np.mean(y_samples)
    
    # Compute covariance matrix
    cov = np.cov(x_samples, y_samples)
    
    # Eigendecomposition to find principal axes
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    
    # Chi-squared value for the confidence level (2 DOF)
    chi2_val = chi2.ppf(confidence_level, df=2)
    
    # Semi-major and semi-minor axes
    semi_major = np.sqrt(eigenvals[0] * chi2_val)
    semi_minor = np.sqrt(eigenvals[1] * chi2_val)
    
    # Rotation angle of the ellipse
    angle = np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0])
    
    # Generate ellipse points
    theta = np.linspace(0, 2*np.pi, 100)
    ellipse_x = semi_major * np.cos(theta)
    ellipse_y = semi_minor * np.sin(theta)
    
    # Rotate ellipse
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    ellipse_x_rot = ellipse_x * cos_angle - ellipse_y * sin_angle + x_mean
    ellipse_y_rot = ellipse_x * sin_angle + ellipse_y * cos_angle + y_mean
    
    return ellipse_x_rot, ellipse_y_rot, semi_major, semi_minor, angle

def compute_ellipse_projected_intervals(x_samples: np.ndarray, y_samples: np.ndarray, 
                                       confidence_level: float = 0.68) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute confidence intervals by projecting the confidence ellipse onto parameter axes.
    
    This method uses the geometry of the confidence ellipse to determine the 
    maximum extent along each parameter axis, accounting for correlation structure.
    
    Args:
        x_samples: X parameter samples
        y_samples: Y parameter samples
        confidence_level: Confidence level (e.g., 0.68 for 68%)
    
    Returns:
        Tuple of (y_intervals, x_intervals) where each is [lower, median, upper]
    """
    from scipy.stats import chi2
    
    # Compute sample medians
    x_median = np.median(x_samples)
    y_median = np.median(y_samples)
    
    # Compute covariance matrix
    cov = np.cov(x_samples, y_samples)
    
    # Eigendecomposition
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    
    # Chi-squared value for the confidence level (2 DOF)
    chi2_val = chi2.ppf(confidence_level, df=2)
    
    # Semi-major and semi-minor axes
    semi_major = np.sqrt(eigenvals[0] * chi2_val)
    semi_minor = np.sqrt(eigenvals[1] * chi2_val)
    
    # Principal axis directions
    major_axis = eigenvecs[:, 0]  # Direction of maximum variance
    minor_axis = eigenvecs[:, 1]  # Direction of minimum variance
    
    # Project ellipse onto parameter axes
    # For X-axis projection: find maximum extent in X direction
    # This occurs at the points where the ellipse tangent is vertical
    # For Y-axis projection: find maximum extent in Y direction  
    # This occurs at the points where the ellipse tangent is horizontal
    
    # X-axis projection: maximum |x - x_median|
    # Solve for points where dy/dx = 0 (vertical tangent)
    angle = np.arctan2(major_axis[1], major_axis[0])
    
    # Maximum X extent from ellipse geometry
    x_extent = np.sqrt(semi_major**2 * np.cos(angle)**2 + semi_minor**2 * np.sin(angle)**2)
    
    # Maximum Y extent from ellipse geometry  
    y_extent = np.sqrt(semi_major**2 * np.sin(angle)**2 + semi_minor**2 * np.cos(angle)**2)
    
    # Construct intervals
    x_intervals = np.array([x_median - x_extent, x_median, x_median + x_extent])
    y_intervals = np.array([y_median - y_extent, y_median, y_median + y_extent])
    
    return y_intervals, x_intervals

def compute_robust_ellipse_intervals(x_samples: np.ndarray, y_samples: np.ndarray,
                                   confidence_level: float = 0.68) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute robust confidence intervals using Minimum Covariance Determinant (MCD) estimator.
    
    This method is more robust to outliers and non-Gaussian distributions by using
    a robust covariance estimator before computing the confidence ellipse.
    
    Args:
        x_samples: X parameter samples
        y_samples: Y parameter samples
        confidence_level: Confidence level (e.g., 0.68 for 68%)
    
    Returns:
        Tuple of (y_intervals, x_intervals) where each is [lower, median, upper]
    """
    from sklearn.covariance import MinCovDet
    from scipy.stats import chi2
    
    # Prepare data
    data = np.column_stack([x_samples, y_samples])
    
    # Robust covariance estimation using MCD
    try:
        mcd = MinCovDet(random_state=42)
        mcd.fit(data)
        
        robust_mean = mcd.location_
        robust_cov = mcd.covariance_
        
    except ImportError:
        # Fallback to regular covariance if sklearn not available
        print("Warning: sklearn not available, using regular covariance")
        robust_mean = np.array([np.mean(x_samples), np.mean(y_samples)])
        robust_cov = np.cov(x_samples, y_samples)
    
    # Eigendecomposition of robust covariance
    eigenvals, eigenvecs = np.linalg.eigh(robust_cov)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    
    # Chi-squared value for the confidence level (2 DOF)
    chi2_val = chi2.ppf(confidence_level, df=2)
    
    # Semi-major and semi-minor axes
    semi_major = np.sqrt(eigenvals[0] * chi2_val)
    semi_minor = np.sqrt(eigenvals[1] * chi2_val)
    
    # Principal axis angle
    angle = np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0])
    
    # Maximum extents along parameter axes
    x_extent = np.sqrt(semi_major**2 * np.cos(angle)**2 + semi_minor**2 * np.sin(angle)**2)
    y_extent = np.sqrt(semi_major**2 * np.sin(angle)**2 + semi_minor**2 * np.cos(angle)**2)
    
    # Use robust medians
    x_median = np.median(x_samples)
    y_median = np.median(y_samples)
    
    # Construct intervals
    x_intervals = np.array([x_median - x_extent, x_median, x_median + x_extent])
    y_intervals = np.array([y_median - y_extent, y_median, y_median + y_extent])
    
    return y_intervals, x_intervals

def setup_paths():
    """Setup paths for data and figures"""
    path = af.get_path(return_pathlib=True)
    path_figures = pathlib.Path('/home/dario/phd/twa2x_paper/figures')
    return path, path_figures

def load_isotope_ratio(h5_file: str) -> np.ndarray:
    """
    Load 12C/13C isotope ratio from HDF5 file.
    
    Args:
        h5_file: Path to HDF5 file containing posterior samples
    
    Returns:
        Array of 12C/13C ratio samples
    """
    with h5py.File(h5_file, 'r') as f:
        # Load 12C/13C ratio from COH_posterior
        if '12C/13C' in f['COH_posterior']:
            ratio_samples = f['COH_posterior']['12C/13C'][:]
            # Average over pressure levels if needed
            if ratio_samples.ndim > 1:
                ratio_samples = np.mean(ratio_samples, axis=-1)
        else:
            raise ValueError("12C/13C ratio not found in HDF5 file")
    
    return ratio_samples

def load_parameter_pair(h5_file: str, param_x: str, param_y: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a parameter pair with special handling for different parameter types.
    
    Args:
        h5_file: Path to HDF5 file
        param_x: X parameter name
        param_y: Y parameter name (assumed to be log_g)
    
    Returns:
        Tuple of (y_samples, x_samples)
    """
    with h5py.File(h5_file, 'r') as f:
        # Load Y parameter (log_g)
        log_g_file = h5_file.replace('chem_posterior.h5', 'log_g_posterior.npy')
        if os.path.exists(log_g_file) and param_y == 'log_g':
            y_samples = np.load(log_g_file)
        else:
            raise ValueError(f"Parameter {param_y} not found")
        
        # Load X parameter
        if param_x in ['12CO', 'H2O', '12C/13C']:
            # VMR parameter - needs log conversion
            if param_x == '12C/13C':
                x_samples = f['VMRs_posterior']['12CO'][:] / f['VMRs_posterior']['13CO'][:]
            else:
                x_samples = f['VMRs_posterior'][param_x][:]
            x_samples = np.mean(x_samples, axis=-1)  # Average over pressure levels
            x_samples = np.log10(x_samples)  # Convert to log scale
        else:
            raise ValueError(f"Parameter {param_x} not supported")
    
    return y_samples, x_samples

def create_comparison_subplot(ax_main, ax_top, ax_right, y_samples, x_samples, 
                            param_names, title, correlation_type):
    """
    Create a single comparison subplot with marginal histograms.
    
    Args:
        ax_main: Main 2D plot axes
        ax_top: Top histogram axes
        ax_right: Right histogram axes
        y_samples: Y parameter samples
        x_samples: X parameter samples
        param_names: Dictionary with parameter labels
        title: Plot title
        correlation_type: 'strong' or 'weak' for color coding
    """
    # Compute 2D KDE
    X_grid, Y_grid, Z_density = compute_2d_kde(y_samples, x_samples)
    
    # Find credible levels
    levels = find_credible_levels(Z_density, [0.68, 0.95])
    
    # Color scheme based on correlation type
    if correlation_type == 'strong':
        contour_colors = ['darkred', 'red']
        scatter_color = 'red'
        hist_color_1d = 'lightcoral'
        hist_color_2d = 'mistyrose'
        line_color_1d = 'darkred'
        line_color_2d = 'red'
    else:  # weak correlation
        contour_colors = ['darkgreen', 'green']
        scatter_color = 'green'
        hist_color_1d = 'lightgreen'
        hist_color_2d = 'palegreen'
        line_color_1d = 'darkgreen'
        line_color_2d = 'green'
    
    # Plot 2D posterior with contours
    ax_main.contourf(X_grid, Y_grid, Z_density, levels=20, 
                    cmap='Reds' if correlation_type == 'strong' else 'Greens', 
                    alpha=0.4)
    contours = ax_main.contour(X_grid, Y_grid, Z_density, levels=levels, 
                              colors=contour_colors, linewidths=[2.5, 1.5])
    
    # Add confidence ellipse overlay
    ellipse_x, ellipse_y, semi_major, semi_minor, angle = compute_covariance_ellipse(
        x_samples, y_samples, 0.68)
    ax_main.plot(ellipse_x, ellipse_y, 
                color='black' if correlation_type == 'strong' else 'darkblue', 
                linewidth=2.5, linestyle='--', alpha=0.8, 
                label='68% Confidence Ellipse')
    
    # Add contour labels
    contour_labels = ['68%', '95%']
    for i, label in enumerate(contour_labels):
        if i < len(levels):
            ax_main.text(0.02, 0.98-i*0.05, label, 
                        transform=ax_main.transAxes, fontsize=10, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Scatter plot of samples
    n_plot = min(2000, len(y_samples))
    idx = np.random.choice(len(y_samples), n_plot, replace=False)
    ax_main.scatter(x_samples[idx], y_samples[idx], s=1, alpha=0.4, color=scatter_color)
    
    # Compute different uncertainty estimates
    # 1. Marginalized 1D intervals
    y_1d = np.percentile(y_samples, [16, 50, 84])
    x_1d = np.percentile(x_samples, [16, 50, 84])
    
    # 2. Ellipse geometry method (principal axes)
    y_ellipse, x_ellipse = compute_ellipse_projected_intervals(x_samples, y_samples, 0.68)
    
    # 3. KDE-based 2D credible region projection (for comparison)
    y_proj, x_proj = project_2d_credible_region(X_grid, Y_grid, Z_density, levels[0])
    y_2d_kde = np.percentile(y_proj, [16, 50, 84])
    x_2d_kde = np.percentile(x_proj, [16, 50, 84])
    
    # Use ellipse method as primary 2D approach
    y_2d = y_ellipse
    x_2d = x_ellipse
    
    # Plot medians on main plot
    ax_main.axhline(y_1d[1], color=line_color_1d, linestyle='--', alpha=0.8, linewidth=2)
    ax_main.axvline(x_1d[1], color=line_color_1d, linestyle='--', alpha=0.8, linewidth=2)
    ax_main.axhline(y_2d[1], color=line_color_2d, linestyle=':', alpha=0.8, linewidth=2)
    ax_main.axvline(x_2d[1], color=line_color_2d, linestyle=':', alpha=0.8, linewidth=2)
    
    # Top histogram (X parameter)
    bins = 20
    ax_top.hist(x_samples, bins=bins, alpha=0.7, color=hist_color_1d, 
               density=True, label='1D marginal')
    ax_top.hist(x_proj, bins=bins, alpha=0.7, color=hist_color_2d, 
               density=True, label='2D projection')
    
    # Add confidence interval lines for X parameter
    ax_top.axvline(x_1d[0], color=line_color_1d, linestyle='--', alpha=0.8, linewidth=2)
    ax_top.axvline(x_1d[2], color=line_color_1d, linestyle='--', alpha=0.8, linewidth=2)
    ax_top.axvline(x_1d[1], color=line_color_1d, linestyle='-', alpha=0.9, linewidth=2.5)
    
    ax_top.axvline(x_2d[0], color=line_color_2d, linestyle=':', alpha=0.8, linewidth=2)
    ax_top.axvline(x_2d[2], color=line_color_2d, linestyle=':', alpha=0.8, linewidth=2)
    ax_top.axvline(x_2d[1], color=line_color_2d, linestyle='-', alpha=0.9, linewidth=2.5)
    
    # Fill between confidence intervals
    y_hist_max = ax_top.get_ylim()[1]
    ax_top.fill_between([x_1d[0], x_1d[2]], 0, y_hist_max, 
                       alpha=0.3, color=line_color_1d, label='1D 68% CI')
    ax_top.fill_between([x_2d[0], x_2d[2]], 0, y_hist_max, 
                       alpha=0.3, color=line_color_2d, label='2D 68% CI')
    
    ax_top.set_xlim(ax_main.get_xlim())
    ax_top.set_xticks([])
    ax_top.set_ylabel('Density', fontsize=10)
    ax_top.tick_params(axis='y', labelsize=9)
    
    # Right histogram (Y parameter)
    bins = 20
    ax_right.hist(y_samples, bins=bins, orientation='horizontal', 
                 alpha=0.7, color=hist_color_1d, density=True, label='1D marginal')
    ax_right.hist(y_proj, bins=bins, orientation='horizontal', 
                 alpha=0.7, color=hist_color_2d, density=True, label='2D projection')
    
    # Add confidence interval lines for Y parameter
    ax_right.axhline(y_1d[0], color=line_color_1d, linestyle='--', alpha=0.8, linewidth=2)
    ax_right.axhline(y_1d[2], color=line_color_1d, linestyle='--', alpha=0.8, linewidth=2)
    ax_right.axhline(y_1d[1], color=line_color_1d, linestyle='-', alpha=0.9, linewidth=2.5)
    
    ax_right.axhline(y_2d[0], color=line_color_2d, linestyle=':', alpha=0.8, linewidth=2)
    ax_right.axhline(y_2d[2], color=line_color_2d, linestyle=':', alpha=0.8, linewidth=2)
    ax_right.axhline(y_2d[1], color=line_color_2d, linestyle='-', alpha=0.9, linewidth=2.5)
    
    # Fill between confidence intervals
    x_hist_max = ax_right.get_xlim()[1]
    ax_right.fill_betweenx([y_1d[0], y_1d[2]], 0, x_hist_max, 
                          alpha=0.3, color=line_color_1d, label='1D 68% CI')
    ax_right.fill_betweenx([y_2d[0], y_2d[2]], 0, x_hist_max, 
                          alpha=0.3, color=line_color_2d, label='2D 68% CI')
    
    ax_right.set_ylim(ax_main.get_ylim())
    ax_right.set_yticks([])
    ax_right.set_xlabel('Density', fontsize=10)
    ax_right.tick_params(axis='x', labelsize=9)
    
    # Labels and title for main plot
    ax_main.set_xlabel(param_names['x'], fontsize=12)
    ax_main.set_ylabel(param_names['y'], fontsize=12)
    ax_main.set_title(title, fontsize=13, fontweight='bold')
    ax_main.tick_params(labelsize=10)
    
    # Compute correlation and uncertainty ratios
    correlation = np.corrcoef(y_samples, x_samples)[0, 1]
    y_ratio = (y_2d[2] - y_2d[0]) / (y_1d[2] - y_1d[0])
    x_ratio = (x_2d[2] - x_2d[0]) / (x_1d[2] - x_1d[0])
    
    # Compute ellipse properties
    _, _, semi_major, semi_minor, angle = compute_covariance_ellipse(x_samples, y_samples, 0.68)
    ellipse_ratio = semi_major / semi_minor
    angle_deg = np.degrees(angle)
    
    # Add statistics text box
    stats_text = f'r = {correlation:.3f}\n'
    stats_text += f'Ellipse Ratio: {ellipse_ratio:.2f}\n'
    stats_text += f'Angle: {angle_deg:.1f}°\n'
    stats_text += f'Uncertainty Ratio:\n'
    stats_text += f'{param_names["y"]}: {y_ratio:.2f}x\n'
    stats_text += f'{param_names["x"]}: {x_ratio:.2f}x'
    
    ax_main.text(0.02, 0.02, stats_text, 
                transform=ax_main.transAxes, fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.9))
    
    return correlation, y_ratio, x_ratio

def main():
    """Main analysis function"""
    print("=== Correlation Comparison Demo: Strong vs Weak Correlations ===\n")
    
    # Setup paths
    path, path_figures = setup_paths()
    
    # Target and run
    target = 'TWA28'
    run = 'freeslab_lbl10_G2G3_1'
    
    # Load data file
    h5_file = path / target / f'retrieval_outputs/{run}/test_data/chem_posterior.h5'
    
    if not os.path.exists(h5_file):
        raise FileNotFoundError(f"HDF5 file not found: {h5_file}")
    
    print(f"Analyzing {target} - {run}")
    print(f"Data file: {h5_file}")
    
    # Parameter pairs to compare
    param_pairs = [
        ('12CO', 'log_g', 'strong'),  # Strong correlation case
        ('12C/13C', 'log_g', 'weak')      # Weak correlation case
    ]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 8))
    
    results = {}
    
    for i, (param_x, param_y, corr_type) in enumerate(param_pairs):
        print(f"\nAnalyzing parameter pair: {param_y} vs {param_x}")
        
        # Load data
        y_samples, x_samples = load_parameter_pair(str(h5_file), param_x, param_y)
        y_samples, x_samples = clean_samples(y_samples, x_samples)
        
        print(f"Valid samples: {len(y_samples)}")
        
        # Create subplot positions
        left = 0.08 + i * 0.46
        bottom = 0.15
        width = 0.35
        height = 0.6
        
        # Main 2D plot
        ax_main = fig.add_axes([left, bottom, width, height])
        
        # Marginal histograms
        ax_top = fig.add_axes([left, bottom + height + 0.02, width, 0.12])
        ax_right = fig.add_axes([left + width + 0.02, bottom, 0.12, height])
        
        # Parameter names
        if param_x == '12CO':
            param_names = {
                'x': 'log $^{12}$CO VMR',
                'y': 'log g [cgs]'
            }
            title = f'Strong Correlation Example\nlog g vs log $^{{12}}$CO VMR'
        else:  # 12C/13C
            param_names = {
                'x': 'log $^{12}$C/$^{13}$C',
                'y': 'log g [cgs]'
            }
            title = f'Weak Correlation Example\nlog g vs log $^{{12}}$C/$^{{13}}$C'
        
        # Create subplot
        correlation, y_ratio, x_ratio = create_comparison_subplot(
            ax_main, ax_top, ax_right, y_samples, x_samples, 
            param_names, title, corr_type
        )
        
        # Store results
        results[param_x] = {
            'correlation': correlation,
            'y_ratio': y_ratio,
            'x_ratio': x_ratio,
            'n_samples': len(y_samples)
        }
        
        print(f"Correlation: {correlation:.3f}")
        print(f"Uncertainty ratios - log_g: {y_ratio:.2f}x, {param_x}: {x_ratio:.2f}x")
    
    # Add overall legend
    legend_elements = [
        plt.Line2D([0], [0], color='black', linestyle='-', linewidth=2.5, label='Median'),
        plt.Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='1D 68% CI'),
        plt.Line2D([0], [0], color='black', linestyle=':', linewidth=2, label='2D Ellipse CI'),
        plt.Line2D([0], [0], color='black', linestyle='--', linewidth=2.5, alpha=0.8, label='Confidence Ellipse'),
        plt.Line2D([0], [0], color='darkred', linestyle='-', linewidth=2, label='Strong Correlation'),
        plt.Line2D([0], [0], color='darkgreen', linestyle='-', linewidth=2, label='Weak Correlation')
    ]
    
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
              ncol=5, fontsize=11)
    
    # Add overall title
    fig.suptitle(f'{target} ({run.split("_")[-1]}): Ellipse Geometry Method for 2D Uncertainty Estimation\n'
                'Principal Axes Analysis for Correlated Parameters', 
                fontsize=16, fontweight='bold', y=0.95)
    
    # Save plot
    output_file = path_figures / f"correlation_comparison_{target}_{run.split('_')[-1]}.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nPlot saved: {output_file}")
    
    # Print summary comparison
    print("\n=== COMPARISON SUMMARY ===")
    print("Parameter Pair\t\tCorrelation\tUncertainty Ratio (log_g)")
    print("-" * 65)
    
    for param_x in ['12CO', '12C/13C']:
        if param_x in results:
            r = results[param_x]
            corr_strength = "STRONG" if abs(r['correlation']) > 0.8 else "WEAK"
            print(f"log_g vs {param_x}\t\t{r['correlation']:.3f}\t\t{r['y_ratio']:.2f}x\t({corr_strength})")
    
    print("\nKey Findings:")
    print("1. Strong correlation (|r| > 0.8): Significant uncertainty underestimation")
    print("2. Weak correlation (|r| < 0.3): Minimal uncertainty underestimation")
    print("3. 2D approaches are essential only for highly correlated parameters")
    print("4. Standard 1D intervals are adequate for weakly correlated parameters")

if __name__ == "__main__":
    main() 