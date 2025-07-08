#!/usr/bin/env python3
"""
Ellipse Geometry Comparison: Advanced 2D Uncertainty Estimation

This script compares three methods for uncertainty estimation:
1. 1D Marginalized intervals (standard approach)
2. KDE-based 2D credible region projection 
3. Ellipse geometry method using principal axes

The ellipse geometry method is more robust for non-Gaussian posteriors
and provides a direct analytical approach to capturing correlation structure.

Usage:
    python ellipse_geometry_comparison.py

Requirements:
    - h5py, numpy, matplotlib, scipy, sklearn (optional)
    - Access to TWA28 G2G3 posterior data
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
import h5py
import os
import pathlib
from scipy.stats import gaussian_kde, chi2
from typing import Tuple, List, Dict, Optional
import retrieval_base.auxiliary_functions as af

# Import functions from correlation comparison script
import sys
sys.path.append('/home/dario/phd/retrieval_base/twx_figs')
from correlation_comparison_demo import (
    compute_covariance_ellipse, compute_ellipse_projected_intervals,
    compute_robust_ellipse_intervals, load_parameter_pair, clean_samples
)
from posterior_2d_uncertainty_demo import (
    compute_2d_kde, find_credible_levels, project_2d_credible_region
)

# Configuration
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['figure.dpi'] = 100

def compare_uncertainty_methods(x_samples: np.ndarray, y_samples: np.ndarray) -> Dict:
    """
    Compare all three uncertainty estimation methods.
    
    Args:
        x_samples: X parameter samples
        y_samples: Y parameter samples
    
    Returns:
        Dictionary with results from all methods
    """
    results = {}
    
    # Method 1: 1D Marginalized intervals
    y_1d = np.percentile(y_samples, [16, 50, 84])
    x_1d = np.percentile(x_samples, [16, 50, 84])
    results['1d'] = {'y': y_1d, 'x': x_1d}
    
    # Method 2: KDE-based 2D credible region projection
    X_grid, Y_grid, Z_density = compute_2d_kde(y_samples, x_samples)
    levels = find_credible_levels(Z_density, [0.68, 0.95])
    y_proj, x_proj = project_2d_credible_region(X_grid, Y_grid, Z_density, levels[0])
    y_2d_kde = np.percentile(y_proj, [16, 50, 84])
    x_2d_kde = np.percentile(x_proj, [16, 50, 84])
    results['kde'] = {'y': y_2d_kde, 'x': x_2d_kde, 'levels': levels, 
                      'X_grid': X_grid, 'Y_grid': Y_grid, 'Z_density': Z_density}
    
    # Method 3: Ellipse geometry method
    y_ellipse, x_ellipse = compute_ellipse_projected_intervals(x_samples, y_samples, 0.68)
    ellipse_x, ellipse_y, semi_major, semi_minor, angle = compute_covariance_ellipse(
        x_samples, y_samples, 0.68)
    results['ellipse'] = {
        'y': y_ellipse, 'x': x_ellipse,
        'ellipse_x': ellipse_x, 'ellipse_y': ellipse_y,
        'semi_major': semi_major, 'semi_minor': semi_minor, 'angle': angle
    }
    
    # Method 4: Robust ellipse method (if sklearn available)
    try:
        y_robust, x_robust = compute_robust_ellipse_intervals(x_samples, y_samples, 0.68)
        results['robust'] = {'y': y_robust, 'x': x_robust}
    except:
        results['robust'] = None
    
    # Compute uncertainty ratios
    y_1d_width = y_1d[2] - y_1d[0]
    x_1d_width = x_1d[2] - x_1d[0]
    
    results['ratios'] = {
        'kde_y': (y_2d_kde[2] - y_2d_kde[0]) / y_1d_width,
        'kde_x': (x_2d_kde[2] - x_2d_kde[0]) / x_1d_width,
        'ellipse_y': (y_ellipse[2] - y_ellipse[0]) / y_1d_width,
        'ellipse_x': (x_ellipse[2] - x_ellipse[0]) / x_1d_width
    }
    
    if results['robust'] is not None:
        results['ratios']['robust_y'] = (y_robust[2] - y_robust[0]) / y_1d_width
        results['ratios']['robust_x'] = (x_robust[2] - x_robust[0]) / x_1d_width
    
    return results

def create_method_comparison_plot(y_samples: np.ndarray, x_samples: np.ndarray, 
                                param_names: Dict[str, str], title: str) -> plt.Figure:
    """
    Create comprehensive comparison plot showing all uncertainty methods.
    
    Args:
        y_samples: Y parameter samples
        x_samples: X parameter samples
        param_names: Dictionary with parameter labels
        title: Plot title
    
    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=(20, 12))
    
    # Get results from all methods
    results = compare_uncertainty_methods(x_samples, y_samples)
    correlation = np.corrcoef(y_samples, x_samples)[0, 1]
    
    # Create subplot layout: 2x3 grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Main comparison plot (top row, spans 2 columns)
    ax_main = fig.add_subplot(gs[0, :2])
    
    # Method details (top right)
    ax_details = fig.add_subplot(gs[0, 2])
    
    # Individual method plots (bottom row)
    ax_1d = fig.add_subplot(gs[1, 0])
    ax_kde = fig.add_subplot(gs[1, 1])
    ax_ellipse = fig.add_subplot(gs[1, 2])
    
    # === Main comparison plot ===
    # Scatter plot
    n_plot = min(3000, len(y_samples))
    idx = np.random.choice(len(y_samples), n_plot, replace=False)
    ax_main.scatter(x_samples[idx], y_samples[idx], s=0.8, alpha=0.4, color='gray', zorder=1)
    
    # KDE contours
    kde_data = results['kde']
    ax_main.contour(kde_data['X_grid'], kde_data['Y_grid'], kde_data['Z_density'], 
                   levels=kde_data['levels'], colors=['blue', 'navy'], 
                   linewidths=[2, 1.5], alpha=0.8, zorder=3)
    
    # Confidence ellipse
    ellipse_data = results['ellipse']
    ax_main.plot(ellipse_data['ellipse_x'], ellipse_data['ellipse_y'], 
                color='red', linewidth=3, linestyle='--', alpha=0.9, zorder=4,
                label='Ellipse Geometry')
    
    # Method intervals as lines
    y_1d, x_1d = results['1d']['y'], results['1d']['x']
    y_kde, x_kde = results['kde']['y'], results['kde']['x']
    y_ellipse, x_ellipse = results['ellipse']['y'], results['ellipse']['x']
    
    # Vertical lines (X intervals)
    ax_main.axvline(x_1d[0], color='green', linestyle='-', alpha=0.7, linewidth=2)
    ax_main.axvline(x_1d[2], color='green', linestyle='-', alpha=0.7, linewidth=2)
    ax_main.axvline(x_kde[0], color='blue', linestyle=':', alpha=0.8, linewidth=2)
    ax_main.axvline(x_kde[2], color='blue', linestyle=':', alpha=0.8, linewidth=2)
    ax_main.axvline(x_ellipse[0], color='red', linestyle='--', alpha=0.8, linewidth=2)
    ax_main.axvline(x_ellipse[2], color='red', linestyle='--', alpha=0.8, linewidth=2)
    
    # Horizontal lines (Y intervals)
    ax_main.axhline(y_1d[0], color='green', linestyle='-', alpha=0.7, linewidth=2)
    ax_main.axhline(y_1d[2], color='green', linestyle='-', alpha=0.7, linewidth=2)
    ax_main.axhline(y_kde[0], color='blue', linestyle=':', alpha=0.8, linewidth=2)
    ax_main.axhline(y_kde[2], color='blue', linestyle=':', alpha=0.8, linewidth=2)
    ax_main.axhline(y_ellipse[0], color='red', linestyle='--', alpha=0.8, linewidth=2)
    ax_main.axhline(y_ellipse[2], color='red', linestyle='--', alpha=0.8, linewidth=2)
    
    ax_main.set_xlabel(param_names['x'], fontsize=12)
    ax_main.set_ylabel(param_names['y'], fontsize=12)
    ax_main.set_title(f'{title}\nMethod Comparison (r = {correlation:.3f})', fontsize=13)
    
    # === Method details text ===
    ax_details.axis('off')
    
    # Calculate uncertainty ratios
    ratios = results['ratios']
    
    details_text = f"""
METHOD COMPARISON

Correlation: r = {correlation:.3f}

1D MARGINALIZED:
{param_names['y']}: {y_1d[1]:.3f} ±{(y_1d[2]-y_1d[0])/2:.3f}
{param_names['x']}: {x_1d[1]:.3f} ±{(x_1d[2]-x_1d[0])/2:.3f}

KDE 2D PROJECTION:
{param_names['y']}: {y_kde[1]:.3f} ±{(y_kde[2]-y_kde[0])/2:.3f}
{param_names['x']}: {x_kde[1]:.3f} ±{(x_kde[2]-x_kde[0])/2:.3f}
Ratio: {ratios['kde_y']:.2f}x, {ratios['kde_x']:.2f}x

ELLIPSE GEOMETRY:
{param_names['y']}: {y_ellipse[1]:.3f} ±{(y_ellipse[2]-y_ellipse[0])/2:.3f}
{param_names['x']}: {x_ellipse[1]:.3f} ±{(x_ellipse[2]-x_ellipse[0])/2:.3f}
Ratio: {ratios['ellipse_y']:.2f}x, {ratios['ellipse_x']:.2f}x

ELLIPSE PROPERTIES:
Semi-major: {ellipse_data['semi_major']:.3f}
Semi-minor: {ellipse_data['semi_minor']:.3f}
Ratio: {ellipse_data['semi_major']/ellipse_data['semi_minor']:.2f}
Angle: {np.degrees(ellipse_data['angle']):.1f}°
    """
    
    if results['robust'] is not None:
        y_robust, x_robust = results['robust']['y'], results['robust']['x']
        details_text += f"""
ROBUST ELLIPSE:
{param_names['y']}: {y_robust[1]:.3f} ±{(y_robust[2]-y_robust[0])/2:.3f}
{param_names['x']}: {x_robust[1]:.3f} ±{(x_robust[2]-x_robust[0])/2:.3f}
Ratio: {ratios['robust_y']:.2f}x, {ratios['robust_x']:.2f}x
        """
    
    ax_details.text(0.05, 0.95, details_text, transform=ax_details.transAxes, 
                   fontsize=9, verticalalignment='top', fontfamily='monospace')
    
    # === Individual method plots ===
    methods = [
        ('1D Marginalized', results['1d'], 'green', ax_1d),
        ('KDE 2D Projection', results['kde'], 'blue', ax_kde),
        ('Ellipse Geometry', results['ellipse'], 'red', ax_ellipse)
    ]
    
    for method_name, method_data, color, ax in methods:
        # Scatter plot
        idx_small = np.random.choice(len(y_samples), min(1000, len(y_samples)), replace=False)
        ax.scatter(x_samples[idx_small], y_samples[idx_small], s=1, alpha=0.3, color='gray')
        
        if method_name == 'KDE 2D Projection':
            # Show KDE contours
            ax.contour(method_data['X_grid'], method_data['Y_grid'], method_data['Z_density'], 
                      levels=method_data['levels'], colors=[color], linewidths=[2, 1.5])
        elif method_name == 'Ellipse Geometry':
            # Show confidence ellipse
            ax.plot(method_data['ellipse_x'], method_data['ellipse_y'], 
                   color=color, linewidth=2.5, linestyle='--')
        
        # Show intervals
        y_int, x_int = method_data['y'], method_data['x']
        
        # Confidence interval box
        width = x_int[2] - x_int[0]
        height = y_int[2] - y_int[0]
        rect = patches.Rectangle((x_int[0], y_int[0]), width, height,
                               linewidth=2, edgecolor=color, facecolor=color, alpha=0.2)
        ax.add_patch(rect)
        
        # Median lines
        ax.axvline(x_int[1], color=color, linestyle='-', linewidth=2, alpha=0.8)
        ax.axhline(y_int[1], color=color, linestyle='-', linewidth=2, alpha=0.8)
        
        ax.set_xlabel(param_names['x'], fontsize=10)
        ax.set_ylabel(param_names['y'], fontsize=10)
        ax.set_title(method_name, fontsize=11, color=color, fontweight='bold')
        ax.tick_params(labelsize=9)
    
    # Overall legend
    legend_elements = [
        plt.Line2D([0], [0], color='green', linestyle='-', linewidth=2, label='1D Marginalized'),
        plt.Line2D([0], [0], color='blue', linestyle=':', linewidth=2, label='KDE 2D Projection'),
        plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Ellipse Geometry'),
        plt.Line2D([0], [0], color='gray', marker='o', linestyle='', markersize=3, label='Posterior Samples')
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02), 
              ncol=4, fontsize=11)
    
    return fig

def main():
    """Main analysis function"""
    print("=== Ellipse Geometry Method Comparison ===\n")
    
    # Setup paths
    path = af.get_path(return_pathlib=True)
    path_figures = pathlib.Path('/home/dario/phd/twa2x_paper/figures')
    
    # Target and run
    target = 'TWA28'
    run = 'freeslab_lbl10_G2G3_1'
    
    # Load data file
    h5_file = path / target / f'retrieval_outputs/{run}/test_data/chem_posterior.h5'
    
    if not os.path.exists(h5_file):
        raise FileNotFoundError(f"HDF5 file not found: {h5_file}")
    
    print(f"Analyzing {target} - {run}")
    
    # Test both strong and weak correlation cases
    test_cases = [
        ('12CO', 'Strong Correlation Example'),
        ('12C/13C', 'Weak Correlation Example')
    ]
    
    for param_x, case_name in test_cases:
        print(f"\n=== {case_name}: log_g vs {param_x} ===")
        
        # Load data
        y_samples, x_samples = load_parameter_pair(str(h5_file), param_x, 'log_g')
        y_samples, x_samples = clean_samples(y_samples, x_samples)
        
        correlation = np.corrcoef(y_samples, x_samples)[0, 1]
        print(f"Correlation: {correlation:.3f}")
        print(f"Valid samples: {len(y_samples)}")
        
        # Parameter names
        if param_x == '12CO':
            param_names = {
                'x': 'log $^{12}$CO VMR',
                'y': 'log g [cgs]'
            }
        else:
            param_names = {
                'x': 'log $^{12}$C/$^{13}$C',
                'y': 'log g [cgs]'
            }
        
        # Compare methods
        results = compare_uncertainty_methods(x_samples, y_samples)
        
        print("\nMethod Comparison:")
        print("-" * 50)
        
        # 1D method
        y_1d, x_1d = results['1d']['y'], results['1d']['x']
        print(f"1D Marginalized:")
        print(f"  {param_names['y']}: {y_1d[1]:.3f} +{y_1d[2]-y_1d[1]:.3f} -{y_1d[1]-y_1d[0]:.3f}")
        print(f"  {param_names['x']}: {x_1d[1]:.3f} +{x_1d[2]-x_1d[1]:.3f} -{x_1d[1]-x_1d[0]:.3f}")
        
        # KDE method
        y_kde, x_kde = results['kde']['y'], results['kde']['x']
        print(f"KDE 2D Projection:")
        print(f"  {param_names['y']}: {y_kde[1]:.3f} +{y_kde[2]-y_kde[1]:.3f} -{y_kde[1]-y_kde[0]:.3f}")
        print(f"  {param_names['x']}: {x_kde[1]:.3f} +{x_kde[2]-x_kde[1]:.3f} -{x_kde[1]-x_kde[0]:.3f}")
        
        # Ellipse method
        y_ellipse, x_ellipse = results['ellipse']['y'], results['ellipse']['x']
        print(f"Ellipse Geometry:")
        print(f"  {param_names['y']}: {y_ellipse[1]:.3f} +{y_ellipse[2]-y_ellipse[1]:.3f} -{y_ellipse[1]-y_ellipse[0]:.3f}")
        print(f"  {param_names['x']}: {x_ellipse[1]:.3f} +{x_ellipse[2]-x_ellipse[1]:.3f} -{x_ellipse[1]-x_ellipse[0]:.3f}")
        
        # Uncertainty ratios
        ratios = results['ratios']
        print(f"\nUncertainty Ratios (vs 1D):")
        print(f"  KDE method: {ratios['kde_y']:.2f}x ({param_names['y']}), {ratios['kde_x']:.2f}x ({param_names['x']})")
        print(f"  Ellipse method: {ratios['ellipse_y']:.2f}x ({param_names['y']}), {ratios['ellipse_x']:.2f}x ({param_names['x']})")
        
        # Ellipse properties
        ellipse_data = results['ellipse']
        ellipse_ratio = ellipse_data['semi_major'] / ellipse_data['semi_minor']
        angle_deg = np.degrees(ellipse_data['angle'])
        print(f"\nEllipse Properties:")
        print(f"  Semi-major axis: {ellipse_data['semi_major']:.3f}")
        print(f"  Semi-minor axis: {ellipse_data['semi_minor']:.3f}")
        print(f"  Axis ratio: {ellipse_ratio:.2f}")
        print(f"  Rotation angle: {angle_deg:.1f}°")
        
        # Create comparison plot
        fig = create_method_comparison_plot(y_samples, x_samples, param_names, 
                                          f"{target} {case_name}")
        
        # Save plot
        safe_param = param_x.replace('/', '_')
        output_file = path_figures / f"ellipse_geometry_comparison_{target}_{safe_param}.pdf"
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"\nPlot saved: {output_file}")
    
    print("\n" + "=" * 60)
    print("ELLIPSE GEOMETRY METHOD ADVANTAGES")
    print("=" * 60)
    print("1. ANALYTICAL: Direct computation from covariance matrix")
    print("2. ROBUST: Works well for non-Gaussian distributions")
    print("3. EFFICIENT: No need for KDE computation or grid evaluation")
    print("4. INTERPRETABLE: Clear geometric meaning (principal axes)")
    print("5. STABLE: Less sensitive to sample size and binning choices")
    print("6. COMPLETE: Captures full correlation structure in ellipse parameters")
    
    print(f"\nAnalysis complete! Enhanced plots saved to {path_figures}")

if __name__ == "__main__":
    main() 