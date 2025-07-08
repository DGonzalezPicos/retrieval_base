#!/usr/bin/env python3
"""
Quick demo of the enhanced 2D uncertainty analysis with marginal histograms.

This script generates the combined comparison plot showing:
- 2D posterior contours with confidence regions
- Marginal histograms on top and right axes
- Confidence intervals with dashed (1D) and dotted (2D) lines
- Color-coded uncertainty regions
- Uncertainty ratio annotations

Usage:
    python test_enhanced_uncertainty_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import pathlib
from scipy.stats import gaussian_kde
import retrieval_base.auxiliary_functions as af

# Import functions from the main script
import sys
sys.path.append('/home/dario/phd/retrieval_base/twx_figs')
from posterior_2d_uncertainty_demo import (
    load_joint_posterior, clean_samples, compute_2d_kde, 
    find_credible_levels, project_2d_credible_region, setup_paths
)

def main():
    """Generate the enhanced combined comparison plot"""
    print("=== Enhanced 2D Uncertainty Analysis Demo ===\n")
    
    # Setup paths
    path, path_figures = setup_paths()
    
    # Define targets and runs
    targets_runs = [
        ('TWA27A', 'freeslab_lbl10_G2G3_2'),
        ('TWA27A', 'freeslab_lbl10_G1G2G3_1'),
        ('TWA28', 'freeslab_lbl10_G2G3_1'),
        ('TWA28', 'freeslab_lbl10_G1G2G3_1'),
    ]
    
    # Parameters
    param_x = '12CO'
    param_y = 'log_g'
    param_names = {
        'x': f'log {param_x} VMR',
        'y': f'{param_y} [cgs]'
    }
    
    # Load and analyze data
    all_results = []
    
    for target, run in targets_runs:
        try:
            # Load data
            h5_file = path / target / f'retrieval_outputs/{run}/test_data/chem_posterior.h5'
            
            print(f"Loading {target} - {run}")
            y_samples, x_samples = load_joint_posterior(str(h5_file), param_x, param_y)
            y_samples, x_samples = clean_samples(y_samples, x_samples)
            
            # Compute intervals
            correlation = np.corrcoef(y_samples, x_samples)[0, 1]
            
            # 1D marginalized intervals
            y_1d = np.percentile(y_samples, [16, 50, 84])
            x_1d = np.percentile(x_samples, [16, 50, 84])
            
            # 2D credible region projection
            X_grid, Y_grid, Z_density = compute_2d_kde(y_samples, x_samples)
            levels = find_credible_levels(Z_density, [0.68, 0.95])
            y_proj, x_proj = project_2d_credible_region(X_grid, Y_grid, Z_density, levels[0])
            y_2d = np.percentile(y_proj, [16, 50, 84])
            x_2d = np.percentile(x_proj, [16, 50, 84])
            
            results = {
                'target': target,
                'run': run,
                'correlation': correlation,
                'y_samples': y_samples,
                'x_samples': x_samples,
                'y_1d': y_1d,
                'x_1d': x_1d,
                'y_2d': y_2d,
                'x_2d': x_2d
            }
            
            all_results.append(results)
            print(f"  Correlation: {correlation:.3f}")
            print(f"  Uncertainty ratio: {(y_2d[2]-y_2d[0])/(y_1d[2]-y_1d[0]):.2f}x")
            
        except Exception as e:
            print(f"Error with {target} - {run}: {e}")
            continue
    
    # Create enhanced combined plot
    print(f"\nCreating enhanced combined plot with {len(all_results)} panels...")
    
    fig = plt.figure(figsize=(16, 12))
    
    for i, results in enumerate(all_results):
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
        
        # Get intervals
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
    fig.suptitle('Enhanced 2D Posterior Analysis: Marginal Histograms with Confidence Intervals\n'
                'Robust Uncertainty Estimation for Highly Correlated Parameters', 
                fontsize=14, y=0.98)
    
    # Save plot
    output_file = path_figures / "enhanced_posterior_2d_uncertainty_demo.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Enhanced plot saved: {output_file}")
    print("\nKey Features:")
    print("✓ 2D posterior contours (68% and 95% credible regions)")
    print("✓ Marginal histograms on top and right axes")
    print("✓ Confidence intervals with dashed lines (1D) and dotted lines (2D)")
    print("✓ Color-coded uncertainty regions")
    print("✓ Uncertainty ratio annotations")
    print("✓ Consistent color scheme across all panels")
    
    # Print summary
    print("\n=== Summary ===")
    for results in all_results:
        target = results['target']
        run = results['run'].split('_')[-1]
        corr = results['correlation']
        y_1d_width = results['y_1d'][2] - results['y_1d'][0]
        y_2d_width = results['y_2d'][2] - results['y_2d'][0]
        ratio = y_2d_width / y_1d_width
        
        print(f"{target} ({run}): r = {corr:.3f}, uncertainty ratio = {ratio:.2f}x")

if __name__ == "__main__":
    main() 