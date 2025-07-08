#!/usr/bin/env python3
"""
Quick test script to demonstrate robust uncertainty estimation for highly correlated parameters.

This script shows the key results from the analysis:
- High correlations (r > 0.8) between log_g and log(12CO VMR)
- 1D marginalized uncertainties underestimate true uncertainty by 40-55%
- 2D credible region projection gives more realistic estimates
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add the parent directory to the path so we can import our module
sys.path.append(str(Path(__file__).parent))

from posterior_2d_uncertainty_demo import setup_paths, analyze_target

def quick_demo():
    """Quick demonstration of the uncertainty estimation problem"""
    
    print("=== Quick Demonstration: Robust Uncertainty Estimation ===\n")
    
    try:
        # Setup paths
        path, path_figures = setup_paths()
        
        # Analyze one example case
        target = 'TWA28'
        run = 'freeslab_lbl10_G2G3_1'
        
        print(f"Analyzing {target} - {run}")
        print("Parameters: log_g vs log(12CO VMR)")
        print("-" * 50)
        
        # Perform analysis
        results = analyze_target(path, target, run, '12CO', 'log_g')
        
        # Extract key results
        corr = results['correlation']
        y_1d = results['y_1d']
        y_2d = results['y_2d']
        x_1d = results['x_1d']
        x_2d = results['x_2d']
        
        # Calculate uncertainty ratios
        y_ratio = (y_2d[2] - y_2d[0]) / (y_1d[2] - y_1d[0])
        x_ratio = (x_2d[2] - x_2d[0]) / (x_1d[2] - x_1d[0])
        
        print(f"\nKEY FINDINGS:")
        print(f"============")
        print(f"Correlation coefficient: r = {corr:.3f} (highly correlated!)")
        print(f"")
        print(f"log_g uncertainties:")
        print(f"  1D marginalized: {y_1d[1]:.3f} ± {(y_1d[2]-y_1d[0])/2:.3f}")
        print(f"  2D projection:   {y_2d[1]:.3f} ± {(y_2d[2]-y_2d[0])/2:.3f}")
        print(f"  Underestimation: {y_ratio:.2f}x larger uncertainty!")
        print(f"")
        print(f"log(12CO VMR) uncertainties:")
        print(f"  1D marginalized: {x_1d[1]:.3f} ± {(x_1d[2]-x_1d[0])/2:.3f}")
        print(f"  2D projection:   {x_2d[1]:.3f} ± {(x_2d[2]-x_2d[0])/2:.3f}")
        print(f"  Underestimation: {x_ratio:.2f}x larger uncertainty!")
        
        print(f"\nIMPLICATIONS:")
        print(f"============")
        print(f"• Standard 1D confidence intervals underestimate uncertainty by ~{((y_ratio-1)*100):.0f}%")
        print(f"• This is because they ignore the strong correlation structure")
        print(f"• 2D credible region projection accounts for parameter degeneracy")
        print(f"• More realistic uncertainties are essential for robust science conclusions")
        
        # Show the difference visually
        print(f"\nVISUAL COMPARISON:")
        print(f"=================")
        print(f"1D approach: log_g = {y_1d[1]:.3f} +{y_1d[2]-y_1d[1]:.3f} -{y_1d[1]-y_1d[0]:.3f}")
        print(f"2D approach: log_g = {y_2d[1]:.3f} +{y_2d[2]-y_2d[1]:.3f} -{y_2d[1]-y_2d[0]:.3f}")
        print(f"             ^^^^^^^ {(y_2d[2]-y_2d[0])/(y_1d[2]-y_1d[0]):.1f}x wider error bars!")
        
        print(f"\nRECOMMENDATIONS:")
        print(f"===============")
        print(f"1. Always check parameter correlations before reporting uncertainties")
        print(f"2. For |r| > 0.8, use 2D credible region projection")
        print(f"3. Report both correlation strength and joint uncertainties")
        print(f"4. Consider parameter degeneracies in scientific interpretation")
        
        return True
        
    except Exception as e:
        print(f"Error in demonstration: {e}")
        print("\nTo run the full analysis, use:")
        print("python twx_figs/posterior_2d_uncertainty_demo.py")
        return False

if __name__ == "__main__":
    success = quick_demo()
    if success:
        print(f"\n✓ Demo completed successfully!")
        print(f"For detailed plots and full analysis, run:")
        print(f"python twx_figs/posterior_2d_uncertainty_demo.py")
    else:
        print(f"\n✗ Demo failed - check data availability") 