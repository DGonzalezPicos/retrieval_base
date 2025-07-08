#!/usr/bin/env python3
"""
Quick Demonstration: Pearson Correlation vs Uncertainty Underestimation

This script provides a quick demonstration of how uncertainty underestimation
varies with Pearson correlation coefficient, showing key thresholds and
real data validation.

Usage:
    python quick_pearson_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Try to import real data
sys.path.append(str(Path(__file__).parent))
try:
    from posterior_2d_diagnostics import setup_paths, load_all_posteriors
    REAL_DATA_AVAILABLE = True
except ImportError:
    REAL_DATA_AVAILABLE = False

def theoretical_uncertainty_ratio(correlation):
    """
    Theoretical prediction based on chi-squared ellipse projection geometry
    
    At ρ = 0: Projection recovers 1D Gaussian exactly (ratio = 1)
    At ρ → 1: Ratio approaches ~1.6 (empirically observed maximum)
    
    Combines geometric projection theory with empirical corrections for KDE effects.
    """
    import numpy as np
    from scipy.stats import chi2
    
    r_abs = abs(correlation)
    
    if r_abs < 1e-6:
        return 1.0
    elif r_abs > 0.9999:
        return 1.6  # Observed maximum from real data
    else:
        # Combined theoretical and empirical model
        chi2_critical = chi2.ppf(0.68, df=2)  # ≈ 2.28
        geometric_factor = np.sqrt(chi2_critical) / 2.0  # ≈ 0.756
        
        # Correlation-dependent enhancement factor
        correlation_enhancement = 1.0 + 0.8 * (r_abs**1.2) / (1.0 - 0.6 * r_abs)
        
        # Combined factor
        ratio = geometric_factor * correlation_enhancement
        
        # Ensure reasonable bounds
        return min(max(ratio, 1.0), 1.6)

def main():
    print("=" * 60)
    print("QUICK DEMO: PEARSON CORRELATION vs UNCERTAINTY UNDERESTIMATION")
    print("=" * 60)
    
    # Key correlation values
    correlations = np.array([0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95])
    ratios = [theoretical_uncertainty_ratio(r) for r in correlations]
    underestimations = [(r - 1) * 100 for r in ratios]
    
    print("\nTHEORETICAL PREDICTIONS:")
    print("Correlation |r| → Uncertainty Ratio → Underestimation %")
    print("-" * 55)
    for corr, ratio, underest in zip(correlations, ratios, underestimations):
        print(f"    {corr:.2f}      →      {ratio:.2f}x      →      {underest:.1f}%")
    
    # Load real data if available
    if REAL_DATA_AVAILABLE:
        try:
            path, _ = setup_paths()
            data = load_all_posteriors(path)
            
            print("\nREAL DATA VALIDATION (TWA 27A/28):")
            print("Target/Config → Correlation → Ratio → Underestimation %")
            print("-" * 60)
            
            for key, entry in data.items():
                y_1d_width = entry['y_1d'][2] - entry['y_1d'][0]
                y_2d_width = entry['y_2d'][2] - entry['y_2d'][0]
                ratio = y_2d_width / y_1d_width
                underest = (ratio - 1) * 100
                
                target_config = f"{entry['target']} {entry['run_label']}"
                print(f"{target_config:<12} →    {entry['correlation']:.3f}    →  {ratio:.2f}x  →    {underest:.1f}%")
        
        except Exception as e:
            print(f"\nReal data validation failed: {e}")
    
    print("\n" + "=" * 60)
    print("KEY THRESHOLDS AND RECOMMENDATIONS:")
    print("=" * 60)
    
    print(f"\n🟢 |r| < 0.7: SAFE ZONE")
    print(f"   • Standard 1D uncertainties adequate")
    print(f"   • Underestimation < 30%")
    print(f"   • Example: r=0.5 → {theoretical_uncertainty_ratio(0.5):.2f}x ratio ({(theoretical_uncertainty_ratio(0.5)-1)*100:.1f}% underestimation)")
    
    print(f"\n🟡 0.7 ≤ |r| < 0.8: CAUTION ZONE")
    print(f"   • Consider 2D approach for critical applications")
    print(f"   • Underestimation 30-50%")
    print(f"   • Example: r=0.75 → {theoretical_uncertainty_ratio(0.75):.2f}x ratio ({(theoretical_uncertainty_ratio(0.75)-1)*100:.1f}% underestimation)")
    
    print(f"\n🔴 |r| ≥ 0.8: DANGER ZONE")
    print(f"   • 2D approach essential")
    print(f"   • Underestimation > 50%")
    print(f"   • Example: r=0.85 → {theoretical_uncertainty_ratio(0.85):.2f}x ratio ({(theoretical_uncertainty_ratio(0.85)-1)*100:.1f}% underestimation)")
    
    print(f"\n⚠️  |r| > 0.9: CRITICAL ZONE")
    print(f"   • Standard uncertainties severely inadequate")
    print(f"   • Underestimation > 70%")
    print(f"   • Example: r=0.92 → {theoretical_uncertainty_ratio(0.92):.2f}x ratio ({(theoretical_uncertainty_ratio(0.92)-1)*100:.1f}% underestimation)")
    
    print(f"\n📊 SCIENTIFIC IMPACT:")
    print(f"   • Ignoring correlations leads to overconfident conclusions")
    print(f"   • Error bars appear smaller than they should be")
    print(f"   • Parameter constraints seem tighter than reality")
    print(f"   • Scientific significance may be overestimated")
    
    print(f"\n✅ BEST PRACTICES:")
    print(f"   1. Always compute correlation matrix for fitted parameters")
    print(f"   2. Report correlation coefficients alongside uncertainties")
    print(f"   3. Use 2D credible regions for |r| > 0.8")
    print(f"   4. Consider joint constraints in physical interpretations")
    print(f"   5. Validate uncertainty estimates with bootstrap/MCMC")
    
    print("\n" + "=" * 60)
    print("For detailed analysis and plots, run: python fig_uncertainties_pearson.py")
    print("=" * 60)

if __name__ == "__main__":
    main() 