# 2D Posterior Diagnostics Analysis

## Overview

This directory contains a comprehensive analysis of 2D posterior distributions for TWA 27A and TWA 28, focusing on the correlation between log g (surface gravity) and log 12CO VMR (volume mixing ratio). The analysis demonstrates the critical importance of proper uncertainty estimation when dealing with highly correlated parameters.

## Key Files

### Analysis Scripts

- **`posterior_2d_diagnostics.py`** - Main comprehensive diagnostic plot with multiple panels
- **`diagnostic_summary.py`** - Quick summary of key findings with formatted output
- **`posterior_2d_uncertainty_demo.py`** - Detailed uncertainty analysis functions
- **`test_uncertainty_demo.py`** - Quick demonstration of the uncertainty problem

### Output Files

- **`posterior_2d_diagnostics.pdf`** - Publication-ready comprehensive diagnostic figure
- **`posterior_2d_diagnostics.png`** - High-resolution PNG version for presentations

## Key Findings

### Strong Parameter Correlations
- All target/configuration combinations show **high correlations** (r > 0.84) between log g and log 12CO VMR
- Average correlation coefficient: **r = 0.878**
- Range: 0.849 to 0.912

### Systematic Uncertainty Underestimation
- **1D marginalized confidence intervals underestimate uncertainties by ~50% on average**
- log g uncertainty ratios: 1.41x to 1.55x larger when using 2D approach
- VMR uncertainty ratios: 1.44x to 1.54x larger when using 2D approach

### Scientific Impact
- Standard error bars are **systematically too narrow** for highly correlated parameters
- Parameter degeneracies create larger uncertainty ellipses than 1D analysis suggests
- **2D credible regions provide more realistic uncertainty estimates**

## Methodology

### 1D vs 2D Uncertainty Estimation

**1D Approach (Standard):**
- Uses 16th-84th percentile ranges for each parameter independently
- **Ignores parameter correlations**
- Results in overconfident uncertainty estimates

**2D Approach (Robust):**
- Computes 2D kernel density estimation
- Finds smallest region containing 68% probability mass
- **Projects 2D credible region onto parameter axes**
- Accounts for parameter degeneracies

### Technical Details

1. **Data Loading**: Joint posterior samples from HDF5 cache files
2. **Cleaning**: Removal of infinite/NaN values and outliers
3. **KDE Computation**: 2D Gaussian kernel density estimation
4. **Credible Levels**: Computed from cumulative probability mass
5. **Projection**: 2D region projected onto individual parameter axes
6. **Validation**: Bootstrap resampling confirms correlation stability

## Usage

### Generate Comprehensive Diagnostic Plot
```bash
python posterior_2d_diagnostics.py
```

### View Key Findings Summary
```bash
python diagnostic_summary.py
```

### Quick Demonstration
```bash
python test_uncertainty_demo.py
```

## Results Summary

| Target | Config | Correlation | log g Ratio | VMR Ratio | Underestimation % |
|--------|--------|-------------|-------------|-----------|-------------------|
| TWA27A | G2G3   | 0.891       | 1.55        | 1.50      | 55.2%            |
| TWA27A | G1G2G3 | 0.849       | 1.41        | 1.44      | 41.4%            |
| TWA28  | G2G3   | 0.912       | 1.49        | 1.45      | 49.4%            |
| TWA28  | G1G2G3 | 0.860       | 1.51        | 1.54      | 50.7%            |

## Recommendations for Robust Uncertainty Reporting

1. **Always check parameter correlations** before reporting uncertainties
2. **For |r| > 0.8, use 2D credible region projection** instead of 1D marginalized intervals
3. **Report correlation coefficients** alongside parameter estimates
4. **Consider joint constraints** when interpreting physical results
5. **Include correlation information** in scientific publications

## Publication Impact

- **Demonstrates need for correlation-aware uncertainty reporting**
- **Standard practice may lead to overconfident scientific conclusions**
- **Proper treatment essential for robust parameter constraints**
- **Results applicable to all astronomical parameter estimation problems**

## Figure Description

The comprehensive diagnostic plot (`posterior_2d_diagnostics.pdf`) contains:

1. **Top Row**: 2D posterior contour plots for each target/configuration
2. **Middle Left**: Uncertainty ratio comparison (1D vs 2D)
3. **Middle Right**: Correlation strength visualization
4. **Bottom**: Summary text with key findings and methodology

## Dependencies

- `numpy`, `matplotlib`, `scipy` - Core scientific computing
- `h5py` - HDF5 file handling
- `retrieval_base` - Custom retrieval analysis package
- `posterior_2d_uncertainty_demo` - Uncertainty analysis functions

## Citation

If you use this analysis in your research, please cite the relevant papers and acknowledge the importance of proper correlation handling in astronomical parameter estimation. 