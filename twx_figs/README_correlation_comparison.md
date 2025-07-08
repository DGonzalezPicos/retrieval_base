# Correlation Comparison Analysis: Strong vs Weak Parameter Correlations

This analysis demonstrates how parameter correlation strength affects uncertainty estimation in atmospheric retrieval analysis, using TWA28 G2G3 as a case study.

## 🎯 Objective

Compare uncertainty estimation methods for:
1. **Strong correlation** case: log_g vs 12CO VMR (r = 0.912)
2. **Weak correlation** case: log_g vs TiO VMR (r = -0.143)

## 📊 Key Results

### Strong Correlation Example (log_g vs 12CO)
- **Correlation coefficient**: r = 0.912
- **Uncertainty underestimation**: 1.49x for log_g, 1.45x for 12CO
- **Visual characteristics**: 
  - Tilted, elongated 2D posterior ellipse
  - Significant difference between 1D and 2D confidence intervals
  - Red color scheme in visualization

### Weak Correlation Example (log_g vs TiO)
- **Correlation coefficient**: r = -0.143
- **Uncertainty underestimation**: Minimal (TiO shows 1.37x ratio)
- **Visual characteristics**:
  - Nearly circular 2D posterior distribution
  - Minimal difference between 1D and 2D confidence intervals
  - Green color scheme in visualization

## 🔬 Full Parameter Survey

Analysis of all available parameters in TWA28 G2G3:

| Parameter | Correlation (r) | Strength | Recommendation |
|-----------|-----------------|----------|----------------|
| 12CO      | 0.912          | STRONG   | 2D approach essential |
| H2O       | 0.966          | STRONG   | 2D approach essential |
| AlH       | 0.580          | MODERATE | Consider 2D approach |
| K         | 0.219          | WEAK     | 1D intervals adequate |
| TiO       | -0.143         | WEAK     | 1D intervals adequate |
| VO        | -0.006         | WEAK     | 1D intervals adequate |

## 🎨 Visualization Features

### Color Coding
- **Red**: Strong correlation (12CO) - shows significant uncertainty underestimation
- **Green**: Weak correlation (TiO) - shows minimal uncertainty underestimation

### Plot Components
- **2D Contours**: 68% and 95% credible regions
- **Marginal Histograms**: Top and right axes showing 1D vs 2D distributions
- **Confidence Intervals**: 
  - Dashed lines (--): 1D marginalized intervals
  - Dotted lines (⋯): 2D credible region projections
- **Statistics Box**: Correlation coefficient and uncertainty ratios

## 🛠️ Scripts

### Main Analysis
```bash
python correlation_comparison_demo.py
```
**Output**: `correlation_comparison_TWA28_1.pdf`

### Quick Summary
```bash
python quick_correlation_demo.py
```
**Output**: Terminal summary with recommendations

## 📈 Scientific Implications

### When to Use 2D Approaches
- **|r| > 0.8**: 2D credible region projection ESSENTIAL
- **0.3 ≤ |r| ≤ 0.8**: Consider 2D approach for critical analysis
- **|r| < 0.3**: Standard 1D intervals adequate

### Uncertainty Underestimation
- **Strong correlations**: 40-55% underestimation with 1D methods
- **Weak correlations**: <10% underestimation with 1D methods

## 🔍 Technical Details

### Data Processing
- VMR parameters averaged over pressure levels
- Log transformation applied to VMR values
- Sample cleaning removes NaN and infinite values
- 10,012 valid posterior samples analyzed

### Statistical Methods
- 2D Kernel Density Estimation (KDE)
- Credible region calculation via cumulative probability
- Projection of 2D regions onto parameter axes
- Pearson correlation coefficient calculation

## 📊 Visual Proof

The side-by-side comparison clearly shows:

1. **Strong Correlation (12CO)**:
   - Tilted elliptical posterior
   - Large difference between 1D (dashed) and 2D (dotted) intervals
   - Marginal histograms show different shapes for 1D vs 2D projections

2. **Weak Correlation (TiO)**:
   - Nearly circular posterior distribution
   - Minimal difference between 1D and 2D intervals
   - Marginal histograms show similar shapes for 1D vs 2D projections

## 🎯 Recommendations

### For Atmospheric Retrieval Analysis
1. **Always check correlations** before reporting uncertainties
2. **Use 2D approaches** for |r| > 0.8 parameter pairs
3. **Standard 1D intervals** are adequate for |r| < 0.3
4. **Consider 2D methods** for moderate correlations (0.3 ≤ |r| ≤ 0.8)

### For Publication
- Report correlation coefficients alongside uncertainties
- Use 2D credible region projections for highly correlated parameters
- Acknowledge uncertainty method in methodology section
- Consider showing correlation matrix for key parameters

## 🚀 Future Work

1. **Multi-target comparison**: Extend to other brown dwarfs
2. **Multi-parameter analysis**: >2 parameters simultaneously
3. **Bayesian model comparison**: Evidence-based uncertainty
4. **Automated correlation screening**: Pipeline integration

## 📚 Related Work

- **Enhanced 2D uncertainty demo**: `posterior_2d_uncertainty_demo.py`
- **Pearson correlation analysis**: `fig_uncertainties_pearson.py`
- **Comprehensive diagnostics**: `posterior_2d_diagnostics.py`

## 🔧 Dependencies

```python
numpy >= 1.20
matplotlib >= 3.5
scipy >= 1.7
h5py >= 3.0
```

## 📞 Contact

For questions about correlation analysis in atmospheric retrievals:
- **Statistical Methods**: 2D credible region projection
- **Visualization**: Side-by-side correlation comparison
- **Scientific Interpretation**: When to use different uncertainty methods

---

**Key Message**: Correlation strength determines the appropriate uncertainty estimation method. This analysis provides both statistical evidence and visual proof of when 2D approaches become essential for robust scientific conclusions. 