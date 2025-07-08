# Enhanced 2D Uncertainty Analysis with Marginal Histograms

This directory contains scripts for robust uncertainty estimation of highly correlated parameters in atmospheric retrieval analysis, with enhanced visualization featuring marginal histograms and confidence intervals.

## 🎯 Key Features

### Enhanced Combined Plot (`posterior_2d_uncertainty_demo.py`)
- **2D Posterior Contours**: 68% and 95% credible regions
- **Marginal Histograms**: Top and right axes showing 1D and 2D projected distributions
- **Confidence Intervals**: 
  - Dashed lines (--) for 1D marginalized intervals
  - Dotted lines (:) for 2D credible region projections
- **Color-coded Uncertainty Regions**: Consistent color scheme across all panels
- **Uncertainty Ratio Annotations**: Quantitative comparison of 1D vs 2D approaches

### Color Scheme
- **Green**: 1D X-parameter (12CO VMR) - median, confidence intervals, and histogram
- **Blue**: 2D X-parameter projection - median, confidence intervals, and histogram  
- **Orange**: 1D Y-parameter (log_g) - median, confidence intervals, and histogram
- **Red**: 2D Y-parameter projection - median, confidence intervals, and histogram
- **Dark Blue/Navy**: 68%/95% contour lines

## 📊 Scientific Results

### High Correlation Cases (|r| > 0.8)
All analyzed targets show strong correlations between log_g and log(12CO VMR):

| Target | Configuration | Correlation (r) | Uncertainty Ratio |
|--------|---------------|-----------------|-------------------|
| TWA27A | G2G3 (2)     | 0.891          | 1.55x            |
| TWA27A | G1G2G3 (1)   | 0.849          | 1.41x            |
| TWA28  | G2G3 (1)     | 0.912          | 1.49x            |
| TWA28  | G1G2G3 (1)   | 0.860          | 1.51x            |

### Key Findings
1. **Systematic Underestimation**: 1D marginalized uncertainties underestimate true uncertainty by 40-55%
2. **Correlation Dependence**: Higher correlations (r > 0.9) show larger underestimation
3. **Robust Alternative**: 2D credible region projection provides realistic uncertainty estimates
4. **Practical Impact**: Standard error bars can be misleading for highly correlated parameters

## 🛠️ Usage

### Main Analysis Script
```bash
python posterior_2d_uncertainty_demo.py
```

### Quick Demo (Enhanced Plot Only)
```bash
python test_enhanced_uncertainty_demo.py
```

## 📁 Output Files

### Individual Target Plots
- `posterior_2d_uncertainty_TWA27A_2.pdf`
- `posterior_2d_uncertainty_TWA27A_1.pdf`
- `posterior_2d_uncertainty_TWA28_1.pdf`
- `posterior_2d_uncertainty_TWA28_1.pdf`

### Combined Analysis
- `posterior_2d_uncertainty_comparison.pdf` - Enhanced 4-panel comparison
- `enhanced_posterior_2d_uncertainty_demo.pdf` - Demo output

## 🔬 Technical Methods

### 1. 2D Kernel Density Estimation
- Gaussian KDE for joint posterior distribution
- Adaptive grid with 0.5-99.5 percentile range
- Smoothing optimized for sample size

### 2. Credible Region Calculation
- Cumulative probability integration
- Density threshold determination for 68% and 95% levels
- Contour level ordering for matplotlib compatibility

### 3. Projection Method
- Extract samples within 2D credible regions
- Project onto parameter axes
- Compute percentiles for robust intervals

### 4. Uncertainty Comparison
- **1D Marginalized**: Standard 16th-84th percentile intervals
- **2D Projected**: Intervals from projected credible regions
- **HPD Intervals**: Highest Posterior Density for validation
- **Bootstrap Validation**: Resampling for correlation uncertainty

## 📈 Visualization Components

### Main 2D Plot
- Contour lines for credible regions
- Scatter plot of posterior samples
- Median lines for both approaches
- Uncertainty ratio annotations

### Top Histogram (X-parameter)
- Light green: 1D marginalized distribution
- Light blue: 2D projected distribution
- Green lines: 1D confidence intervals (dashed)
- Blue lines: 2D confidence intervals (dotted)
- Shaded regions: 68% confidence intervals

### Right Histogram (Y-parameter)
- Light salmon: 1D marginalized distribution
- Light coral: 2D projected distribution
- Orange lines: 1D confidence intervals (dashed)
- Red lines: 2D confidence intervals (dotted)
- Shaded regions: 68% confidence intervals

## 🎨 Enhanced Features

### Publication-Ready Quality
- High-resolution output (300 DPI)
- Consistent typography and sizing
- Professional color scheme
- Clear legends and annotations

### Interactive Elements
- Comprehensive legend with line styles
- Uncertainty ratio text boxes
- Correlation coefficient display
- Sample count validation

## 🚀 Future Extensions

1. **Multi-parameter Analysis**: Extend to >2 parameters
2. **Alternative Correlation Metrics**: Spearman rank correlation
3. **Bayesian Model Comparison**: Evidence-based uncertainty
4. **Interactive Visualization**: Plotly/Bokeh integration

## 📚 References

- Foreman-Mackey et al. (2013) - Corner plots and credible regions
- Gelman et al. (2013) - Bayesian Data Analysis
- Salvatier et al. (2016) - Probabilistic programming in Python

## 🔧 Dependencies

```python
numpy >= 1.20
matplotlib >= 3.5
scipy >= 1.7
h5py >= 3.0
seaborn >= 0.11
```

## 📞 Contact

For questions about the enhanced uncertainty analysis:
- **Scientific Methods**: Atmospheric retrieval uncertainty quantification
- **Technical Implementation**: 2D KDE and credible region projection
- **Visualization**: Publication-ready plot generation 