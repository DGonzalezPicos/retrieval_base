# Data set: Disentangling disc and atmospheric signatures of young brown dwarfs in JWST/NIRSpec
dwarfs in JWST/NIRSpec spectra

This directory contains spectral data extracted from retrieval outputs for TWA 27A and TWA 28, organized by NIRSpec grating.

## File Naming Convention

Files are named as `{TARGET}_{GRATING}.dat` where:
- `TARGET`: TWA27A or TWA28
- `GRATING`: G140H, G235H, or G395H

## Data Format

Each file contains tab-separated columns with the following data:

| Column | Description | Units |
|--------|-------------|-------|
| 1 | Wavelength | nm |
| 2 | Observed Flux | erg/s/cm²/nm |
| 3 | Flux Error | erg/s/cm²/nm |
| 4 | Best Fit Model | erg/s/cm²/nm | (atmosphere and blackbody combined)
| 5 | Blackbody Model | erg/s/cm²/nm |

## NIRSpec Grating Wavelength Ranges

- **G140H**: 900 - 1900 nm
- **G235H**: 1650 - 3180 nm  
- **G395H**: 2890 - 5290 nm


## Usage Example

```python
import numpy as np

# Load data for TWA28 G140H grating
data = np.loadtxt('TWA28_G140H.dat')
wavelength = data[:, 0]  # nm
flux = data[:, 1]        # erg/s/cm²/nm
flux_err = data[:, 2]    # erg/s/cm²/nm
model = data[:, 3]       # erg/s/cm²/nm
blackbody = data[:, 4]   # erg/s/cm²/nm
```

## Citation

If you use this data in your research, please cite:

```bibtex
@article{gonzalezpicos2025_disentangling,
  title={Disentangling disc and atmospheric signatures of young brown
dwarfs in JWST/NIRSpec spectra},
  author={González Picos, D. and de Regt, S. and Gandhi, S. and Grasser, N. and Snellen, I.},
  journal={Astronomy & Astrophysics},
  year={2025},
}
```