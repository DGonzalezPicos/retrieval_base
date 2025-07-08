# Zenodo Dataset: Disentangling disc and atmospheric signatures of young brown dwarfs in JWST/NIRSpec spectra

**Authors:** D. González Picos, S. de Regt, S. Gandhi, N. Grasser, I.A.G. Snellen  
**Institution:** Leiden Observatory, Leiden University, P.O. Box 9513, 2300 RA, Leiden, The Netherlands  
**Contact:** picos@strw.leidenuniv.nl  
**Year:** 2025

## Description

This dataset contains the key data products from the atmospheric retrieval analysis of TWA 27A and TWA 28 brown dwarfs using JWST/NIRSpec observations. The data supports the results presented in our paper on disentangling disk and atmospheric signatures in young brown dwarf spectra.

## Data Products

### 1. Spectral Data (`*_spectral_data.h5`)
- **Observational data**: Wavelength, flux, and uncertainties organised by grating (flattened arrays)
- **Best-fit models**: Total model flux and blackbody disk component
- **Grating coverage**: G140H (0.97-1.90 μm), G235H (1.65-3.18 μm), G395H (2.89-5.27 μm)
- **Metadata**: Instrument configuration, flux units, and processing information

### 2. Pressure-Temperature Profiles (`*_pt_profiles.h5`)
- **Temperature envelopes**: Confidence intervals (68%, 95%, 99.7%) and median profiles
- **Pressure grid**: Atmospheric pressure levels
- **Contribution function**: Integrated emission contribution
- **Surface gravity**: Posterior samples for log g

### 3. Posterior Distributions (`*_posteriors.h5`)
- **Parameter samples**: MCMC chains for all free parameters
- **Statistics**: Median, standard deviation, and percentiles
- **Metadata**: Parameter descriptions and units

## File Structure

```
zenodo_twa/
├── README.md                     # This file
├── TWA27A_spectral_data.h5      # TWA 27A spectral data
├── TWA27A_pt_profiles.h5        # TWA 27A PT profiles
├── TWA27A_posteriors.h5         # TWA 27A posteriors
├── TWA28_spectral_data.h5       # TWA 28 spectral data
├── TWA28_pt_profiles.h5         # TWA 28 PT profiles
├── TWA28_posteriors.h5          # TWA 28 posteriors
├── examples/                     # Usage examples
│   ├── load_spectral_data.py
│   ├── load_pt_profiles.py
│   └── load_posteriors.py
└── figures/                      # Generated example figures
    ├── TWA27A_spectrum_g140h.png
    ├── TWA27A_spectrum_g235h.png
    ├── TWA27A_spectrum_g395h.png
    ├── TWA27A_spectrum_all_gratings.png
    ├── TWA27A_pt_profile.png
    ├── TWA27A_corner_plot.png
    ├── TWA28_spectrum_g140h.png
    ├── TWA28_spectrum_g235h.png
    ├── TWA28_spectrum_g395h.png
    ├── TWA28_spectrum_all_gratings.png
    ├── TWA28_pt_profile.png
    └── TWA28_corner_plot.png
```

## Usage

### Requirements
- Python 3.7+
- numpy
- h5py
- matplotlib
- json

### Quick Start

1. **Load spectral data:**
```python
import h5py
import numpy as np

with h5py.File('TWA28_spectral_data.h5', 'r') as f:
    # Load G235H grating data
    wavelength = f['g235h/observational_data/wavelength'][:]
    flux = f['g235h/observational_data/flux'][:]
    model_flux = f['g235h/model_data/flux_total'][:]
```

2. **Load PT profiles:**
```python
with h5py.File('TWA28_pt_profiles.h5', 'r') as f:
    pressure = f['pt_profiles/pressure'][:]
    temperature = f['pt_profiles/temperature_envelopes'][:]
    contribution = f['pt_profiles/integrated_contribution'][:]
```

3. **Load posteriors:**
```python
with h5py.File('TWA28_posteriors.h5', 'r') as f:
    param_keys = [k.decode('utf-8') for k in f['parameters/param_keys'][:]]
    log_g_samples = f['posterior_samples/log_g'][:]
```

### Examples

See the `examples/` directory for complete working examples:
- `load_spectral_data.py`: Load and plot observed vs. model spectra by grating with 3:1 height ratios
- `load_pt_profiles.py`: Load and plot pressure-temperature profiles with confidence envelopes
- `load_posteriors.py`: Load and analyze posterior distributions with corner plots

**Figure styling**: All example scripts use the same colour scheme and formatting as the published paper:
- TWA 28: Orange (#D55E00) for models, black for data
- TWA 27A: Green (#009E73) for models, grey for data
- Publication-ready formatting with 300 DPI output

## Data Format

All data files use HDF5 format with the following structure:

### Metadata
Each file contains a `metadata` attribute with:
- Publication information
- Target and run identifiers
- Data creation timestamp
- Instrument configuration
- Data-specific parameters

### Data Groups
- **g140h/**, **g235h/**, **g395h/**: Grating-specific spectral data
  - **observational_data/**: Raw observational data
  - **model_data/**: Best-fit model results
- **pt_profiles/**: Pressure-temperature information
- **posterior_samples/**: MCMC parameter samples
- **statistics/**: Summary statistics

### Units
- Wavelength: nm
- Flux: erg s⁻¹ cm⁻² nm⁻¹
- Pressure: bar
- Temperature: K
- Surface gravity: log₁₀(cm s⁻²)

## Targets

### TWA 27A
- **Spectral type:** M9
- **Distance:** ~55 pc
- **Age:** ~10 Myr
- **Gratings:** G140H, G235H, G395H
- **Wavelength range:** 0.97-5.27 μm

### TWA 28
- **Spectral type:** M8.5
- **Distance:** ~55 pc
- **Age:** ~10 Myr
- **Gratings:** G140H, G235H, G395H
- **Wavelength range:** 0.97-5.27 μm

## Retrieval Method

The atmospheric retrieval analysis used:
- **Code:** retrieval_base (custom Python package)
- **Sampler:** PyMultiNest
- **Atmospheric model:** petitRADTRANS
- **Chemistry:** Free chemistry with slab disk model
- **Wavelength coverage:** 0.97-5.27 μm across three gratings

## Citation

If you use this dataset, please cite our paper:

```bibtex
@article{gonzalez_picos_2024,
    title = {Disentangling disc and atmospheric signatures of young brown dwarfs in JWST/NIRSpec spectra},
    author = {D. González Picos and S. de Regt and S. Gandhi and N. Grasser and I.A.G. Snellen},
    journal = {Astronomy & Astrophysics},
    year = {2025},
    institution = {Leiden Observatory, Leiden University, P.O. Box 9513, 2300 RA, Leiden, The Netherlands}
}
```

## Contact

For questions about this dataset, please contact:
- **Primary contact:** picos@strw.leidenuniv.nl
- **Institution:** Leiden Observatory, Leiden University, P.O. Box 9513, 2300 RA, Leiden, The Netherlands

## License

This dataset is released under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
You are free to use, modify, and distribute this data with proper attribution.

## Acknowledgments

This work is based on observations made with the NASA/ESA/CSA James Webb Space Telescope, 
which is operated by the Space Telescope Science Institute (STScI) under NASA contract NAS5-03127.

We thank the JWST team for their excellent work in commissioning and operating the observatory.
