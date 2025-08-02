"""Validate the generated zenodo data files for integrity and basic statistics"""

import numpy as np
import pathlib
from typing import Dict, List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
path = pathlib.Path('/home/dario/phd/retrieval_base')
zenodo_data_path = path / 'zenodo_data'

# Expected files
expected_files = [
    'TWA27A_G140H.dat', 'TWA27A_G235H.dat', 'TWA27A_G395H.dat',
    'TWA28_G140H.dat', 'TWA28_G235H.dat', 'TWA28_G395H.dat'
]

# Grating wavelength ranges for validation
gratings = dict(
    G140H=(900, 1900),
    G235H=(1650, 3180),
    G395H=(2890, 5290),
)


def validate_file(filepath: pathlib.Path) -> Dict:
    """
    Validate a single data file.
    
    Parameters
    ----------
    filepath : pathlib.Path
        Path to the data file
        
    Returns
    -------
    dict
        Validation results
    """
    logger.info(f"Validating {filepath.name}")
    
    try:
        # Load data
        data = np.loadtxt(filepath)
        
        # Check basic structure
        if data.ndim != 2 or data.shape[1] != 5:
            return {
                'valid': False,
                'error': f"Invalid data shape: {data.shape}"
            }
        
        # Extract columns
        wavelength = data[:, 0]
        flux = data[:, 1]
        flux_err = data[:, 2]
        model = data[:, 3]
        blackbody = data[:, 4]
        
        # Extract grating from filename
        grating = filepath.stem.split('_')[1]
        wmin, wmax = gratings[grating]
        
        # Validation checks
        checks = {
            'file_exists': True,
            'valid_shape': data.shape[1] == 5,
            'no_nan_values': not np.any(np.isnan(data)),
            'wavelength_sorted': np.all(np.diff(wavelength) >= 0),
            'wavelength_in_range': np.all((wavelength >= wmin) & (wavelength <= wmax)),
            'positive_flux': np.all(flux >= 0),
            'positive_errors': np.all(flux_err >= 0),
            'positive_model': np.all(model >= 0),
            'positive_blackbody': np.all(blackbody >= 0),
        }
        
        # Calculate statistics
        stats = {
            'n_points': len(wavelength),
            'wavelength_min': np.min(wavelength),
            'wavelength_max': np.max(wavelength),
            'flux_mean': np.mean(flux),
            'flux_std': np.std(flux),
            'error_mean': np.mean(flux_err),
            'model_mean': np.mean(model),
            'blackbody_mean': np.mean(blackbody),
        }
        
        return {
            'valid': all(checks.values()),
            'checks': checks,
            'stats': stats,
            'error': None
        }
        
    except Exception as e:
        return {
            'valid': False,
            'error': str(e),
            'checks': {},
            'stats': {}
        }


def main():
    """Main validation function."""
    logger.info("Starting validation of zenodo data files")
    
    if not zenodo_data_path.exists():
        logger.error(f"Zenodo data directory not found: {zenodo_data_path}")
        return
    
    # Validate all expected files
    results = {}
    all_valid = True
    
    for filename in expected_files:
        filepath = zenodo_data_path / filename
        results[filename] = validate_file(filepath)
        
        if not results[filename]['valid']:
            all_valid = False
            logger.error(f"Validation failed for {filename}: {results[filename]['error']}")
        else:
            logger.info(f"Validation passed for {filename}")
    
    # Print summary
    print("\n" + "="*80)
    print("ZENODO DATA VALIDATION SUMMARY")
    print("="*80)
    
    if all_valid:
        print("✅ ALL FILES VALIDATED SUCCESSFULLY")
    else:
        print("❌ SOME FILES FAILED VALIDATION")
    
    print(f"\nValidation Results:")
    for filename, result in results.items():
        status = "✅ PASS" if result['valid'] else "❌ FAIL"
        print(f"  {filename}: {status}")
        
        if result['valid']:
            stats = result['stats']
            print(f"    - Data points: {stats['n_points']:,}")
            print(f"    - Wavelength range: {stats['wavelength_min']:.1f} - {stats['wavelength_max']:.1f} nm")
            print(f"    - Mean flux: {stats['flux_mean']:.2e} erg/s/cm²/nm")
            print(f"    - Mean error: {stats['error_mean']:.2e} erg/s/cm²/nm")
        else:
            print(f"    - Error: {result['error']}")
    
    # Check for missing files
    existing_files = [f.name for f in zenodo_data_path.glob("*.dat")]
    missing_files = [f for f in expected_files if f not in existing_files]
    
    if missing_files:
        print(f"\n❌ Missing files:")
        for f in missing_files:
            print(f"  - {f}")
    else:
        print(f"\n✅ All expected files present")
    
    # Check for unexpected files
    unexpected_files = [f for f in existing_files if f not in expected_files]
    if unexpected_files:
        print(f"\n⚠️  Unexpected files found:")
        for f in unexpected_files:
            print(f"  - {f}")
    
    print("="*80)
    
    return all_valid


if __name__ == "__main__":
    main() 