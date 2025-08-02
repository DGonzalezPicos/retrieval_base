"""Save spectral data to zenodo_data/ directory with separate files for each target and grating
containing arrays of wavelength, flux, error, best fit model, and blackbody model"""

import numpy as np
import os
import pathlib
from typing import Dict, Tuple, List
import logging

from retrieval_base.retrieval import Retrieval
import retrieval_base.auxiliary_functions as af
from retrieval_base.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
path = af.get_path(return_pathlib=True)
config_file = 'config_jwst.txt'
w_set = 'NIRSpec'

# Define targets and their corresponding runs
runs = dict(
    TWA27A='freeslab_lbl10_G1G2G3_1',
    TWA28='freeslab_lbl10_G1G2G3_1',
)

# Define NIRSpec gratings and their wavelength ranges
gratings = dict(
    G140H=(900, 1900),
    G235H=(1650, 3180),
    G395H=(2890, 5290),
)
n_orders = 6

# Create zenodo_data directory
zenodo_data_path = path / 'zenodo_data'
zenodo_data_path.mkdir(exist_ok=True)
logger.info(f"Created zenodo data directory: {zenodo_data_path}")


def load_data(target: str, run: str) -> Tuple[object, object]:
    """
    Load spectral data for a given target and run.
    
    Parameters
    ----------
    target : str
        Target name (e.g., 'TWA28')
    run : str
        Run identifier
        
    Returns
    -------
    tuple
        (d_spec, m_spec) - data and model spectrum objects
    """
    cwd = os.getcwd()
    if target not in cwd:
        os.chdir(f'{path}/{target}')
        logger.info(f'Changed directory to {target}')

    conf = Config(path=path, target=target, run=run)(config_file)        
    
    # Load spectral data
    m_spec = af.pickle_load(f'{conf.prefix}data/bestfit_m_spec_NIRSpec.pkl')
    d_spec = af.pickle_load(f'{conf.prefix}data/d_spec_NIRSpec.pkl')
    
    # Load covariance and calculate errors
    cov = af.pickle_load(f'{conf.prefix}data/bestfit_Cov_NIRSpec.pkl')
    err = np.array([cov[cov_i,0].get_err(mask=d_spec.mask_isfinite[cov_i]) 
                   for cov_i in range(len(cov))])
    d_spec.err = err
    logger.info(f'Loaded data for {target}: err shape = {d_spec.err.shape}')

    # Process model spectrum
    m_spec.flux = m_spec.flux.squeeze()
    m_spec.wave = d_spec.wave 
    m_spec.flux_bb = m_spec.blackbody_disk(**m_spec.blackbody_disk_args).squeeze()
    d_spec.squeeze()
    
    return d_spec, m_spec


def get_grating_mask(wavelength: np.ndarray, grating: str) -> np.ndarray:
    """
    Create a mask for data within a specific grating's wavelength range.
    
    Parameters
    ----------
    wavelength : np.ndarray
        Wavelength array
    grating : str
        Grating name (G140H, G235H, G395H)
        
    Returns
    -------
    np.ndarray
        Boolean mask for wavelengths within the grating range
    """
    if grating not in gratings:
        raise ValueError(f"Unknown grating: {grating}. Available: {list(gratings.keys())}")
    
    wmin, wmax = gratings[grating]
    return (wavelength >= wmin) & (wavelength <= wmax)


def save_grating_data(target: str, d_spec: object, m_spec: object, output_path: pathlib.Path) -> None:
    """
    Save spectral data for a specific target and grating to a text file.
    
    Parameters
    ----------
    target : str
        Target name
    d_spec : object
        Data spectrum object
    m_spec : object
        Model spectrum object
    output_path : pathlib.Path
        Output directory path
    """
    
    
    gratings = ['G140H', 'G235H', 'G395H']
    # max_wave_grating = {'G140H': 1.887558e3, 'G235H': 3.170146e3, 'G395H': 5300}
    bad_indices = {'G140H': (3800, 3929),
                   'G235H': (3799, 3838),
                   'G395H': None}
    for i in range(3):
        logger.info(f"Processing {target} {gratings[i]}")
        grating_name = gratings[i]
        filename = f"{target}_{grating_name}.dat"
        filepath = output_path / filename
        # Get wavelength and apply grating mask
            
        # Extract data for this grating
        wave_grating = d_spec.wave[i*6:(i+1)*6].flatten()
        flux_data = d_spec.flux[i*6:(i+1)*6].flatten()
        flux_err = d_spec.err[i*6:(i+1)*6].flatten()
        model_flux = m_spec.flux[i*6:(i+1)*6].flatten()
        bb_flux = m_spec.flux_bb[i*6:(i+1)*6].flatten() * d_spec.flux_unit_factor
        
        wave_nans = np.isnan(wave_grating)
        wave_grating = wave_grating[~wave_nans]
        flux_data = flux_data[~wave_nans]
        flux_err = flux_err[~wave_nans]
        model_flux = model_flux[~wave_nans]
        bb_flux = bb_flux[~wave_nans]
        
        # find where wavelength diff changes sign and ignore all pixels after that
        try:
            wave_diff = np.diff(wave_grating)
            wave_diff_sign = np.sign(wave_diff)
            wave_diff_sign_change = np.where(wave_diff_sign != wave_diff_sign[0])[0]
            wave_grating = wave_grating[:wave_diff_sign_change[0]]
            flux_data = flux_data[:wave_diff_sign_change[0]]
            flux_err = flux_err[:wave_diff_sign_change[0]]
            model_flux = model_flux[:wave_diff_sign_change[0]]
            bb_flux = bb_flux[:wave_diff_sign_change[0]]
        except IndexError:
            logger.warning(f"No sign change found for {target} {grating_name}")
            continue
        
        # sort by wavelength
        # sort_indices = np.argsort(wave_grating)
        # wave_grating = wave_grating[sort_indices]
        # flux_data = flux_data[sort_indices]
        # flux_err = flux_err[sort_indices]
        # model_flux = model_flux[sort_indices]
        # bb_flux = bb_flux[sort_indices]
        
        

        # Convert to arrays and sort by wavelength
        data_arrays = np.array([wave_grating, flux_data, flux_err, model_flux, bb_flux])
        
        # Save to file
        header = f"# Spectral data for {target} {grating_name}\n"
        header += "# Wavelength (nm), Flux (erg/s/cm2/nm), Flux Error (erg/s/cm2/nm), "
        header += "Best Fit Model (erg/s/cm2/nm), Blackbody Model (erg/s/cm2/nm)\n"
        header += "# Data extracted from retrieval outputs\n"
        
        np.savetxt(filepath, data_arrays.T, header=header, fmt='%.6e', delimiter='\t')
        logger.info(f"Saved {len(data_arrays[0])} data points to {filepath}")


def main():
    """Main function to load data and save to zenodo_data directory."""
    logger.info("Starting data extraction for Zenodo")
    
    # Load data for all targets
    d_specs, m_specs = {}, {}
    for target in runs.keys():
        logger.info(f"Loading data for {target}")
        d_specs[target], m_specs[target] = load_data(target, runs[target])
    
    # Save data for each target and grating
    for target in runs.keys():
        logger.info(f"Processing data for {target}")
        d_spec = d_specs[target]
        m_spec = m_specs[target]
        
        
        save_grating_data(target, d_spec, m_spec, zenodo_data_path)
    
    logger.info("Data extraction completed successfully")
    
    # Print summary
    print("\n" + "="*60)
    print("ZENODO DATA EXTRACTION SUMMARY")
    print("="*60)
    print(f"Output directory: {zenodo_data_path}")
    print(f"Targets processed: {list(runs.keys())}")
    print(f"Gratings: {list(gratings.keys())}")
    print("\nExpected files:")
    for target in runs.keys():
        for grating in gratings.keys():
            filename = f"{target}_{grating}.dat"
            print(f"  - {filename}")
    print("="*60)


if __name__ == "__main__":
    main() 