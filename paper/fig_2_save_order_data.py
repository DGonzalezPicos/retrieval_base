#!/usr/bin/env python3
"""
Script to save order data to text files for plotting.
Saves wavelength, data, best fit model, model without 13CO, and model without C18O for each order.
"""

import numpy as np
import pathlib
from retrieval_base.retrieval import Retrieval
from retrieval_base.config import Config

def save_order_data(target: str, base_path: str = '/home/dario/phd/retrieval_base/') -> None:
    """
    Save order data to text files for each order.
    
    Parameters:
    -----------
    target : str
        Target name (e.g., 'gl205')
    base_path : str
        Base path to the retrieval base directory
    """
    # Change to target directory
    target_path = pathlib.Path(base_path) / target
    if not target_path.exists():
        raise FileNotFoundError(f"Target directory {target_path} not found")
    
    os.chdir(target_path)
    
    # Find outputs directory
    outputs = pathlib.Path(base_path) / target / 'retrieval_outputs'
    dirs = [d for d in outputs.iterdir() if d.is_dir() and 'fc' in d.name and '_' not in d.name]
    runs = [int(d.name.split('fc')[-1]) for d in dirs]
    
    if len(runs) == 0:
        raise FileNotFoundError(f"No runs found in {outputs}")
    
    # Use the latest run
    run = 'fc' + str(max(runs))
    test_output = outputs / run / 'test_output'
    
    if not test_output.exists():
        raise FileNotFoundError(f"No test_output folder found in {test_output}")
    
    # Load the saved data
    bestfit_spec_file = test_output / 'bestfit_spec.npy'
    bestfit_spec_file_no13CO = test_output / 'bestfit_spec_no13CO.npy'
    bestfit_spec_file_noC18O = test_output / 'bestfit_spec_noC18O.npy'
    
    if not all([bestfit_spec_file.exists(), bestfit_spec_file_no13CO.exists(), bestfit_spec_file_noC18O.exists()]):
        raise FileNotFoundError("One or more bestfit model files not found. Run the main script first.")
    
    # Load data
    wave, flux, err, mask, m, spline_cont = np.load(bestfit_spec_file)
    _, _, _, _, m_no13CO, spline_cont_no13CO = np.load(bestfit_spec_file_no13CO)
    _, _, _, _, m_noC18O, spline_cont_noC18O = np.load(bestfit_spec_file_noC18O)
    
    # Define orders and their wavelength ranges
    orders = [0, 1, 2]
    cenwaves = [2347.4, 2376.5, 2467.5]
    dwave = 3.02
    xlim_list = [(cenwave-dwave, cenwave+dwave) for cenwave in cenwaves]
    
    # Create output directory for text files
    output_dir = test_output / 'order_data'
    output_dir.mkdir(exist_ok=True)
    
    # Save data for each order
    for i, order in enumerate(orders):
        # Get wavelength range for this order
        xlim = xlim_list[i]
        
        # Create mask for this wavelength range
        wave_mask = (wave[order] >= xlim[0]) & (wave[order] <= xlim[1])
        
        # Extract data for this order and wavelength range
        wave_order = wave[order][wave_mask]
        flux_order = flux[order][wave_mask]
        m_order = m[order][wave_mask]
        m_no13CO_order = m_no13CO[order][wave_mask]
        m_noC18O_order = m_noC18O[order][wave_mask]
        
        # Save to text file
        output_file = output_dir / f'order_{order}_data.txt'
        header = f"# Order {order} data for {target}\n"
        header += "# Wavelength (nm) | Data | Best_fit_model | Model_no_13CO | Model_no_C18O\n"
        
        data_to_save = np.column_stack([
            wave_order,
            flux_order,
            m_order,
            m_no13CO_order,
            m_noC18O_order
        ])
        
        np.savetxt(output_file, data_to_save, header=header, fmt='%.6f')
        print(f"Saved order {order} data to {output_file}")
        print(f"  Wavelength range: {xlim[0]:.1f} - {xlim[1]:.1f} nm")
        print(f"  Number of points: {len(wave_order)}")

if __name__ == '__main__':
    import os
    
    # Example usage
    target = 'gl205'  # Change this to your target
    try:
        save_order_data(target)
        print(f"Successfully saved order data for {target}")
    except Exception as e:
        print(f"Error saving order data: {e}")
