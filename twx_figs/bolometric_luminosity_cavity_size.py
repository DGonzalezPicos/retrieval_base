""""
Estimate the bolometric luminosity and cavity size of the blackbody disk following
eq 1 in CUgno+2024:
L_bol = 4*pi*R_disk**2*sigma*T_eff**4 (1)
R_cavity = sqrt(L_bol/(4*pi*sigma*T_eff**4)) (2)

"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Tuple

# stefan boltzmann constant in cgs units
sigma = 5.6703744191843561646e-5 # erg s^-1 cm^-2 K^-4
rjup_to_cm = 7.1492e10 # cm

# convert bolometric luminosity to Lsun
l_sun = 3.828e33 # erg s^-1

# Error propagation formulas used:
# For a function f(x,y) = x^n * y^m, the relative error is:
# (df/f)^2 = (n*dx/x)^2 + (m*dy/y)^2
# 
# For bolometric luminosity: L = 4*pi*R^2*sigma*T^4
# (dL/L)^2 = (2*dR/R)^2 + (4*dT/T)^2
#
# For cavity size: R = sqrt(L/(16*pi*sigma*T^4))
# (dR/R)^2 = (1/2 * dL/L)^2 + (2*dT/T)^2

def _propagate_relative_errors(*args: Tuple[float, float]) -> float:
    """
    Propagate relative errors using quadrature addition.
    
    Parameters:
    -----------
    *args : Tuple[float, float]
        Variable number of (value, error) pairs
        
    Returns:
    --------
    float
        Combined relative error
    """
    total_relative_error_sq = 0.0
    for value, error in args:
        if error is not None and value != 0:
            total_relative_error_sq += (error / value)**2
    return np.sqrt(total_relative_error_sq)

def bolometric_luminosity(teff: float, radius_rjup: float,
                          teff_err: float = None, 
                          radius_err: float = None) -> Union[Tuple[float, float], float]:
    """
    Estimate the bolometric luminosity of the blackbody disk following
    eq 1 in CUgno+2024:
    L_bol = 4*pi*R_disk**2*sigma*T_eff**4 (1)
    
    Parameters:
    -----------
    teff : float
        Effective temperature in K
    radius_rjup : float
        Disk radius in Jupiter radii
    teff_err : float, optional
        Uncertainty in effective temperature in K
    radius_err : float, optional
        Uncertainty in disk radius in Jupiter radii
        
    Returns:
    --------
    Union[Tuple[float, float], float]
        If errors provided: (luminosity, luminosity_error) in erg/s
        If no errors: luminosity in erg/s
        
    Raises:
    -------
    ValueError
        If teff <= 0 or radius_rjup <= 0
        If any error values are negative
    """
    # Input validation
    if teff <= 0:
        raise ValueError("Effective temperature must be positive")
    if radius_rjup <= 0:
        raise ValueError("Disk radius must be positive")
    
    if teff_err is not None and teff_err < 0:
        raise ValueError("Temperature error must be non-negative")
    if radius_err is not None and radius_err < 0:
        raise ValueError("Radius error must be non-negative")
    
    radius_cm = radius_rjup * rjup_to_cm
    l_bol = 4 * np.pi * radius_cm**2 * sigma * teff**4
    
    # If no uncertainties provided, return just the value
    if teff_err is None and radius_err is None:
        return l_bol
    
    # Error propagation for L = 4*pi*R^2*sigma*T^4
    # Using the formula: (dL/L)^2 = (2*dR/R)^2 + (4*dT/T)^2
    # For power laws: if y = x^n, then dy/y = n*dx/x
    relative_errors = [
        (2, radius_rjup, radius_err),  # R^2 term: 2*dR/R
        (4, teff, teff_err)           # T^4 term: 4*dT/T
    ]
    
    total_relative_error = 0.0
    for power, value, error in relative_errors:
        if error is not None and value != 0:
            total_relative_error += (power * error / value)**2
    
    l_bol_err = l_bol * np.sqrt(total_relative_error)
    
    return l_bol, l_bol_err

def cavity_size(teff_bb: float, l_bol: float,
                teff_bb_err: float = None, 
                l_bol_err: float = None) -> Union[Tuple[float, float], float]:
    """
    Estimate the cavity size of the blackbody disk following
    eq 2 in CUgno+2024:
    R_cavity = sqrt(L_bol/(4*pi*sigma*T_eff_BB**4)) (2)
    
    Parameters:
    -----------
    teff_bb : float
        Blackbody effective temperature in K
    l_bol : float
        Bolometric luminosity in erg/s
    teff_bb_err : float, optional
        Uncertainty in blackbody effective temperature in K
    l_bol_err : float, optional
        Uncertainty in bolometric luminosity in erg/s
        
    Returns:
    --------
    Union[Tuple[float, float], float]
        If errors provided: (cavity_radius, cavity_radius_error) in R_jup
        If no errors: cavity_radius in R_jup
        
    Raises:
    -------
    ValueError
        If teff_bb <= 0 or l_bol <= 0
        If any error values are negative
    """
    # Input validation
    if teff_bb <= 0:
        raise ValueError("Blackbody effective temperature must be positive")
    if l_bol <= 0:
        raise ValueError("Bolometric luminosity must be positive")
    
    if teff_bb_err is not None and teff_bb_err < 0:
        raise ValueError("Temperature error must be non-negative")
    if l_bol_err is not None and l_bol_err < 0:
        raise ValueError("Luminosity error must be non-negative")
    
    r_cavity = np.sqrt(l_bol / (16 * np.pi * sigma * teff_bb**4)) / rjup_to_cm
    
    # If no uncertainties provided, return just the value
    if teff_bb_err is None and l_bol_err is None:
        return r_cavity
    
    # Error propagation for R = sqrt(L/(16*pi*sigma*T^4))
    # Using the formula: (dR/R)^2 = (1/2 * dL/L)^2 + (2 * dT/T)^2
    # For power laws: if y = x^n, then dy/y = n*dx/x
    relative_errors = [
        (0.5, l_bol, l_bol_err),  # L term: 1/2 * dL/L
        (2, teff_bb, teff_bb_err) # T term: 2 * dT/T
    ]
    
    total_relative_error = 0.0
    for power, value, error in relative_errors:
        if error is not None and value != 0:
            total_relative_error += (power * error / value)**2
    
    r_cavity_err = r_cavity * np.sqrt(total_relative_error)
    
    return r_cavity, r_cavity_err



if __name__ == '__main__':
    # Example without uncertainties
    
    targets = {
        'TWA27A': {'teff': (2450, 31), 'radius': (3.10, 0.01),
                   'teff_bb': (643, 4), 'radius_bb': (11.9, 0.15)},
        'TWA28': {'teff': (2382, 42), 'radius': (3.08, 0.01),
                   'teff_bb': (653, 2), 'radius_bb': (13.9, 0.08)},
    }
    
    # target = 'TWA27A'
    for target in targets.keys():
        print(f'========== {target} ==========')
        teff, teff_err = targets[target]['teff']
        radius, radius_err = targets[target]['radius']
        teff_bb, teff_bb_err = targets[target]['teff_bb']
        radius_bb, radius_bb_err = targets[target]['radius_bb']
        
        l_bol = bolometric_luminosity(teff, radius)
        r_cavity = cavity_size(teff_bb, l_bol)
        
        print(f'L_bol = {l_bol:.2e} erg/s, {l_bol/l_sun:.2f} Lsun')
        print(f'R_cavity = {r_cavity:.2f} Rjup')
        
        # Example with uncertainties
        teff_err = 100.0  # ±100 K
        radius_err = 0.5   # ±0.5 Rjup
        teff_bb_err = 50.0 # ±50 K
        
        l_bol_with_err, l_bol_err = bolometric_luminosity(teff, radius, teff_err, radius_err)
        r_cavity_with_err, r_cavity_err = cavity_size(teff_bb, l_bol_with_err, teff_bb_err, l_bol_err)
        
        print(f'\nWith uncertainties:')
        print(f'L_bol = ({l_bol_with_err:.2e} ± {l_bol_err:.2e}) erg/s')
        print(f'L_bol = ({l_bol_with_err/l_sun:.2f} ± {l_bol_err/l_sun:.2f}) Lsun')
        print(f'R_cavity = ({r_cavity_with_err:.2f} ± {r_cavity_err:.2f}) Rjup')
        
        # Example with partial uncertainties (only temperature error)
        l_bol_partial, l_bol_partial_err = bolometric_luminosity(teff, radius, teff_err, None)
        print(f'\nPartial uncertainties (only T_eff error):')
        print(f'L_bol = ({l_bol_partial:.2e} ± {l_bol_partial_err:.2e}) erg/s')
        print('')
        