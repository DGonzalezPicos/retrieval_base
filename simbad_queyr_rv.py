"""Query radial velocity from SIMBAD for a list of objects."""

import numpy as np
import pandas as pd
import astropy.units as u
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord

def query_rv(targets):
    """Query radial velocity from SIMBAD for a list of objects.
    
    Parameters
    ----------
    targets : list
        List of strings with object names.
        
    Returns
    -------
    dict
        Dictionary with object names as keys and radial velocities as values.
    """
    
    customSimbad = Simbad()
    # customSimbad.add_votable_fields('rv_value')
    # get spectra type, parallax, mass, radius, distance, radial velocity
    customSimbad.add_votable_fields('sptype', 'plx', 'distance', 'rv_value')
    
    # rv_dict = {}
    result = {}
    for target in targets:
        print(f' Querying radial velocity for {target}...')
        try:
            result_table = customSimbad.query_object(target)
            print(result_table.columns)
            rv = result_table['RV_VALUE'][0]
            spt = result_table['SP_TYPE'][0]
            distance = result_table['Distance_distance'][0]
            
            result[target] = [rv, spt, distance]
            # rv_dict[target] = rv
        except Exception as e:
            print(f' -> Error querying {target}: {e}')
            # rv_dict[target] = np.nan
            result[target] = [np.nan, np.nan, np.nan]
    return result

targets_rv = {
    'gl15A': 11.73,
    'gl15B': 11.17,
    'gl205': 8.5,
    'gl338B': 12.43,
    'gl382': 7.87,
    'gl408': 3.29,
    'gl411': -84.64,
    'gl412A': 68.8,
    'gl436': 9.59,
    'gl445': -111.51,
    'gl447': -30.66,
    'gl687': -28.65,
    'gl699': -110.11,
    'gl725A': -0.58,
    'gl725B': 1.19,
    'gl752A': 35.884,
    'gl849': -15.3,
    'gl876': -1.47,
    'gl880': -27.5,
    'gl905': -77.51,
    'gl1002': -33.7,
    'gl1151': -35.12,
    'gl1286': -41.0, # WARNING: SIMBAD has wrong RV (Davison+2015; RV = -40 km/s)
    'gl3622': 2.18,
    'gl4063': 12.533,
    
    'gl48': 0.0,
    'gl317': 0.0,
    'gl410': 0.0,
    'gl480': 0.0, 
    'gl514': 0.0,
    'gl617B': 0.0,
    'gl846': 0.0,   
    'gl4333': 0.0,
    }
targets = list(targets_rv.keys())
result = query_rv(targets)
rv_dict = {k:v[0] for k,v in result.items()}

# compare rvs from SIMBAD with the ones in the dictionary
for target in targets:
    rv_simbad = rv_dict[target]
    print(f' {target}: {targets_rv[target]} vs. {rv_simbad}')
    # print distance in pc
    # print(f' {target}: {targets_rv[target]} vs. {rv_simbad}')