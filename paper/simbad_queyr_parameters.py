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
    
    rv_dict = {}
    for target in targets:
        print(f' Querying {target}...')
        try:
            result_table = customSimbad.query_object(target)
            
        except Exception as e:
            print(f' -> Error querying {target}: {e}')
            rv_dict[target] = np.nan
    return result_table

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
    'gl4063': 12.533
    }
targets = list(targets_rv.keys())
# result = query_rv(targets)

# compare rvs from SIMBAD with the ones in the dictionary
attrs = ['SP_TYPE', 'RV_VALUE', 'PLX_VALUE']

ref_dict = {}

info = {}
for target in targets:
    result = query_rv([target])
    spt = result['SP_TYPE'][0]
    spt_ref = result['SP_BIBCODE'][0]
    rv = result['RV_VALUE'][0]
    
    distance = result['Distance_distance'][0]
    distance_unit = result['Distance_unit'][0]
    distance_ref = result['Distance_bibcode'][0]
    print(f'{target}: {spt} ({spt_ref}), RV = {rv} km/s, Distance = {distance} {distance_unit} ({distance_ref})')