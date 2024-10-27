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
    customSimbad.add_votable_fields('rv_value')
    
    rv_dict = {}
    for target in targets:
        print(f' Querying radial velocity for {target}...')
        try:
            result_table = customSimbad.query_object(target)
            rv = result_table['RV_VALUE'][0]
            rv_dict[target] = rv
        except Exception as e:
            print(f' -> Error querying {target}: {e}')
            rv_dict[target] = np.nan
    return rv_dict

targets_rv = {
                'gl338B': 12.0,
                'gl382' : 8.0,
                'gl408' : 3.0,
                'gl411' :-85.0,
                'gl436' : -40.0,
                'gl699' : -111.0,
                'gl752A': 36.0,
                'gl832': 36.0,
                'gl905' : -78.0,
                'gl1286': 8.0,
                'gl15A': 12.0,
                'gl15B': 11.0,
                'gl687': -29.0,
                'gl725A': -31.0,
                'gl725B': 1.0,
                'gl849': -15.0,
                'gl876': -2.0,
                'gl880': -27.0,
                'gl1151': -35.0,
                'gl205': 8.5,
                'gl412A': 9.0,
                'gl445': 9.0,
                'gl447': -31.0,
                'gl1002': -40.0,
                'gl412A': 69.0,
                'gl1286': -41.0,
                'gl3622': 2.0,
                'gl4063': 12.0,
                
}
targets = list(targets_rv.keys())
rv_dict = query_rv(targets)