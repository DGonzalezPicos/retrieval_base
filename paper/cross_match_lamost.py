import pandas as pd
import numpy as np
# Read data from catalog
# df = pd.read_csv('paper/data/DR10_Cycle-SN_M_dwarfs.csv')
# query targets from simbad
from astropy.coordinates import SkyCoord, FK4
my_targets = ['Gl 699', 'Gl 15A', 'Gl 15B']

my_targets = {
    'Gl 699' : "17:57:48.5 +04:41:36.1",
}

# define astropy skycoord

for target in my_targets:
    source = SkyCoord(my_targets[target], frame=FK4)
    print(source)

