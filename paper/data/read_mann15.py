import pathlib
import numpy as np
import pandas as pd

path = pathlib.Path('/home/dario/phd/retrieval_base/paper/data/')
fits_file = path / 'mann15.fits' 
from astropy.io import fits
with fits.open(fits_file) as hdul:
    # print(hdul.info())
    # hdr = hdul[0].header
    # print(hdr.columns)
    data = hdul[1].data
    
# create pandas dataframe with columns of interest
# cols = ['Name', 'Teff', 'e_Teff'

names = data['CNS3'] # Gl or GJ names
# replace GJ with Gl
names = [n.replace('GJ', 'Gl') for n in names]
empty = np.array([len(n) == 0 for n in names])
# remove empty names
names = np.array(names)[~empty]
# remove spaces from names
names = [n.replace(' ', '') for n in names]

teff = data['Teff'][~empty]
spt = data['SpT'][~empty]
feh = data['[Fe/H]'][~empty]
feh_e = data['e_[Fe/H]'][~empty]
# save as txt file with three columns: name, value, error
np.savetxt(f'paper/data/mann15_feh.txt', np.array([names, feh, feh_e]).T, fmt='%s')