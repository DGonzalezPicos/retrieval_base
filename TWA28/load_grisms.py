import pathlib
from retrieval_base.spectrum_jwst import SpectrumJWST


path = pathlib.Path('TWA28/jwst/')
gratings = ['g140h-f100lp', 'g235h-f170lp', 'g395h-f290lp']
files = [path/f'TWA28_{g}.fits' for g in gratings]

for i, f in enumerate(files):
    spec = SpectrumJWST(file=f, grating=gratings[i])
    spec.split_grating( 
                     keep='both',
                     fig_name=None)
    
    