import pathlib
from retrieval_base.spectrum_jwst import SpectrumJWST


path = pathlib.Path('TWA28/jwst/')
grisms = ['g140h-f100lp', 'g235h-f170lp', 'g395h-f290lp']
files = [path/f'TWA28_{g}.fits' for g in grisms]

for i, f in enumerate(files):
    spec = SpectrumJWST(file=f, grism=grisms[i])
    spec.split_grism( 
                     keep='both',
                     fig_name=None)
    
    