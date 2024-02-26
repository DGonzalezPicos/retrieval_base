import numpy as np
import matplotlib.pyplot as plt
import pathlib

from retrieval_base.retrieval import pre_processing, prior_check, Retrieval
from retrieval_base.spectrum_jwst import SpectrumJWST
from retrieval_base.pRT_model import pRT_model
import retrieval_base.auxiliary_functions as af
from retrieval_base.parameters import Parameters

import config_jwst as conf

target = 'TWA27b'
path = pathlib.Path('/home/dario/phd/retrieval_base/') / target.upper()


## Pre-processing data
spec = SpectrumJWST(file=path / f'jwst/{target}_g395h-f290lp.fits')
spec.split_grism(4155., keep=1)
# spec.sigma_clip(sigma=3, width=5, max_iter=5, fun='median')
spec.sigma_clip(spec.err, sigma=2.5, width=25, max_iter=5, fun='median',
                fig_name=path / f'{conf.prefix}plots/sigma_clip_{spec.w_set}.pdf')
# spec.reshape(1,1)

# spec.plot()
# plt.show()

