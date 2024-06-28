'''Quick check at forward modeling for JWST G395
date: 2024-02-24
'''
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pickle
import pathlib

import retrieval_base.figures as figs
import retrieval_base.auxiliary_functions as af
from retrieval_base.spectrum_jwst import SpectrumJWST
from retrieval_base.pRT_model import pRT_model
from retrieval_base.retrieval import Retrieval
# Create the Radtrans objects
import config_jwst as conf
conf_data = conf.config_data['G395H_F290LP']


base = pathlib.Path('/home/dario/phd/retrieval_base')
path = pathlib.Path('TWA28')
# spec = SpectrumJWST(file=path/'TWA28_g235h-f170lp.fits')
spec = SpectrumJWST(file=base/path/'jwst/TWA28_g395h-f290lp.fits')
spec.split_grating(4155., keep=1)
# spec.sigma_clip(sigma=3, width=5, max_iter=5, fun='median')
spec.sigma_clip(spec.err, sigma=3, width=50, max_iter=5, fun='median')
spec.reshape(1,1)
spec.prepare_for_covariance()

print(f'--> Saving {base / path / f"{conf.prefix}data/d_spec_{spec.w_set}.pkl"}')
af.pickle_save(base / path / f'{conf.prefix}data/d_spec_{spec.w_set}.pkl', spec)



pRT_file =base / path / f'{conf.prefix}data/pRT_atm_{spec.w_set}.pkl'

if not pRT_file.exists():
    print(f'--> Creating {pRT_file}')

    pRT_atm = pRT_model(
        line_species=conf.line_species, 
        d_spec=spec, 
        mode='lbl', 
        lbl_opacity_sampling=conf_data['lbl_opacity_sampling'], 
        cloud_species=conf.cloud_species, 
        # rayleigh_species=['H2', 'He'], 
        # continuum_opacities=['H2-H2', 'H2-He'], 
        rayleigh_species=conf.rayleigh_species,
        continuum_opacities=conf.continuum_opacities,
        log_P_range=conf_data.get('log_P_range'), 
        n_atm_layers=conf_data.get('n_atm_layers'), 
        rv_range=conf.free_params['rv'][0], 
        )
    # check parent directory
    pRT_file.parent.mkdir(parents=True, exist_ok=True)
    af.pickle_save(pRT_file, pRT_atm)
    
    
ret = Retrieval(conf=conf, evaluation=False)
order, det = 0,0 

fig = plt.figure(figsize=(16,8), layout='constrained')
gs0 = fig.add_gridspec(4,5, hspace=0.00, wspace=0.1)

ax = fig.add_subplot(gs0[:3,:3])
plt.setp(ax.get_xticklabels(), visible=False)
ax_res = fig.add_subplot(gs0[3,:3], sharex=ax)
ax_PT = fig.add_subplot(gs0[:4,3:])

for ix, i in enumerate([0.0, 0.5, 1.0]):
    ret.Param(i * np.ones(len(ret.Param.param_keys)))

        
    sample = {k:ret.Param.params[k] for k in ret.Param.param_keys}
    print(sample)

    ln_L = ret.PMN_lnL_func()
    print(f'ln_L = {ln_L:.4e}\n')
    x = ret.d_spec['G395H_F290LP'].wave[order,det]

    f = ret.LogLike['G395H_F290LP'].f[order,det]
    if ix == 0:
        mask = ret.d_spec['G395H_F290LP'].mask_isfinite[order,det]
        ax.plot(x, ret.d_spec['G395H_F290LP'].flux[order,det], lw=1.5, label='data', color='k')
        
    model = f * ret.m_spec['G395H_F290LP'].flux[order,det]
    ax.plot(x, model, lw=2.5, label=f'logL = {ln_L:.3e}', ls='--')

    res = ret.d_spec['G395H_F290LP'].flux[order,det] - model
    res[~mask] = np.nan
    ax_res.plot(x, res, lw=2.5)

    ax_PT.plot(ret.PT.temperature, ret.PT.pressure, lw=4.5)
        
ax_PT.set(yscale='log', ylim=(ret.PT.pressure.max(), ret.PT.pressure.min()))
ax_res.axhline(0, color='k', ls='-', alpha=0.9) 
ax.legend()
plt.show()