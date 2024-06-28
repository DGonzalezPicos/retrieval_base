""" Compare best-fit K-band model to M-band data """
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os

from retrieval_base.retrieval import pre_processing, prior_check, Retrieval
from retrieval_base.spectrum_jwst import SpectrumJWST
from retrieval_base.pRT_model import pRT_model
import retrieval_base.auxiliary_functions as af
from retrieval_base.parameters import Parameters
from retrieval_base.config import Config
import config_jwst as conf

path = pathlib.Path('/home/dario/phd/retrieval_base')
config_file = 'config_jwst.txt'
target = 'TWA28'
cwd = os.getcwd()
if target not in cwd:
    print(f'Changing directory to {target}')
    os.chdir(target)

run = 'jwst_K_N5'


gratings = [
            # 'g140h-f100lp', 
            'g235h-f170lp', 
            'g395h-f290lp',
            ]
colors= [
        # '#0096c3',
         '#269000',
         '#c30034',
]

plot = True
files = [f'jwst/TWA28_{g}.fits' for g in gratings]
spec = SpectrumJWST(Nedge=40).load_gratings(files)
spec.reshape(spec.n_orders, 1)
# spec.fix_wave_nans() # experimental...
spec.sigma_clip_reshaped(use_flux=False, 
                            sigma=3, 
                            width=31, 
                            max_iter=5,
                            fun='median', 
                            debug=False)
err = np.copy(spec.err)
spec.prepare_for_covariance()
# spec.err = err

# spec.plot_orders(fig_name=f'{conf.prefix}plots/spec_test.pdf')
conf = Config(path=path, target=target, run=run)
conf(config_file)  
w_set = 'NIRSpec'
  
conf_data = conf.config_data['NIRSpec']
conf.config_data[w_set]['wave_range'] = (1650, 5300)
conf.scale_flux = False
conf.N_knots = 1
conf.scale_flux_eps = 0.0

## Create pRT_atm object covering both gratings
pRT_file =pathlib.Path(f'{conf.prefix}data/pRT_atm_{spec.w_set}_G2G3.pkl')
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
    # pRT_file.parent.mkdir(parents=True, exist_ok=True)
    af.pickle_save(pRT_file, pRT_atm)
    print(f'   --> Saved {pRT_file}')    
    
ret = Retrieval(
    conf=conf, 
    evaluation=False,
    d_spec={f'{w_set}': spec},
    )

ret.Param.params['N_knots'] = 1

ret.d_spec[w_set] = spec
ret.pRT_atm[w_set] = af.pickle_load(pRT_file)
bestfit_params, posterior = ret.PMN_analyze()
ret.evaluate_model(bestfit_params)
lnL = ret.PMN_lnL_func()


print(f' lnL = {lnL}')
if plot:
    fig, ax = plt.subplots(2, 1, figsize=(12, 5), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
    lw = 0.7

    for i in range(spec.n_orders):
        ax[0].plot(spec.wave[i,0], spec.flux[i,0], color=colors[i//2], lw=lw)
        
        beta = ret.LogLike[w_set].beta[i,0]
        # ax[0].fill_between(spec.wave[i,0], spec.flux[i,0]-beta*err[i,0],
        #                 spec.flux[i,0]+beta*err[i,0], 
        #                 color=colors[i//2], alpha=0.3)
        
        ax[0].plot(spec.wave[i,0], ret.LogLike[w_set].m_flux[i,0], 
                color='magenta', lw=lw, ls='-')
        
        res = spec.flux[i,0] - ret.LogLike[w_set].m_flux[i,0]
        ax[1].plot(spec.wave[i,0], res, color=colors[i//2], lw=lw)
    
    ax[1].axhline(0, color='k', lw=0.5, ls='-')
        
    xlim = (np.nanmin(spec.wave), np.nanmax(spec.wave))
    xpad = 0.01 * (xlim[1] - xlim[0])
    ax[0].set_xlim(xlim[0]-xpad, xlim[1]+xpad)
    ax[0].set(ylabel='Flux / Jy')
    ax[-1].set(xlabel='Wavelength / nm')
    plt.show()