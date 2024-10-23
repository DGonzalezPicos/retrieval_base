import pathlib
import numpy as np
import os
import matplotlib.pyplot as plt
# pdf pages
from matplotlib.backends.backend_pdf import PdfPages
import copy
from scipy.ndimage import gaussian_filter1d
# import medfilt scipy
from scipy.signal import medfilt
import petitRADTRANS.nat_cst as nc

from retrieval_base.retrieval import Retrieval
from retrieval_base.spectrum import ModelSpectrum
import retrieval_base.auxiliary_functions as af
from retrieval_base.config import Config
from retrieval_base.pRT_model import pRT_model
# import config_jwst as conf

path = pathlib.Path(af.get_path())
config_file = 'config_jwst.txt'
target = 'TWA28'
# w_set='NIRSpec'
run = 'lbl15_G2G3_3'

cwd = os.getcwd()
if target not in cwd:
    nwd = os.path.join(cwd, target)
    print(f'Changing directory to {nwd}')
    os.chdir(nwd)
    
conf = Config(path=path, target=target, run=run)(config_file)
conf_data = conf.config_data['NIRSpec']
d_spec = af.pickle_load(conf.prefix + 'data/d_spec_NIRSpec.pkl')


def load_spitzer(file, sigma_clip=3.0, sigma_width=5, wmax=20.0, n_edge=1):
    
    wave, flux, err, flag = np.loadtxt(file, skiprows=0).T
    mask = flag > 0
    print(f' Number of flagged points: {np.sum(mask)}')
    if n_edge > 0:
        wave = wave[n_edge:-n_edge]
        flux = flux[n_edge:-n_edge]
        err = err[n_edge:-n_edge]
        
    # mask points beyond 33 micron
    mask_wave = wave > wmax
    # flux[mask_wave] = np.nan
    wave = wave[~mask_wave]
    flux = flux[~mask_wave]
    err = err[~mask_wave]
    
    
    ## Unit conversion ##
    wave_cm = wave * 1e-4  # [microns] -> [cm]
    # convert Jy to [erg cm^{-2} s^{-1} Hz^{-1}]
    flux *= 1e-23
    # convert [erg cm^{-2} s^{-1} Hz^{-1}] -> [erg cm^{-2} s^{-1} cm^{-1}]
    flux *= 2.998e10 / wave_cm**2 # wave in cm
    # Convert [erg cm^{-2} s^{-1} cm^{-1}] -> [erg cm^{-2} s^{-1} nm^{-1}]
    flux *= 1e-7
    err = err * (1e-23) * (2.998e10 / wave_cm**2) * 1e-7
    
    # clip 3 sigma outliers
    if sigma_clip > 0:
        flux_medfilt = medfilt(flux, kernel_size=sigma_width)
        mask_clip = np.abs(flux - flux_medfilt) > sigma_clip*err
        # flux[mask_clip] = np.nan
        wave = wave[~mask_clip]
        flux = flux[~mask_clip]
        err = err[~mask_clip]
        
    
        
    return wave, flux, err

swave, sflux, serr = load_spitzer('spitzer/1102-3430.txt', wmax=20.0)

# add them to d_spec
n_pix = d_spec.wave.shape[-1]
swave_pad = np.pad(swave, (0, n_pix-len(swave)), 'constant', constant_values=np.nan) * 1e3
sflux_pad = np.pad(sflux, (0, n_pix-len(sflux)), 'constant', constant_values=np.nan)
serr_pad = np.pad(serr, (0, n_pix-len(serr)), 'constant', constant_values=np.nan)

d_spec.wave = np.vstack([np.squeeze(d_spec.wave), swave_pad])[:,None,:]
d_spec.flux = np.vstack([np.squeeze(d_spec.flux), sflux_pad])[:,None,:]
d_spec.err = np.vstack([np.squeeze(d_spec.err), serr_pad])[:,None,:]

load_pRT = True
cache = True
if load_pRT:
    
    ## Create pRT_atm object
    run_spitzer = 'spitzer'
    w_set = 'NIRSpec'
    prefix = f'./retrieval_outputs/{run_spitzer}/test_data'
    pathlib.Path(prefix).mkdir(parents=True, exist_ok=True)
    rv_range = (-60.0, 60.0)

    pRT_file =pathlib.Path(f'{prefix}/pRT_atm_{w_set}.pkl')
    d_spec_file = pathlib.Path(f'{prefix}/d_spec_{w_set}.pkl')
        
    if not d_spec_file.exists():
        af.pickle_save(d_spec_file, d_spec)
    # else:
        # d_spec = 
    if not pRT_file.exists() or not cache:
        print(f'--> Creating {pRT_file}')
        lbl = 200
        pRT_atm = pRT_model(
            line_species=conf.line_species, 
            # line_species=['H2O_pokazatel_main_iso', 'CO_high_Sam'],
            d_spec=d_spec, 
            mode='lbl' if (lbl is not None) else 'c-k',
            lbl_opacity_sampling=lbl,
            rayleigh_species=['H2', 'He'], 
            continuum_opacities=['H2-H2', 'H2-He'], 
            log_P_range=conf_data.get('log_P_range'), 
            n_atm_layers=conf_data.get('n_atm_layers'), 
            rv_range=rv_range,
            disk_species=getattr(conf, 'disk_species', []),
            T_ex_range=getattr(conf, 'T_ex_range', None),
            N_mol_range=getattr(conf, 'N_mol_range', None),
            T_cutoff=conf_data.get('T_cutoff', None),
            P_cutoff=conf_data.get('P_cutoff', None),
            )
        # check parent directory
        # pRT_file.parent.mkdir(parents=True, exist_ok=True)
        af.pickle_save(pRT_file, pRT_atm)
        print(f'   --> Saved {pRT_file}')
        
        
        
# evaluate the model


ret = Retrieval(
    conf=conf, 
    evaluation=False,
    # d_spec={w_set: d_spec},
)
bestfit_params, posterior = ret.PMN_analyze()



conf.prefix = f'./retrieval_outputs/{run_spitzer}/test_'

ret = Retrieval(
    conf=conf, 
    evaluation=False,
    # d_spec={w_set: d_spec},
)
# define new retrieval for spitzer
# ret.d_spec[w_set] = d_spec
# ret.pRT_atm = {w_set: af.pickle_load(pRT_file)}
ret.Param.params['gratings'] += ['spitzer']


ret.evaluate_model(bestfit_params)
lnL = ret.PMN_lnL_func()


R_jup = nc.r_jup_mean
wave_cm = d_spec.wave*1e-7
bb = np.squeeze(af.blackbody(wave_cm, ret.Param.params['T_d']) * (ret.Param.params['R_d']*R_jup / (ret.Param.params['d_pc'] * nc.pc))**2)


print(f' lnL = {lnL}')
colors = plt.cm.viridis(np.linspace(0, 1, d_spec.n_orders))
plot = True
if plot:
    fig, ax = plt.subplots(2, 1, figsize=(12, 5), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
    lw = 0.7

    for i in range(d_spec.n_orders):
        
        beta = ret.LogLike[w_set].beta[i,0]
        # ax[0].fill_between(spec.wave[i,0], spec.flux[i,0]-beta*err[i,0],
        #                 spec.flux[i,0]+beta*err[i,0], 
        #                 color=colors[i//2], alpha=0.3)
        if i < d_spec.n_orders-1:
            d_flux_i = gaussian_filter1d(d_spec.flux[i,0], sigma=5.0)
            m_flux = gaussian_filter1d(ret.LogLike[w_set].m_flux[i,0], sigma=5.0)
            m_flux_atm = gaussian_filter1d(ret.m_spec[w_set].flux[0,i,0,:] - bb[i], sigma=5.0)
            
        else:
            d_flux_i = d_spec.flux[i,0]
            m_flux = ret.LogLike[w_set].m_flux[i,0]
            m_flux_atm = ret.m_spec[w_set].flux[0,i,0,:] - bb[i]
            
            
        mask = (d_spec.wave[i,0] > 5240.0) & (d_spec.wave[i,0] < 5300.0)
        if np.sum(mask) > 0:
            d_flux_i[mask] = np.nan
            m_flux[mask] = np.nan
            m_flux_atm[mask] = np.nan
            
        labels = [""] * d_spec.n_orders
        if i == 0:
            labels = ['Data', 'Atm. + BB', 'Atm.', 'BB']
            
            
        ax[0].plot(d_spec.wave[i,0], d_flux_i, color='k', label=labels[0], lw=lw)

        ax[0].plot(d_spec.wave[i,0], m_flux,
                color='limegreen', lw=lw, ls='-', label=labels[1])
        ax[0].plot(d_spec.wave[i,0], m_flux_atm, color='red', lw=lw, ls='-', label=labels[2])
        ax[0].plot(d_spec.wave[i,0], bb[i], color='blue', lw=lw, ls='-', label=labels[3])
        
        res = d_flux_i / m_flux
        ax[1].plot(d_spec.wave[i,0], res, color=colors[i//2], lw=lw)
    
    # ax[1].axhline(0, color='k', lw=0.5, ls='-')
        
    xlim = (np.nanmin(d_spec.wave), np.nanmax(d_spec.wave))
    xpad = 0.01 * (xlim[1] - xlim[0])
    ax[0].set_xlim(xlim[0]-xpad, xlim[1]+xpad)
    ax[0].set(ylabel=r'Flux / erg s$^{-1}$ cm$^{-2}$ nm$^{-1}$', yscale='log')
    ax[0].legend(frameon=False)
    ax[-1].set(xlabel='Wavelength / nm', yscale='log', ylabel='Residuals (Data / Model)')
    ax[-1].axhline(1.0, color='k', lw=0.5, ls='-')
    ax[-1].set_ylim(0.5, 2.0)
    yticks = [0.5, 1.0, 1.5, 2.0]
    # first remove existing minor and major yticks
    ax[-1].set_yticks([])
    ax[-1].set_yticks(yticks, minor=True)
    ax[-1].set_yticks(yticks)
    ax[-1].set_yticklabels([f'{y:.1f}' for y in yticks])
    plt.show()
    
    fig_name = f'{conf.prefix}plots/bestfit_spitzer'
    exts = ['.pdf', '.png']
    for ext in exts:
        fig.savefig(fig_name + ext, transparent=(ext == '.png'), dpi=300, bbox_inches='tight')
        print(f' --> Saved {fig_name}{ext}')
    plt.close(fig)