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
spitzer_files = dict(TWA28='spitzer/1102-3430.txt',
                    TWA27A='spitzer/1207-3932.txt',
)
target = 'TWA27A'
# w_set='NIRSpec'
# run = 'lbl11_G2G3_fastchem_GP_0'
run = 'freeslab_lbl10_G2G3_0'

cwd = os.getcwd()
if target not in cwd:
    nwd = os.path.join(cwd, target)
    print(f'Changing directory to {nwd}')
    os.chdir(nwd)
    
conf = Config(path=path, target=target, run=run)(config_file)
conf.cov_mode = 'None'
conf.lck_kwargs = {}
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
        # flux_medfilt = apply_medfilt(flux, kernel_size=sigma_width)
        flux_medfilt = medfilt(flux, kernel_size=sigma_width)
        mask_clip = np.abs(flux - flux_medfilt) > sigma_clip*err
        # flux[mask_clip] = np.nan
        wave = wave[~mask_clip]
        flux = flux[~mask_clip]
        err = err[~mask_clip]
        
    
        
    return wave, flux, err

swave, sflux, serr = load_spitzer(spitzer_files[target], wmax=20.0)

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
    run_spitzer = 'spitzer_G2G3'
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
        ignore_line_species = ['K_static', 'Na_Sam', 'FeH_main_iso_Sam',
                               'H2S_Sid_main_iso']
        ignore_line_species += [l for l in conf.line_species if l.endswith('_high')]
        pRT_atm = pRT_model(
            line_species=[l for l in conf.line_species if l not in ignore_line_species],
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
            disk_kwargs=getattr(conf, 'disk_kwargs', {}),
            T_ex_range=getattr(conf, 'T_ex_range', None),
            N_mol_range=getattr(conf, 'N_mol_range', None),
            T_cutoff=conf_data.get('T_cutoff', None),
            P_cutoff=conf_data.get('P_cutoff', None),
            species_wave=getattr(conf, 'species_wave', {}),
            )
        # check parent directory
        # pRT_file.parent.mkdir(parents=True, exist_ok=True)
        af.pickle_save(pRT_file, pRT_atm)
        print(f'   --> Saved {pRT_file}')
        
        
def apply_medfilt(x, y, width=10):
    """ apply median filter to y, with robust nan handling """
    from scipy.signal import medfilt
    y_medfilt = medfilt(y, kernel_size=width)
    mask = np.isnan(y)
    y_medfilt[mask] = np.nan
    return y_medfilt
        
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
# save npy with spitzer_data_model.npy, create array with wave, flux, err, bb, model_flux, model_flux_atm
spitzer_data_model = np.array([d_spec.wave.squeeze(), d_spec.flux.squeeze(), d_spec.err.squeeze(),
                               bb, ret.m_spec[w_set].flux.squeeze()])
np.save(f'{conf.prefix}data/spitzer_model.npy', spitzer_data_model)
print(f'--> Saved {conf.prefix}data/spitzer_model.npy')

# flux scaling factor
flux_unit_factor = conf.config_data['NIRSpec']['flux_unit_factor']

print(f' lnL = {lnL}')
colors = plt.cm.viridis(np.linspace(0, 1, d_spec.n_orders))
plot = True
medfilt_width = 101
if plot:
    fig, ax = plt.subplots(2, 1, figsize=(12, 5), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
    lw = 1.4

    def gaussian_filter_nan(x, y, sigma=5.0):
        """Apply Gaussian filter to y, handling NaN values through interpolation"""
        from scipy.ndimage import gaussian_filter1d
        
        # Create mask of valid values
        mask = np.isfinite(y)
        
        if np.all(mask):
            # No NaNs - simple case
            return gaussian_filter1d(y, sigma=sigma)
        
        # Interpolate NaNs using valid neighbors
        y_interp = np.interp(x, x[mask], y[mask])
        
        # Apply Gaussian filter to interpolated data
        y_filtered = gaussian_filter1d(y_interp, sigma=sigma)
        
        # Restore original NaNs in filtered data
        y_filtered[~mask] = np.nan
        
        return y_filtered

    for i in range(d_spec.n_orders):
        
        beta = ret.LogLike[w_set].beta[i,0]
        # ax[0].fill_between(spec.wave[i,0], spec.flux[i,0]-beta*err[i,0],
        #                 spec.flux[i,0]+beta*err[i,0], 
        #                 color=colors[i//2], alpha=0.3)
        
        wave_i = d_spec.wave[i,0]

        if i < d_spec.n_orders-1:
            d_flux_i = d_spec.flux[i,0] / flux_unit_factor
            m_flux = ret.m_spec[w_set].flux[i,0,:] / flux_unit_factor
            bb_i = bb[i] / 1.0
            m_flux_atm = (m_flux - bb_i)
            
            res_i = d_flux_i / m_flux
            
            # Apply robust Gaussian filtering
            d_flux_i = gaussian_filter_nan(wave_i, d_flux_i, sigma=20.0)
            m_flux = gaussian_filter_nan(wave_i, m_flux, sigma=20.0)
            m_flux_atm = gaussian_filter_nan(wave_i, m_flux_atm, sigma=20.0)
            # res_i = gaussian_filter_nan(wave_i, res_i, sigma=20.0)
            
        else:
            d_flux_i = d_spec.flux[i,0] / 1.0
            m_flux = gaussian_filter_nan(wave_i, ret.m_spec[w_set].flux[i,0,:] / flux_unit_factor, sigma=20.0)
            bb_i = bb[i] / 1.0
            m_flux_atm = (m_flux - bb_i)
            res_i = d_flux_i / m_flux
            
        mask = (wave_i > 5240.0) & (wave_i < 5300.0)
        if np.sum(mask) > 0:
            d_flux_i[mask] = np.nan
            m_flux[mask] = np.nan
            m_flux_atm[mask] = np.nan
            
        labels = [""] * d_spec.n_orders
        if i == 0:
            labels = ['Observations', 'Full model', 'Atmosphere', 'Blackbody']
            
        
        ax[0].plot(wave_i, d_flux_i, color='k', label=labels[0], lw=lw, marker='o', markersize=2)

        ax[0].plot(wave_i, m_flux,
                color='mediumseagreen', lw=lw, ls='-', label=labels[1])
        ax[0].plot(wave_i, m_flux_atm, color='brown', lw=lw, ls='-', label=labels[2])
        ax[0].plot(wave_i, bb[i], color='navy', lw=lw, ls='-', label=labels[3])
        
        # res = d_flux_i / m_flux
        ax[1].plot(wave_i, res_i, color='k', lw=lw, marker='o', markersize=2)
    
    # ax[1].axhline(0, color='k', lw=0.5, ls='-')
        
    xlim = (np.nanmin(d_spec.wave), np.nanmax(d_spec.wave))
    xpad = 0.01 * (xlim[1] - xlim[0])
    # ax[0].set_xlim(xlim[0]-xpad, xlim[1]+xpad)
    ax[0].set_xlim(xlim[0], 13e3)
    ax[0].set_ylim(1e-17, None)
    
    ax[0].set(ylabel=r'Flux / erg s$^{-1}$ cm$^{-2}$ nm$^{-1}$', yscale='log')
    ax[0].legend(frameon=False)
    ax[-1].set(xlabel='Wavelength / nm', yscale='log', ylabel='Residuals (Data / Model)')
    ax[-1].axhline(1.0, color='mediumseagreen', lw=1.0, ls='-')
    ax[-1].set_ylim(0.8, 1.2)
    yticks = [0.8, 1.0, 1.2]
    # first remove existing minor and major yticks
    ax[-1].set_yticks([])
    ax[-1].set_yticks(yticks, minor=True)
    ax[-1].set_yticks(yticks)
    ax[-1].set_yticklabels([f'{y:.1f}' for y in yticks])
    
    # Draw instrument labels above plot with Unicode arrows
    nirspec = (xlim[0], 5.3e3)  # Convert to nm
    spitzer = (5.3e3, 13e3)
    
    # Remove old arrow annotations
    # Instead use Unicode arrows in text
    arrow_symbol = '←→'  # Unicode left-right arrow
    
    # Text style
    text_props = dict(
        ha='center',
        fontsize=12,
        weight='bold',
        bbox=dict(
            facecolor='white',
            edgecolor='none',
            alpha=0.8,
            pad=2
        )
    )
    
    # Create a new axes above the plot for the labels
    label_ax = fig.add_axes([0.125, 0.9, 0.775, 0.05])  # [left, bottom, width, height]
    label_ax.set_xticks([])
    label_ax.set_yticks([])
    label_ax.spines['top'].set_visible(False)
    label_ax.spines['right'].set_visible(False)
    label_ax.spines['bottom'].set_visible(False)
    label_ax.spines['left'].set_visible(False)
    
    # Calculate normalized positions for text
    nirspec_center = np.mean(nirspec)
    spitzer_center = np.mean(spitzer)
    
    # Convert wavelength positions to axis coordinates
    total_range = xlim[1] - xlim[0]
    nirspec_pos = (nirspec_center - xlim[0]) / total_range
    spitzer_pos = (spitzer_center - xlim[0]) / total_range
    
    # Add text with arrows
    label_ax.text(nirspec_pos, 0.5, f'JWST/NIRSpec\n{arrow_symbol}', 
                 va='center', **text_props)
    label_ax.text(spitzer_pos, 0.5, f'Spitzer/IRS\n{arrow_symbol}', 
                 va='center', **text_props)
    
    # Set the x-limits of the label axis to match the main plot
    label_ax.set_xlim(0, 1)
    
    plt.show()
    
    plots_dir = pathlib.Path(f'{conf.prefix}plots')
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig_name = str(plots_dir / 'bestfit_spitzer')
    exts = ['.pdf']
    for ext in exts:
        fig.savefig(fig_name + ext, transparent=(ext == '.png'), dpi=300, bbox_inches='tight')
        print(f' --> Saved {fig_name}{ext}')
    plt.close(fig)