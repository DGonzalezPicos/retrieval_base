""" Test the local covariance kernel """

import numpy as np
import matplotlib.pyplot as plt

import os
import matplotlib.pyplot as plt
# pdf pages

from matplotlib.backends.backend_pdf import PdfPages
import copy
import pathlib
from retrieval_base.retrieval import Retrieval
import retrieval_base.auxiliary_functions as af
from retrieval_base.config import Config
from retrieval_base.local_covariance_kernel import LocalCovarianceKernel

path = af.get_path(return_pathlib=True)
path_figures = pathlib.Path('/home/dario/phd/twa2x_paper/figures')
config_file = 'config_jwst.txt'
target = 'TWA28'
# run = None
w_set='NIRSpec'

runs = dict(
    # TWA27A='lbl11_G1G2G3_fastchem_0',
    TWA28='lbl11_G1G2G3_fastchem_0',
            )


def load_data(target, run):
    cwd = os.getcwd()
    if target not in cwd:
        os.chdir(f'{path}/{target}')
        print(f'Changed directory to {target}')

    conf = Config(path=path, target=target, run=run)(config_file)        
        
    m_spec = af.pickle_load(f'{conf.prefix}data/bestfit_m_spec_NIRSpec.pkl')
    d_spec = af.pickle_load(f'{conf.prefix}data/d_spec_NIRSpec.pkl')

    m_spec.flux = m_spec.flux.squeeze()
    
    m_spec.wave = d_spec.wave 
    m_spec.flux_bb = m_spec.blackbody_disk(**m_spec.blackbody_disk_args).squeeze()
    d_spec.squeeze()
    return d_spec, m_spec

d_spec, m_spec = load_data(target, runs[target])
order = 11

wave = d_spec.wave[order]
flux = d_spec.flux[order]
err  = d_spec.err[order]
m_flux = m_spec.flux[order]
res = flux-m_flux
chi2_pp = res**2 / err**2

mean_chi2_pp = np.nanmean(chi2_pp)
std_chi2_pp = np.nanstd(chi2_pp)

lck = LocalCovarianceKernel(wave, flux, err, lck_width=10)
lck.s = lck(m_flux, n_max_regions=6)

lck_regions = lck.regions
s_lck = lck.s


fig, ax = plt.subplots(3,1, figsize=(14,5), sharex=True, gridspec_kw={'height_ratios':[3,2,2]})
ax[0].plot(wave, flux, label='data', color='k')
ax[0].plot(wave, m_flux, label='model', color='darkorange')
ax[0].legend()
ax[1].plot(wave, res, color='k')
# ax[1].fill_between(wave, -err, err, alpha=0.5)
for sigma in [1,2]:
    # ax[0].fill_between(wave, flux-sigma*s_global_new*err, flux+sigma*s_global_new*err, alpha=0.1, color='k', lw=0)
    # ax[1].fill_between(wave, -sigma*s_rest*err, sigma*s_rest*err, alpha=0.1, color='k', lw=0)
    ax[0].fill_between(wave, flux-sigma*s_lck*err, flux+sigma*s_lck*err, alpha=0.1, color='k', lw=0)
    ax[1].fill_between(wave, -sigma*s_lck*err, sigma*s_lck*err, alpha=0.3, color='darkorange', lw=0)

# ylim = np.quantile(s_lck*err, [0.01, 0.99])
ylim = np.array([np.nanmin(flux)*0.9, np.nanmax(flux)*1.1])
ax[0].set_ylim(*ylim)
ylim = np.nanmax(np.abs(res))*1.5
ax[1].set_ylim(-ylim, ylim)

ax[2].plot(wave, chi2_pp, color='brown')
ax[2].axhline(5 * std_chi2_pp + mean_chi2_pp, color='red', label=r'$10\sigma_{\chi^2_{pp}}$', ls='--')
for lck_region in lck_regions:
    ax[2].axvspan(lck_region[0], lck_region[1], color='red', alpha=0.2)
# add labels
ax[0].set_ylabel('Flux')
ax[1].set_ylabel('Residuals')
ax[2].set_ylabel(r'$\chi^2_{pp}$')
ax[2].set_xlabel('Wavelength')


plt.show()