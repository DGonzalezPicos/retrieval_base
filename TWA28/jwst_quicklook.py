import numpy as np 
import matplotlib.pyplot as plt

import pathlib
import pickle
import json

from astropy.io import fits


path = pathlib.Path('TWA28/jwst')
files = list(path.glob('*.fits'))
gratings = ['f100lp', 'f170lp', 'f290lp']

def sigma_clip(array, sigma=3, max_iter=5, return_mask=False):
    """Sigma clip an array by setting to NaN values that are more than sigma"""
    clipped = array.copy()
    for i in range(max_iter):
        mean = np.nanmean(clipped)
        std = np.nanstd(clipped)
        mask = np.abs(clipped - mean) < sigma * std
        clipped[~mask] = np.nan
        
    if return_mask:
        return clipped, mask
    return clipped

def read_data(file):
    
    with fits.open(file) as hdul:
        data = hdul[1].data
        wave, flux, err = data['WAVELENGTH'], data['FLUX'], data['ERR']
    
    return wave, flux, err

targets = dict(J1200='freechem_10', 
               TWA28='freechem_6', 
               J0856='freechem_9'
               )
target = 'TWA28'
retrieval_id = targets[target]
order = 2
flux_factor = 1.6 * 1e12
crires_color = 'green'
jwst_color = 'brown'
lw = 0.9

data_path = pathlib.Path('/home/dario/phd/retrieval_base') / f'{target}'
print(data_path)

fig, ax = plt.subplots(2, 1, figsize=(14, 5), sharex=False,
                       gridspec_kw={'height_ratios': [3, 1], 'wspace': 0.05, 'hspace': 0.30})
zoom_in = (3, 1)

# bestfit_params = 
retrieval_path = data_path / f'retrieval_outputs/{retrieval_id}'
assert retrieval_path.exists(), f'Retrieval path {retrieval_path} does not exist.'
# m_spec = np.load(retrieval_path / 'test_data/bestfit_m_spec_K1266.pkl')
m_spec = pickle.load(open(retrieval_path / 'test_data/bestfit_m_spec_K2166.pkl', 'rb'))
d_spec = pickle.load(open(retrieval_path / 'test_data/d_spec_K2166.pkl', 'rb'))
transm = np.load(retrieval_path / 'test_data/d_spec_transm_K2166.npy')
loglike = pickle.load(open(retrieval_path / 'test_data/bestfit_LogLike_K2166.pkl', 'rb'))

    # load json file with bestfit parameters
with open(retrieval_path / 'test_data/bestfit.json', 'r') as f:
    bestfit_params = json.load(f)

params = bestfit_params['params']
# RV = params['rv']
RV = 0 # no RV shift
# print(params.keys())
# RV = bestfit_params['params']['RV']
for order in range(5):
    Cov = pickle.load(open(retrieval_path / 'test_data/bestfit_Cov_K2166.pkl', 'rb'))[order]

    det_err = []
    for det in range(3):
        x = d_spec.wave[order,det]

        if RV != 0:
            # shift to rest-frame
            x *= (1-RV/2.998e5)
        # x_shift = x * (1-RV/2.998e5)

        y = d_spec.flux[order,det] * flux_factor
        # print(f'Min flux = {np.nanmin(y):.3e} erg s-1cm-2nm-1')
        # print(f'Max flux = {np.nanmax(y):.3e} erg s-1cm-2nm-1')


        cov = Cov[det].get_dense_cov()
        err = np.sqrt(np.diag(cov)) * loglike.beta[order,det,None] * flux_factor
        err_full = np.nan * np.ones_like(x)

        nans = np.isnan(y)
        # err_full = np.where(nans, np.nan, err_full)
        err_full[~nans] = err
        # err = d_spec.err[order,det] * loglike.beta[order,det,None] * flux_factor
        median_err = np.nanmedian(err)
        # scatter median error to show uncertainty
        det_err.append(median_err)
        # print(f'Median error = {median_err:.3e} erg s-1cm-2nm-1')

        m = m_spec.flux[order,det] * loglike.f[order,det,None] * flux_factor
        ax[0].plot(x* 1e-3, y, color='k', lw=lw)
        ax[0].fill_between(x* 1e-3, y-err_full, y+err_full, color='k', alpha=0.2, lw=0)
        label = 'CRIRES+' if (order+det) == 0 else None
        ax[0].plot(x * 1e-3, m, color=crires_color, label=label, lw=lw)
        if (order == zoom_in[0]) and (det==zoom_in[1]):
            ax[1].plot(x* 1e-3, y, color='k')
            ax[1].fill_between(x* 1e-3, y-err_full, y+err_full, color='k', alpha=0.2)
            ax[1].plot(x * 1e-3, m, color=crires_color, label=label)
            zoom_in_xlim = (x.min()* 1e-3, x.max()* 1e-3)
            zoom_in_ylim = (0.98 * np.nanmin(y), 1.02 * np.nanmax(y))
            ax[0].axvspan(zoom_in_xlim[0], zoom_in_xlim[1], alpha=0.1, color='grey')

# fig, ax = plt.subplots(1, 1, figsize=(8, 4))
jwst_wave = np.array([])
for i, f in enumerate(files):
    wave, flux, err = read_data(f)
    _, mask = sigma_clip(err, sigma=3, max_iter=5, return_mask=True)
    flux[~mask] = np.nan
    
    label = 'JWST/NIRSpec' if i == 0 else None
    for axi in ax:
        axi.plot(wave, flux, color=jwst_color, label=label, lw=lw)
        axi.fill_between(wave, flux-err, flux+err, alpha=0.5, color=jwst_color, lw=0)
    jwst_wave = np.append(jwst_wave, wave)


ax[1].set_xlabel('Wavelength (micron)')
ax[0].set_ylabel('Flux (mJy)')
ax[1].set(ylabel='Flux (mJy)', xlim=zoom_in_xlim, ylim=zoom_in_ylim)
ax[0].set(xlim=(np.min(jwst_wave), np.max(jwst_wave)))

ax[0].set_title('TWA28')
ax[0].legend()
fig.savefig(path / 'TWA28_jwst.pdf')
plt.show()
