""" Compare spectra with PHOENIX templates for NIR excess (veiling)"""
from retrieval_base.retrieval import pre_processing, Retrieval
from retrieval_base.parameters import Parameters
from retrieval_base.auxiliary_functions import get_PHOENIX_model
from retrieval_base.chemistry import Chemistry
from retrieval_base.spectrum import ModelSpectrum
atomic_mass = {k:v[2] for k,v in Chemistry.species_info.items()}

import numpy as np
import matplotlib.pyplot as plt
# set fontsize to 16
# plt.rcParams.update({'font.size': 16})
plt.style.use('/home/dario/phd/retrieval_base/HBDs/my_science.mplstyle')

import pathlib
import pickle
import json
# import nnls
from scipy.optimize import nnls

path = pathlib.Path('/home/dario/phd/retrieval_base')
targets = dict(J1200='freechem_15', 
               TWA28='freechem_12', 
               J0856='freechem_13'
               )
colors = dict(
            # J1200='royalblue', 
            J1200='limegreen',
            #   TWA28='seagreen',
            TWA28='limegreen',
            # J0856='indianred',
            J0856='limegreen',
            )

target = 'TWA28'
retrieval_id = targets[target]

data_path = pathlib.Path('/home/dario/phd/retrieval_base') / f'{target}'
print(data_path)
lw = 1.


# bestfit_params = 
retrieval_path = data_path / f'retrieval_outputs/{retrieval_id}'
assert retrieval_path.exists(), f'Retrieval path {retrieval_path} does not exist.'
# m_spec = np.load(retrieval_path / 'test_data/bestfit_m_spec_K1266.pkl')
m_spec = pickle.load(open(retrieval_path / 'test_data/bestfit_m_spec_K2166.pkl', 'rb'))
d_spec = pickle.load(open(retrieval_path / 'test_data/d_spec_K2166.pkl', 'rb'))
cov = pickle.load(open(retrieval_path / 'test_data/bestfit_Cov_K2166.pkl', 'rb'))


# err = np.array([np.sqrt(np.diag(cov[i,j].get_dense_cov())) for i,j in np.ndindex(cov.shape)])
# normalize data and model
d_spec.normalize_flux_per_order()
m_spec.normalize_flux_per_order()



transm = np.load(retrieval_path / 'test_data/d_spec_transm_K2166.npy')
loglike = pickle.load(open(retrieval_path / 'test_data/bestfit_LogLike_K2166.pkl', 'rb'))
with open(retrieval_path / 'test_data/bestfit.json', 'r') as f:
    bestfit_params = json.load(f)
            
params = bestfit_params['params']
RV = params['rv']

order, det = 4, 2
err = np.nan * np.ones_like(d_spec.flux[order,det])
mask = np.isfinite(d_spec.flux[order,det])
cov_ij = cov[order,det]
err[mask] = np.sqrt(np.diag(cov_ij.get_dense_cov())) / d_spec.norm[order,det]
err *= loglike.beta[order,det,None]


Teff = 2700 if target == 'J1200' else 2400
log_g = 4.0

# get PHOENIX model
ph_wave, ph_flux = get_PHOENIX_model(Teff, log_g, FeH=0, wave_range=(1900,2500), PHOENIX_path='./data/PHOENIX/')
ph_spec = ModelSpectrum(ph_wave, ph_flux)
ph_spec.shift_broaden_rebin(params['rv'], params['vsini'], epsilon_limb=params['epsilon_limb'], out_res=d_spec.resolution,
                            in_res=1e6, d_wave=d_spec.wave[order,det], rebin=True)
ph_spec.flux /= np.nanmedian(ph_spec.flux)

# create design matrix to check for veiling continuum
m_veiling = np.ones_like(m_spec.flux[order,det])
M = np.vstack([ph_spec.flux[mask], m_veiling[mask]])

d = d_spec.flux[order,det, mask]
phi = nnls(M @ cov_ij.solve(M.T), M @ cov_ij.solve(d))[0]
m_ph_veiling = phi @ M

print(f' Spectrum amplitude = {phi[0]:.2e}')
print(f' Veiling amplitude = {phi[-1]:.2e}')
# ph_spec.normalize_flux_per_order()
    
fig, ax = plt.subplots(2,1, figsize=(16,6), sharex=True, gridspec_kw={'height_ratios': [3,1]})
# ax[0].plot(d_spec.wave[order,det], d_spec.flux[order,det], label='Data', alpha=0.7,
#         color='k', lw=lw)
ax[0].fill_between(d_spec.wave[order,det], d_spec.flux[order,det] - err, d_spec.flux[order,det] + err, 
                   alpha=0.2, color='k', lw=0)




res_m_spec = m_spec.flux[order,det] - d_spec.flux[order,det]
res_ph_spec = ph_spec.flux / np.nanmedian(ph_spec.flux) - d_spec.flux[order,det]
res_ph_veiling = m_ph_veiling - d

chi2_m = np.nansum(res_m_spec**2 / err**2) / np.sum(mask)
chi2_ph = np.nansum(res_ph_spec**2 / err**2) / np.sum(mask)
chi2_ph_veiling = np.nansum(res_ph_veiling**2 / err[mask]**2) / np.sum(mask)
ax[0].plot(d_spec.wave[order,det], m_spec.flux[order,det], label='Best-fit model '+r'($\chi$'+f'={chi2_m:.2f})',
           color=colors[target], lw=lw)
ax[0].plot(ph_spec.wave, ph_spec.flux, 
           label='PHOENIX '+r'($\chi$'+f'={chi2_ph:.2f})', color='b', lw=lw)

ax[0].plot(ph_spec.wave[mask], m_ph_veiling, label=f'{phi[0]:.2f} * PHOENIX + {phi[1]:.2f} * veiling ' + r'($\chi$'+f'={chi2_ph_veiling:.2f})',
           color='r', lw=lw)

ax[0].legend(ncol=4, loc=(0.0, 1.01))
ax[0].set(ylabel='Normalized flux')

ax[1].plot(d_spec.wave[order,det], res_m_spec, label='Best-fit model', color=colors[target], lw=lw)
ax[1].plot(d_spec.wave[order,det], res_ph_spec, label='PHOENIX', color='b', lw=lw)
ax[1].plot(d_spec.wave[order,det,mask], res_ph_veiling, label='PHOENIX + veiling', color='r', lw=lw)
ax[1].fill_between(d_spec.wave[order,det], -err, err, alpha=0.2, color='k', lw=0)
ax[1].axhline(0, color='k', linestyle='-')
ax[1].set(ylabel='Residuals', xlabel='Wavelength / nm', xlim=(d_spec.wave[order,det].min(), d_spec.wave[order,det].max()))
# ax[1].legend()
fig.savefig(retrieval_path / f'test_plots/veiling_order{order}_det{det}.pdf')
print(f' Saved plot to {retrieval_path / f"test_plots/veiling_order{order}_det{det}.pdf"}')
plt.show()
