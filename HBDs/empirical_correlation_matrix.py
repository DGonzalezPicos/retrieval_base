from retrieval_base.retrieval import pre_processing, Retrieval
from retrieval_base.parameters import Parameters

import numpy as np
import matplotlib.pyplot as plt
# set fontsize to 16
# plt.rcParams.update({'font.size': 16})
plt.style.use('/home/dario/phd/retrieval_base/HBDs/my_science.mplstyle')

import pathlib
import pickle
import corner
import pandas as pd
import json

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    result /= result.max()
    return result[result.size // 2:]

path = pathlib.Path('/home/dario/phd/retrieval_base')
# out_path = path / 'HBDs'
out_path = pathlib.Path('/home/dario/phd/Hot_Brown_Dwarfs_Retrievals/figures/')

targets = dict(J1200='final_full',
                TWA28='final_full',
                J0856='final_full',
                )
colors = dict(J1200='royalblue', TWA28='seagreen', J0856='indianred')

prefix = 'freechem'
fig, ax = plt.subplots(6,3, figsize=(12,12))
lw= 2.0

for i, target in enumerate(targets.keys()):
    retrieval_id = targets[target]

    data_path = pathlib.Path('/home/dario/phd/retrieval_base') / f'{target}'
    print(data_path)


    # bestfit_params = 
    retrieval_path = data_path / f'retrieval_outputs/{retrieval_id}'
    assert retrieval_path.exists(), f'Retrieval path {retrieval_path} does not exist.'
    # m_spec = np.load(retrieval_path / 'test_data/bestfit_m_spec_K1266.pkl')
    m_spec = pickle.load(open(retrieval_path / 'test_data/bestfit_m_spec_K2166.pkl', 'rb'))
    d_spec = pickle.load(open(retrieval_path / 'test_data/d_spec_K2166.pkl', 'rb'))
    transm = np.load(retrieval_path / 'test_data/d_spec_transm_K2166.npy')
    loglike = pickle.load(open(retrieval_path / 'test_data/bestfit_LogLike_K2166.pkl', 'rb'))
    cov = pickle.load(open(retrieval_path / 'test_data/bestfit_Cov_K2166.pkl', 'rb'))

    # to estimate the sample covariance matrix we need to compute the residuals
    # order, det = 5, 0

    residuals = d_spec.flux - loglike.m
    
    for order in range(6):
        for det in range(3):
            r = residuals[order,det]

            n = np.sum(np.isfinite(r))
            if n==0:
                ax[order,det].set_visible(False)
                continue
            r[np.isnan(r)] = 0

            corr = autocorr(r)
            ax[order,det].plot(corr, lw=lw, color=colors[target], label=target)
            if (i+1) == len(targets):
                g_noise = np.random.randn(len(r))
                corr_gaussian = autocorr(g_noise)
                ax[order,det].plot(corr_gaussian, alpha=0.6, color='k', lw=lw, label='Gaussian noise')
                # plot the lengthscale of the correlation function
                ax[order,det].axhline(1/np.e, color='k', linestyle='--')
                
                
                # ax.set_xlabel('Pixels')
                ax[order,det].set(xlim=(-1, 20))
                ax[order,det].text(s=r'$\lambda$'+f'{d_spec.wave[order,det].mean():.0f} nm', x=19, y=0.3+1/np.e, ha='right', va='bottom')
ax[0,0].legend(loc=(1.10, -0.02))
ax[0,0].text(s='1/e', x=19, y=1/np.e, ha='right', va='bottom')
fig.text(0.5, 0.070, 'Lag / pixel', ha='center')
fig.text(0.082, 0.5, 'Autocorrelation', va='center', rotation='vertical')
# plt.show()
fig.savefig(out_path / 'empirical_correlation_matrix.pdf', bbox_inches='tight')
print(f'Figure saved at {out_path}/empirical_correlation_matrix.pdf')
plt.close(fig)