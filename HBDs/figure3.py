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



path = pathlib.Path('/home/dario/phd/retrieval_base')
# out_path = path / 'HBDs'
out_path = pathlib.Path('/home/dario/phd/Hot_Brown_Dwarfs_Retrievals/figures/')

targets = dict(J1200='freechem_10', 
               TWA28='freechem_6', 
               J0856='freechem_9'
               )
colors = dict(J1200='royalblue', TWA28='seagreen', J0856='indianred')

prefix = 'freechem'

# orders = np.arange(5)
orders = [3]

for order in orders:
    fig, ax = plt.subplots(len(targets)+1,1, figsize=(16,6), sharex=True,
                        gridspec_kw={'height_ratios': [1,1,1,0.5], 'wspace': 0.05, 'hspace': 0.05})
    # ax_transm = ax[-1].twinx()


    handles = []
    for i, (target, retrieval_id) in enumerate(targets.items()):
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
        
        # snr = np.nanmedian(d_spec.flux[order]/d_spec.err[order])
        # err = d_spec.err[order] * loglike.beta[order,:,None]
        # snr = np.nanmedian(d_spec.flux[order]/err)
        
        # print(f' {target} --> median SNR = {snr:.2f}')
        # average pixel size
        sample_rate = np.mean(np.diff(d_spec.wave[order,0]))
        print(f'order {order} --> sample rate = {sample_rate:.3e} nm')
                # load json file with bestfit parameters
        with open(retrieval_path / 'test_data/bestfit.json', 'r') as f:
            bestfit_params = json.load(f)
            
        params = bestfit_params['params']
        RV = params['rv']
        # print(params.keys())
        # RV = bestfit_params['params']['RV']
        
        det_err = []
        for det in range(3):
            x = d_spec.wave[order,det]
            sample_rate = np.mean(np.diff(x))
            # print(f'sample rate = {sample_rate:.3e} nm')
            y = d_spec.flux[order,det] * 1e15
            # estimate SNR of the spectrum (from pipeline)
            
            # err = d_spec.err[order,det] * loglike.beta[order,det,None] * 1e15
            # median_err = np.nanmedian(err)
            finite = d_spec.mask_isfinite[order,det]
            # cov[order,det].get_covariance_matrix()
            err = np.nan * np.ones_like(y)
            err[finite] = np.sqrt(np.diag(cov[order,det].get_dense_cov())) * loglike.beta[order,det,None] * 1e15
            snr = np.nanmedian(y[finite]/err[finite])
            print(f' {target} --> median SNR of detector {det} = {snr:.2f}')
            # scatter median error to show uncertainty
            det_err.append(np.nanmean(err))
            # print(f' median error of detector {det} = {np.nanmedian(err):.2f} x 10^-15 erg s^-1cm^-2nm^-1')
            
            
            m = m_spec.flux[order,det] * loglike.f[order,det,None] * 1e15
            ax[i].plot(x, y, lw=1.2, color='k')
            ax[i].fill_between(x, y-err, y+err, color='k', alpha=0.2)
            ax[i].plot(x, m, lw=1.2, ls='-', color=colors[target])
            # ax[i].set(xlim=(x.min(), x.max()))
            if i == 1:
                # increase padding of ylabel 
                ax[i].set_ylabel('Flux\n[10$^{-15}$ erg s$^{-1}$cm$^{-2}$nm$^{-1}$]', labelpad=20)
            # add transparent box with target name
            ax[i].text(0.912, 0.78, target, transform=ax[i].transAxes, fontsize=18, weight='bold',
                        bbox=dict(facecolor='white', edgecolor='white', alpha=0.2, boxstyle='round'))

            res = y - m
            # shift residuals to rest-frame 
            res_shift = np.interp(x, x*(1-RV/2.998e5), res)
            
            # ax[-1].plot(x, res, lw=1., label='residuals', color=colors[target], alpha=0.8)
            ax[-1].plot(x, res_shift, lw=1., color=colors[target], alpha=0.8)

            # ax_transm.plot(x, transm[order,det], lw=2., label='transmission', color='k', alpha=0.1)
        # plot the average error of the three detectors
        mean_err = np.mean(det_err)
        pad = 0.1
        ax[i].errorbar(np.max(x)+1.0, np.nanmean(y), yerr=mean_err, color=colors[target], lw=2.5, capsize=5, capthick=2.5)
        # ax[i].legend()
        ax[-1].errorbar(x.max()+pad+(0.8*i), 0, yerr=mean_err, color=colors[target], lw=2.5, capsize=5, capthick=2.5)
        
        # ax[-1].axhspan(-median_err, median_err, color='k', alpha=0.2)
    ax[-1].axhline(0, ls='-', color='k', lw=1.5)
    ax[-1].set(xlabel='Wavelength [nm]', ylabel='Residual')

    ylim = 1.4
    ax[-1].set_ylim(-ylim, ylim)
    ax[-1].set_xlim(d_spec.wave[order].min()-0.2, d_spec.wave[order].max()+2.5)

    # ax[0].legend()
    plt.show()
    fig.savefig(out_path / f'fig3_order{order}.pdf', bbox_inches='tight', dpi=300)
    print('- Saved figure to ', out_path / f'fig3_order{order}.pdf')




