""" 
Manually adjust abundance of selected species to check if they have any impact
on the spectrum.

date: 2025-01-14
"""
import pathlib
import numpy as np
import os
import matplotlib.pyplot as plt
# pdf pages
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

from retrieval_base.retrieval import Retrieval
import retrieval_base.auxiliary_functions as af
from retrieval_base.config import Config
# import config_jwst as conf

path = pathlib.Path(af.get_path())
config_file = 'config_jwst.txt'
target = 'TWA28'
w_set='NIRSpec'

cwd = os.getcwd()
if target not in cwd:
    nwd = os.path.join(cwd, target)
    print(f'Changing directory to {nwd}')
    os.chdir(nwd)


def get_bestfit_params(run):
    conf = Config(path=path, target=target, run=run)(config_file)        
        
    ret = Retrieval(
        conf=conf, 
        evaluation=False
        )

    bestfit_params, posterior = ret.PMN_analyze()
    bestfit_params_dict = dict(zip(ret.Param.param_keys, bestfit_params))
    return bestfit_params_dict

run = 'lbl12_G1_fastchem_1'
run_bestfit = 'lbl12_G1G2G3_fastchem_1'
conf = Config(path=path, target=target, run=run)(config_file)        
conf_bestfit = Config(path=path, target=target, run=run_bestfit)(config_file)        
    
ret_bestfit = Retrieval(
    conf=conf_bestfit, 
    evaluation=False
    )

bestfit_params, posterior = ret_bestfit.PMN_analyze()
bestfit_params_dict = dict(zip(ret_bestfit.Param.param_keys, bestfit_params))

print(f' --> Best-fit parameters: {bestfit_params_dict}')
bestfit_params = np.array(list(bestfit_params_dict.values()))

ret = Retrieval(
    conf=conf, 
    evaluation=False
    )
ret.evaluate_model(bestfit_params)
ret.Chem.set_species_info()
# ret.PMN_lnL_func()

wave = np.squeeze(ret.d_spec[w_set].wave)
d_flux = np.squeeze(ret.d_spec[w_set].flux)

def new_model(bestfit_params_dict, update_params):
    # call model again with new K profile

    bestfit_params_dict_copy = bestfit_params_dict.copy()
    bestfit_params_dict_copy.update(update_params)
    bestfit_params_copy = np.array(list(bestfit_params_dict_copy.values()))
    print(f' --> New parameters: {bestfit_params_dict_copy}')
    ret.Param.param_keys = list(bestfit_params_dict_copy.keys())
    ret.evaluate_model(bestfit_params_copy)

    ret.PMN_lnL_func()

    m_flux_full_new = np.squeeze(ret.LogLike[w_set].m_flux)
    chi2_full_new = ret.LogLike[w_set].chi_squared_red
    print(f' Chi2 = {chi2_full_new:.2f}')
    mf_new = ret.Chem.mass_fractions
    return m_flux_full_new, chi2_full_new, mf_new


species = '46TiO'
line_species = ret.Chem.pRT_name_dict_r[species]
update_params = {
    # f'alpha_{species}': 2.0,
    'log_TiO/46TiO': 0.0,
    }
xlim = (900, 1900)

m_flux_full, chi2_full, mf = new_model(bestfit_params_dict, update_params={})
m_flux_full_new, chi2_full_new, mf_new = new_model(bestfit_params_dict, update_params)

n_orders = m_flux_full.shape[0]
p = ret.Chem.pressure

fig_name = path / f'{target}/{conf.prefix}plots/{species}_test.pdf'
colors = dict(data='k', old='forestgreen', new='orange')
lw = 1.2
# use PDF pages
with PdfPages(fig_name) as pdf:
    
    # first page is the PT profile and mass fractions profiles
    fig, ax = plt.subplots(1,2, figsize=(10, 4), tight_layout=True, sharey=True)
    ax[0].plot(ret.PT.temperature, ret.PT.pressure)
    ax[1].plot(mf[line_species], p, color=colors['old'])
    ax[1].plot(mf_new[line_species], p, color=colors['new'])
    ax[0].set_xlabel('Temperature [K]')
    ax[0].set_ylabel('Pressure [bar]')
    ax[1].set_xlabel('Mass Fraction')
    ax[0].set(ylim=(p.max(), p.min()), yscale='log')
    ax[1].set(ylim=(p.max(), p.min()), yscale='log', xscale='log', xlim=(1e-10, 1e-2))
    pdf.savefig(fig)
    
    for order in range(n_orders):
        fig, ax = plt.subplots(2,1, figsize=(10, 4), tight_layout=True, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        ax[0].plot(wave[order], d_flux[order], color='k', lw=lw)
        ax[0].plot(wave[order], m_flux_full[order], lw=lw, color=colors['old'], alpha=0.8)
        ax[0].plot(wave[order], m_flux_full_new[order], lw=lw, color=colors['new'], alpha=0.8)
        ax[0].set_ylabel('Flux [erg/s/cm2/nm]')
        # ax[0].set_xlim(xlim)
        # ax[0].set_title(f'Order {order}')
        res = d_flux[order] - m_flux_full[order]
        res_new = d_flux[order] - m_flux_full_new[order]
        ax[1].plot(wave[order], res, lw=lw, color=colors['old'], alpha=0.8)
        ax[1].plot(wave[order], res_new, lw=lw, color=colors['new'], alpha=0.8)
        ax[1].set_ylabel('Residuals')
        ax[1].set_xlabel('Wavelength [nm]')
        pdf.savefig(fig)
        plt.close(fig)


print(f'Saved to {fig_name}')
# plt.show()
# plt.close()