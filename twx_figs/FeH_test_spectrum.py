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

run_old = 'lbl12_G1_fastchem_1_FeH_old'
run_new = 'lbl12_G1_fastchem_1_FeH_new'
run_bestfit = 'lbl12_G1G2G3_fastchem_1'
conf_old = Config(path=path, target=target, run=run_old)(config_file)        
conf_new = Config(path=path, target=target, run=run_new)(config_file)        
conf_bestfit = Config(path=path, target=target, run=run_bestfit)(config_file)        

ret_old = Retrieval(
    conf=conf_old, 
    evaluation=False
    )
ret_new = Retrieval(
    conf=conf_new, 
    evaluation=False
    )
ret_bestfit = Retrieval(
    conf=conf_bestfit, 
    evaluation=False
    )

bestfit_params, posterior = ret_bestfit.PMN_analyze()
bestfit_params_dict = dict(zip(ret_bestfit.Param.param_keys, bestfit_params))

print(f' --> Best-fit parameters: {bestfit_params_dict}')
bestfit_params = np.array(list(bestfit_params_dict.values()))


def get_model(ret, bestfit_params_dict, update_params):
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

m_old, chi2_old, mf_old = get_model(ret_old, bestfit_params_dict, {})
m_new, chi2_new, mf_new = get_model(ret_new, bestfit_params_dict, {})

fig_name = path / 'twx_figs/FeH_test_spectrum.pdf'
colors = dict(data='k', old='forestgreen', new='orange')
lw = 1.2

# data wave and flux
wave = ret_old.d_spec['NIRSpec'].wave.squeeze()
flux = ret_old.d_spec['NIRSpec'].flux.squeeze()
n_orders = wave.shape[0]
# use PDF pages
with PdfPages(fig_name) as pdf:
    for order in range(n_orders):
        # first page is the PT profile and mass fractions profiles
        fig, ax = plt.subplots(2,1, figsize=(10, 4), tight_layout=True, sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        if order == 0:
            labels = ['data', 'old', 'new'] 
        else:
            labels = [None, None, None]
        ax[0].plot(wave[order], flux[order], color=colors['data'], label=labels[0])
        ax[0].plot(wave[order], m_old[order], color=colors['old'], label=labels[1])
        ax[0].plot(wave[order], m_new[order], color=colors['new'], label=labels[2])
        if order == 0:
            ax[0].legend()
        
        res_old = flux[order] - m_old[order]
        res_new = flux[order] - m_new[order]
        ax[1].plot(wave[order], res_old, color=colors['old'])
        ax[1].plot(wave[order], res_new, color=colors['new'])
        pdf.savefig(fig)

print(f' --> Saved to {fig_name}')