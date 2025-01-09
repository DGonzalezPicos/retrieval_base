"""Plot best fit spectra for the chunks for both TWA 28, 27A
overplot the BT-Settl models from Manjavacas+2024"""

import numpy as np
import matplotlib.pyplot as plt

import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# pdf pages

from matplotlib.backends.backend_pdf import PdfPages
import copy

from retrieval_base.retrieval import Retrieval
import retrieval_base.auxiliary_functions as af
from retrieval_base.config import Config

path = af.get_path(return_pathlib=True)
config_file = 'config_jwst.txt'
target = 'TWA28'
# run = None
run = 'lbl12_G1G2G3_fastchem_0'
w_set='NIRSpec'

runs = dict(TWA28='lbl12_G1G2G3_fastchem_0',
            TWA27A='lbl15_G1G2G3_fastchem_0')

def load_data(target, run):
    cwd = os.getcwd()
    if target not in cwd:
        os.chdir(f'{path}/{target}')
        print(f'Changed directory to {target}')

    conf = Config(path=path, target=target, run=run)(config_file)        
        
    m_spec = af.pickle_load(f'{conf.prefix}data/bestfit_m_spec_NIRSpec.pkl')
    m_spec.flux = m_spec.flux.squeeze()
    d_spec = af.pickle_load(f'{conf.prefix}data/d_spec_NIRSpec.pkl')
    d_spec.squeeze()
    return d_spec, m_spec

d_specs, m_specs = {}, {}
for target in runs.keys():
    d_specs[target], m_specs[target] = load_data(target, runs[target])

colors = dict(TWA28={'data':'k', 'model':'orange'},
              TWA27A={'data':'k', 'model':'green'})
lw = 0.9
def plot_chunk(d_spec, m_spec, ax=None, idx=0, relative_residuals=False, colors=None, offset=0.0, ls='-'):
    
    new_ax = (ax is None)
    if new_ax:
        fig, ax = plt.subplots(2,1, figsize=(10,5), sharex=True)
    else:
        assert len(ax) == 2, f'ax must be a list of 2 elements'
        
    ax[0].plot(d_spec.wave[idx], d_spec.flux[idx] + offset, color=colors['data'], lw=lw, alpha=0.8, ls=ls)
    ax[0].plot(d_spec.wave[idx], m_spec.flux[idx] + offset, color=colors['model'], lw=lw, alpha=0.8, ls=ls, label=target)
    
    res = d_spec.flux[idx] - m_spec.flux[idx]
    if relative_residuals:
        res = res / d_spec.flux[idx]
    ax[1].plot(d_spec.wave[idx], res, color=colors['model'], lw=lw, alpha=0.8)
    
    # if new_ax:
    ax[1].set_xlabel('Wavelength / nm')
    ax[0].set_ylabel('Flux / erg/s/cm2/nm')
    
    res_label = r'$\Delta F / F$' if relative_residuals else r'$\Delta F / erg/s/cm^2/nm$'
    ax[1].set_ylabel(res_label)
    
    # ax[1].axhline(0.0,color=colors['model'], lw=0.7)
        # ax.set_title(f'Chunk {idx}')
        # plt.show()
    return ax
    

offsets = dict(TWA28=np.zeros(d_specs['TWA28'].n_orders),
               TWA27A=1e-22*np.array([5.0, 5.0, 5.0, 5.0, 
                                      0.5, 0.5, 0.5, 0.5,
                                      0.0, 0.0, 0.0, 0.0]))

pdf_name = path / 'twx_figs/twx_best_fit_spec_chunks.pdf'
n_orders = d_specs['TWA28'].n_orders
with PdfPages(pdf_name) as pdf:
    
    for idx in range(n_orders):
        fig, ax = plt.subplots(2,1, figsize=(14,4), sharex=True, gridspec_kw={'height_ratios':[3,1]})
        for target in runs.keys():
            d_spec, m_spec = d_specs[target], m_specs[target]
            ax = plot_chunk(d_spec, m_spec, ax=ax, relative_residuals=True, idx=idx, colors=colors[target], offset=offsets[target][idx])
        
            
        if target == list(runs.keys())[-1]:
            ax[1].axhline(0.0, color='k', lw=0.7)
            ylim = ax[1].get_ylim()
            ylim_res = max(abs(ylim[0]), abs(ylim[1]))
            ax[1].set_ylim(-ylim_res, ylim_res)
            
            nans = np.isnan(d_spec.flux[idx])
            eps = 0.002
            xlim = (1-eps)*np.nanmin(d_spec.wave[idx][~nans]), (1+eps)*np.nanmax(d_spec.wave[idx][~nans])
            ax[0].set_xlim(xlim)
            ax[0].legend(loc='upper right')
            
        # save with tight layout
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
            
        
    print(f'Saved to {pdf_name}')

# plot Na, K lines between 1130, 1280 nm

fig_name = path / 'twx_figs/twx_best_fit_spec_chunks_Na_K.pdf'
xlims = [(1130, 1190), (1230, 1280)]
with PdfPages(fig_name) as pdf:

    for idx in range(2):
        fig, ax = plt.subplots(2,1, figsize=(14,4), sharex=True, gridspec_kw={'height_ratios':[3,1]})
        for target in runs.keys():
            d_spec, m_spec = d_specs[target], m_specs[target]
            ax = plot_chunk(d_spec, m_spec, ax=ax, relative_residuals=True, idx=idx, colors=colors[target], offset=offsets[target][idx],
                            ls='-' if target == 'TWA28' else '--')
        
        # save with tight layout
        ax[0].set_xlim(xlims[idx])
        ax[0].set_ylim(0.8e-14, 2.2e-14)
        ax[1].set_ylim(-0.25, 0.25)
        ax[1].axhline(0.0, color='k', lw=0.7)
        # create custom legend, TWA28: solid, TWA27A: dashed
        ax[0].legend(handles=[Line2D([0], [0], color='k', lw=lw, alpha=0.8, ls='-', label='TWA28'),
                               Line2D([0], [0], color='k', lw=lw, alpha=0.8, ls='--', label='TWA27A')],
                     loc='upper right')
        
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        
print(f'Saved to {fig_name}')