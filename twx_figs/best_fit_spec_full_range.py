"""Plot best fit spectra for full wavelength range with contribution from disk and atmosphere
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
            TWA27A='lbl15_G1G2G3_fastchem_0',
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

d_specs, m_specs = {}, {}
for target in runs.keys():
    d_specs[target], m_specs[target] = load_data(target, runs[target])

colors = dict(TWA28={'data':'k', 'model':'orange'},
              TWA27A={'data':'k', 'model':'green'})
lw = 0.9
def plot_chunk(d_spec, m_spec, ax=None, idx=0, relative_residuals=False, colors=None, offset=0.0, ls='-',
               plot_bb=False, inset_args={}, inset=None):
    
    new_ax = (ax is None)
    if new_ax:
        fig, ax = plt.subplots(2,1, figsize=(10,5), sharex=True)
    else:
        # assert len(ax) == 2, f'ax must be a list of 2 elements'
        pass
        
    ax[0].plot(d_spec.wave[idx], d_spec.flux[idx] + offset, color=colors['data'], lw=lw, alpha=0.8, ls=ls)
    ax[0].plot(d_spec.wave[idx], m_spec.flux[idx] + offset, color=colors['model'], lw=lw, alpha=0.8, ls=ls)
    
    ax_inset = None
    if len(inset_args) > 0 and 'inset' not in inset_args:
        # create inset axes for ax[0]
        ax_inset = ax[0].inset_axes(inset_args['xywh'])
        ax_inset.set_xlim(inset_args['xlim'])
        inset_args['inset'] = ax_inset

    
    if inset is not None:
        ax_inset = inset
    if ax_inset is not None:
        mask = (d_spec.wave[idx] > inset_args['xlim'][0]) & (d_spec.wave[idx] < inset_args['xlim'][1])
        ax_inset.plot(d_spec.wave[idx][mask], d_spec.flux[idx][mask] + offset, color=colors['data'], lw=lw, alpha=0.8, ls=ls)
        ax_inset.plot(d_spec.wave[idx][mask], m_spec.flux[idx][mask] + offset, color=colors['model'], lw=lw, alpha=0.8, ls=ls)
    
    if plot_bb:
        ax[0].plot(d_spec.wave[idx], m_spec.flux_bb[idx], color=colors['model'], lw=lw, alpha=0.8, ls=ls)
        # ax[0].fill_between(d_spec.wave[idx], m_spec.flux_bb[idx], color=colors['model'], alpha=0.2)
    ax[0].set_ylabel('Flux / erg/s/cm2/nm')

    if len(ax) == 2:
        res = d_spec.flux[idx] - m_spec.flux[idx]
        if relative_residuals:
            res = res / d_spec.flux[idx]
        ax[1].plot(d_spec.wave[idx], res, color=colors['model'], lw=lw, alpha=0.8)
        
        # if new_ax:
        ax[1].set_xlabel('Wavelength / nm')
    
        # res_label = r'$\Delta F / F$' if relative_residuals else r'$\Delta F / erg/s/cm^2/nm$'
        res_label = 'Data - Model'
        if relative_residuals:
            res_label = '(Data - Model)\n/ Data'
        ax[1].set_ylabel(res_label)
    
    # ax[1].axhline(0.0,color=colors['model'], lw=0.7)
        # ax.set_title(f'Chunk {idx}')
        # plt.show()
    return ax, inset_args
    

offsets = dict(TWA28=np.zeros(d_specs['TWA28'].n_orders),
               TWA27A=1e-22*np.array([5.0, 5.0, 5.0, 5.0, 
                                      0.5, 0.5, 0.5, 0.5,
                                      0.0, 0.0, 0.0, 0.0]))

pdf_name = path / 'twx_figs/fig1_spec_full_range.pdf'
n_orders = d_specs['TWA28'].n_orders

fig, ax = plt.subplots(3,1, figsize=(14,4), sharex=True, gridspec_kw={'height_ratios':[3,3,1]})

inset_idx = {'5': [(2260, 2390), [0.4, 0.45, 0.2, 0.4]],
             '10': [(4400, 4700), [0.75, 0.45, 0.2, 0.4]]}
inset_args = {}
for idx in range(n_orders):
    for t, target in enumerate(runs.keys()):
        d_spec, m_spec = d_specs[target], m_specs[target]
        
        # inset_args = {}
        # inset_i = None

        if idx in np.array(list(inset_idx.keys())).astype(int):
            if not 'inset' in inset_args.get(str(idx), {}):
                inset_xlim = inset_idx[str(idx)][0]
                inset_args[str(idx)] = dict(xlim=inset_xlim, xywh=inset_idx[str(idx)][1])
                print(f'Inset xlim: {inset_xlim} at idx: {idx}')
            # else:
            #     inset_args[str(idx)]['inset'] = inset_i
        
        axes, inset_args[str(idx)] = plot_chunk(d_spec, m_spec, ax=[ax[0], ax[2]], 
                        relative_residuals=True, 
                        idx=idx, 
                        colors=colors[target], 
                        offset=offsets[target][idx],
                        plot_bb=True,
                        inset_args=inset_args.get(str(idx), {})
        )
        
        _ = plot_chunk(d_spec, m_spec, ax=[ax[1]],
                       relative_residuals=True,
                       idx=idx,
                       colors=colors[target],
                       offset=offsets[target][idx],
                       plot_bb=True)
    
ax[-1].axhline(0.0, color='k', lw=0.7)
ylim = ax[-1].get_ylim()
ylim_res = max(abs(ylim[0]), abs(ylim[1]))
ax[-1].set_ylim(-ylim_res, ylim_res)

xlim = np.nanmin(d_spec.wave), np.nanmax(d_spec.wave)
ax[0].set_xlim(xlim)
ax[0].legend(loc='upper right')

ax[0].set_yscale('linear')
ax[0].set_ylim(0.0, None)

ax[1].set_yscale('log')
ax[1].set_ylim(1e-16, None)

fig.tight_layout()
plt.savefig(pdf_name)
plt.close(fig)
plt.close()
print(f'Saved to {pdf_name}')
