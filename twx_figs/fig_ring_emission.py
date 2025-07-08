"""Plot the contribution from the ring line emission to the total spectrum
"""

import numpy as np
import matplotlib.pyplot as plt

import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

from matplotlib.backends.backend_pdf import PdfPages
import copy
import pathlib
from retrieval_base.retrieval import Retrieval
import retrieval_base.auxiliary_functions as af
from retrieval_base.config import Config

from fig1_insets import create_insets

fontsize = 12
path = af.get_path(return_pathlib=True)
path_figures = pathlib.Path('/home/dario/phd/twa2x_paper/figures')
config_file = 'config_jwst.txt'
# target = 'TWA28'
# run = None
# run = 'lbl12_G1G2G3_fastchem_0'
w_set='NIRSpec'

# runs = dict(
#     TWA27A='lbl11_G1G2G3_fastchem_0',
#     TWA28='lbl11_G1G2G3_fastchem_0',
#             )
runs = dict(
    TWA27A='freeslab_lbl10_G1G2G3_1',
    TWA28='freeslab_lbl10_G1G2G3_1',
            )

flux_plot_unit = 1e14

colors = dict(TWA28={'data':'k', 
                    #  'model':'orange',
                    'model':'#D55E00',
                     },
              TWA27A={'data':'gray', 
                    #   'model':'#0a74da',
                      'model':'#009E73',
                      })

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
    m_spec.wave = m_spec.wave.squeeze()
    
    
    
    # pRT_model = af.pickle_load(f'{conf.prefix}data/pRT_atm_NIRSpec.pkl')
    slab_file = f'{conf.prefix}data/m_slab.npy' 
    if not os.path.exists(slab_file):
        w_set = 'NIRSpec'
        conf = Config(path=path, target=target, run=run)(config_file)        
        
        ret = Retrieval(
            conf=conf, 
            evaluation=False
            )

        bestfit_params, _ = ret.PMN_analyze()
        bestfit_params_dict = dict(zip(ret.Param.param_keys, bestfit_params))

        print(f' --> Best-fit parameters: {bestfit_params_dict}')
        bestfit_params = np.array(list(bestfit_params_dict.values()))

        ret.evaluate_model(bestfit_params)
        ret.PMN_lnL_func()
        m_full = np.squeeze(np.copy(ret.LogLike[w_set].m_flux))
        # generate model without disk species to isolate the slab contribution
        ret.pRT_atm[w_set].disk_species = []
        ret.evaluate_model(bestfit_params)
        ret.PMN_lnL_func()
        print(f' ret.pRT_atm[w_set].disk_species = {ret.pRT_atm[w_set].disk_species}')
        m_noslab = np.squeeze(ret.LogLike[w_set].m_flux)
        m_slab = m_full - m_noslab
        wave = np.squeeze(ret.d_spec[w_set].wave)
        print(f'shape m_slab = {m_slab.shape}')
        # store slab model as txt with two columns: wave and flux, save as (2)
        m_slab = np.array([wave, m_slab]).T
        print(f' saving m_slab array with shape {m_slab.shape}')
        np.save(slab_file, m_slab)
        print(f'Saved slab model to {slab_file}')
    else:
        m_slab = np.load(slab_file).T
        print(f'shape m_slab = {m_slab.shape}')
        # print(f'm_slab[:,0] = {m_slab[:,0]}')
        assert np.shape(m_slab[0,:]) == np.shape(m_spec.wave), f'shape m_slab = {np.shape(m_slab)} != shape m_spec.wave = {np.shape(m_spec.wave)}'
        
    m_spec.flux_slab = np.copy(m_slab[1,])
    return d_spec, m_spec

d_specs, m_specs = {}, {}
for target in runs.keys():
    d_specs[target], m_specs[target] = load_data(target, runs[target])

fig, ax = plt.subplots(4,1, figsize=(6, 7), gridspec_kw={'height_ratios': [2, 1, 0.6, 0.6]}, sharex=True)
lw = 0.6
def plot_chunk(d_spec, m_spec, idx=0, colors=None, ls='-', lw=1.0):
    
    f = d_spec.flux_unit_factor
    ax[0].plot(d_spec.wave[idx], flux_plot_unit*d_spec.flux[idx] / f, color=colors['data'], lw=lw, alpha=0.8, ls=ls)
    ax[0].plot(d_spec.wave[idx], flux_plot_unit*m_spec.flux[idx] / f, color=colors['model'], lw=lw, alpha=0.8, ls=ls)
    
    residuals_list = []
    if hasattr(m_spec, 'flux_slab'):
        print(f'm_spec.flux_slab.shape = {m_spec.flux_slab.shape}')
        ax[1].plot(m_spec.wave[idx], flux_plot_unit*m_spec.flux_slab[idx] / f, color=colors['model'], lw=lw, alpha=0.8, ls=ls)
        m_flux_no_slab = m_spec.flux[idx] - m_spec.flux_slab[idx]
        res_slab = flux_plot_unit*(d_spec.flux[idx] - m_flux_no_slab) / f
        # res_slab /= d_spec.flux[idx]
        MAD = np.nanmedian(np.abs(res_slab))
        
        chi2_r = np.nansum((d_spec.flux[idx] - m_flux_no_slab)**2 / d_spec.err[idx]**2) / np.sum(~np.isnan(d_spec.flux[idx]))
        print(f'chi2_r = {chi2_r:.2e} (no slab)')
        ax[2].plot(d_spec.wave[idx], 10*res_slab, color=colors['model'], lw=lw, alpha=0.8, ls=ls)
        residuals_list.append(res_slab)
    res = flux_plot_unit*(d_spec.flux[idx] - m_spec.flux[idx]) / f
    # res /= d_spec.flux[idx]
    MAD = np.nanmedian(np.abs(res))
    print(f'MAD = {MAD:.2e} (total)')
    ax[-1].plot(d_spec.wave[idx], 10*res, color=colors['model'], lw=lw, alpha=0.8, ls=ls)
    residuals_list.append(res)
    return residuals_list
    
MAD_dict = {}
for target in runs.keys():
    run = runs[target]
    d_spec, m_spec = d_specs[target], m_specs[target]
    
    residuals_no_slab = 0.0
    residuals_with_slab = 0.0
    for idx in [-3,-2, -1]:
        residuals_list = plot_chunk(d_spec, m_spec, idx=idx, colors=colors[target], lw=lw)
        residuals_no_slab += residuals_list[0]
        residuals_with_slab += residuals_list[1]
    MAD_dict[target] = [np.nanmedian(np.abs(residuals_no_slab)), np.nanmedian(np.abs(residuals_with_slab))]
    
    
xlim = (4200, 5200)
ax[0].set_xlim(xlim)
ax[0].set_ylim(flux_plot_unit*0.60e-15, flux_plot_unit*1.65e-15)

ax[1].text(0.02, 0.84, r'$^{12}\mathrm{CO}$ slab', transform=ax[1].transAxes, fontsize=fontsize, ha='left', va='top')

res_text = ['w/o slab', 'w/ slab']
pe = pe.withStroke(linewidth=1, foreground='white')
yticks = [-0.03, 0.0, 0.03]
for axi, text in zip(ax[2:], res_text):
    axi.set(ylim=(-0.035, 0.05))
    axi.axhline(0, color='k', lw=0.5,zorder=-1)
    axi.text(0.02, 0.84, text, transform=axi.transAxes, fontsize=fontsize, ha='left', va='top',
            #  bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'),
             path_effects=[pe],
             )
    axi.set_yticks(yticks)
    
# add text in upper right corner of ax[0] listing the MAD values for both objects and both cases
xm = 0.62
ym = 0.848
ax[0].text(s='MAD / 10$^{-16}$', x=xm+0.152, y=ym+0.12, transform=ax[0].transAxes, fontsize=fontsize, ha='left', va='top')
ax[0].text(s='slab', x=xm, y=ym, transform=ax[0].transAxes, fontsize=fontsize, ha='left', va='top')
ax[0].text(s='no', x=xm+0.16, y=ym, transform=ax[0].transAxes, fontsize=fontsize, ha='left', va='top')
ax[0].text(s='yes', x=xm+0.265, y=ym, transform=ax[0].transAxes, fontsize=fontsize, ha='left', va='top')
# TODO: add table lines in transAxes, only do when shape of the figure is final....
ax[0].plot([xm+0.14, xm+0.35], [ym+0.02, ym+0.02], color='k', lw=0.5, ls='-', transform=ax[0].transAxes)
ax[0].plot([xm-0.03, xm+0.35], [ym-0.10, ym-0.10], color='k', lw=0.5, ls='-', transform=ax[0].transAxes)

for t, target in enumerate(runs.keys()):
    ax[0].text(s=f'TWA {target.replace("TWA", "")}', x=xm-0.04, y=ym-0.12*(t+1) - 0.005, transform=ax[0].transAxes, fontsize=fontsize, ha='left', va='top', weight='bold',
               color=colors[target]['model'], path_effects=[pe])
    ax[0].text(s=f'{100*MAD_dict[target][0]:.2f}', x=xm+0.15, y=ym-0.12*(t+1), transform=ax[0].transAxes, fontsize=fontsize, ha='left', va='top')
    ax[0].text(s=f'{100*MAD_dict[target][1]:.2f}', x=xm+0.26, y=ym-0.12*(t+1), transform=ax[0].transAxes, fontsize=fontsize, ha='left', va='top',
               weight='bold')
# add text to indicate a,b,c, of each panel for easy reference
panels = ['a', 'b', 'c', 'd']
ax[0].text(s=panels[0], x=0.02, y=0.12, transform=ax[0].transAxes, fontsize=fontsize, ha='left', va='top', weight='bold', path_effects=[pe])
ax[1].text(s=panels[1], x=0.02, y=0.22, transform=ax[1].transAxes, fontsize=fontsize, ha='left', va='top', path_effects=[pe], weight='bold')
ax[2].text(s=panels[2], x=0.02, y=0.28, transform=ax[2].transAxes, fontsize=fontsize, ha='left', va='top', path_effects=[pe], weight='bold')
ax[3].text(s=panels[3], x=0.02, y=0.28, transform=ax[3].transAxes, fontsize=fontsize, ha='left', va='top', path_effects=[pe], weight='bold')
# add axes labels
fig.text(-0.15, 0.12, r'$F_{\lambda}$' + r' / 10$^{-14}$ $\mathrm{erg\,s^{-1}\,cm^{-2}}$', transform=ax[0].transAxes, fontsize=fontsize, ha='left', va='center', rotation='vertical')
fig.text(0.5, -0.92, r'Wavelength / nm', transform=ax[-1].transAxes, fontsize=fontsize, ha='center', va='bottom')
fig.text(-0.14, 0.12, r'Residuals', transform=ax[-1].transAxes, fontsize=fontsize, ha='center', va='bottom', rotation='vertical')
# fig_name = path / 'twx_figs' / 'fig_ring_emission.pdf'
fig_name = path_figures / 'fig_ring_emission.pdf'
fig.savefig(fig_name, bbox_inches='tight')
print(f'Saved figure to {fig_name}')
plt.close()