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

from fig1_insets import create_insets

path = af.get_path(return_pathlib=True)
config_file = 'config_jwst.txt'
target = 'TWA28'
# run = None
run = 'lbl12_G1G2G3_fastchem_0'
w_set='NIRSpec'

runs = dict(TWA28='lbl12_G1G2G3_fastchem_0',
            TWA27A='lbl15_G1G2G3_fastchem_0',
            )

dw = 90
xc = [1110, 2290, 4510]
inset_regions = [[(xc[0]-dw, xc[0]+dw), (8e-15, 2.30e-14)],
                 [(xc[1]-dw, xc[1]+dw), (4.5e-15, 8.3e-15)],
                 [(xc[2]-dw, xc[2]+dw), (8e-16, 1.35e-15)]]
fig, ax, axins = create_insets(inset_regions)


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
              TWA27A={'data':'brown', 'model':'green'})

lw = 0.9
def plot_chunk(d_spec, m_spec, idx=0, relative_residuals=False, colors=None, ls='-',
               plot_bb=False):
    
        
    ax[0].plot(d_spec.wave[idx], d_spec.flux[idx], color=colors['data'], lw=lw, alpha=0.8, ls=ls)
    ax[0].plot(d_spec.wave[idx], m_spec.flux[idx], color=colors['model'], lw=lw, alpha=0.8, ls=ls)
    
    res = (d_spec.flux[idx] - m_spec.flux[idx]) / d_spec.flux[idx]
    ax[1].plot(d_spec.wave[idx], res, color=colors['model'], lw=lw, alpha=0.8, ls=ls)


fig, ax, axins = create_insets(inset_regions, residuals_plot=True)
# add an inset showing the logscale y axis with the disk flux
axins_disk = ax[0].inset_axes([0.5, 0.36, 0.48, 0.58])

n_orders = d_specs['TWA28'].n_orders
for idx in range(n_orders):
    for t, target in enumerate(runs.keys()):
        d_spec, m_spec = d_specs[target], m_specs[target]
        plot_chunk(d_spec, m_spec, idx=idx, colors=colors[target])
        
        for r, region in enumerate(inset_regions):
            x1, x2 = region[0]
            mask = (d_spec.wave[idx] > x1) & (d_spec.wave[idx] < x2)
            if mask.sum() > 0:
                axins[r].plot(d_spec.wave[idx][mask], d_spec.flux[idx][mask], color=colors[target]['data'], lw=lw, alpha=0.8)
                axins[r].plot(d_spec.wave[idx][mask], m_spec.flux[idx][mask], color=colors[target]['model'], lw=lw, alpha=0.8)
                
        axins_disk.plot(d_spec.wave[idx], d_spec.flux[idx], color=colors[target]['data'], lw=lw*0.6, alpha=0.8)
        axins_disk.plot(d_spec.wave[idx], m_spec.flux[idx], color=colors[target]['model'], lw=lw*0.6, alpha=0.8)
        axins_disk.plot(d_spec.wave[idx], m_spec.flux_bb[idx], color=colors[target]['model'], lw=lw*1.7, alpha=0.8, ls='--')
        # axins_disk.plot(d_spec.wave[idx],m_spec.flux[idx] -  m_spec.flux_bb[idx], color=colors[target]['model'], lw=lw, alpha=0.8, ls='--')
        
ax[0].set_ylim(1e-16, 2.5e-14)
ax[0].set_xlim(920, 5300)
ax[1].axhline(0, color='k', lw=0.5)
# make ylims for residuals symmetric
ylim = ax[1].get_ylim()
ylim_s = max(abs(ylim[0]), abs(ylim[1]))
ax[1].set_ylim(-ylim_s, ylim_s)
# add label to the y axis
ax[0].set_ylabel(r'$F_{\lambda}$' '  / ' 'erg ' r'$s^{-1} cm^{-2} nm^{-1}$')
ax[1].set_ylabel(r'$\Delta F_{\lambda} / F_{\lambda}$')

axins_disk.set_ylabel(r'$F_{\lambda}$' '  / ' 'erg ' r'$s^{-1} cm^{-2} nm^{-1}$')
axins_disk.set_xlabel(r'Wavelength / nm')
# add common xlabel 
fig.text(0.5, -0.53, r'Wavelength / nm', ha='center', va='center')

# create custom legend with names of the targets and disk contribution as dashed lines
# legend_elements = [Line2D([0], [0], color=colors['TWA28']['data'], lw=lw, label='TWA28 data'),
#                     Line2D([0], [0], color=colors['TWA28']['model'], lw=lw, label='TWA28 model', ls='-'),
#                     Line2D([0], [0], color=colors['TWA28']['model'], lw=lw*1.7, label='TWA28 blackbody', ls='--'),
#                     Line2D([0], [0], color=colors['TWA27A']['data'], lw=lw, label='TWA27A data'),
#                     Line2D([0], [0], color=colors['TWA27A']['model'], lw=lw, label='TWA27A model', ls='-'),
#                     Line2D([0], [0], color=colors['TWA27A']['model'], lw=lw*1.7, label='TWA27A blackbody', ls='--')]
# ax[0].legend(handles=legend_elements, ncol=2, loc=(0.15, 0.7), fontsize=10)

legend_elements_twa28 = [
    Line2D([0], [0], color=colors['TWA28']['data'], lw=lw, label='Data'),
    Line2D([0], [0], color=colors['TWA28']['model'], lw=lw, label='Model', ls='-'),
    Line2D([0], [0], color=colors['TWA28']['model'], lw=lw*1.7, label='BB', ls='--')
]

legend_elements_twa27a = [
    Line2D([0], [0], color=colors['TWA27A']['data'], lw=lw, label='Data'),
    Line2D([0], [0], color=colors['TWA27A']['model'], lw=lw, label='Model', ls='-'),
    Line2D([0], [0], color=colors['TWA27A']['model'], lw=lw*1.7, label='BB', ls='--')
]

# Add legends to the plot, make title bold
legend1 = ax[0].legend(handles=legend_elements_twa28, title='TWA 28', loc=(0.13, 0.64), fontsize=10, frameon=False,
                       title_fontproperties={'weight': 'bold', 'size': 11})
legend2 = ax[0].legend(handles=legend_elements_twa27a, title='TWA 27A', loc=(0.26, 0.64), fontsize=10, frameon=False,
                       title_fontproperties={'weight': 'bold', 'size': 11})

# Add the legends to the axes
ax[0].add_artist(legend1)
ax[0].add_artist(legend2)

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

mark_inset(ax[0], axins[0], loc1=1, loc2=2, fc="none", ec="0.5", zorder=-1)
mark_inset(ax[0], axins[1], loc1=1, loc2=2, fc="none", ec="0.5", zorder=-1)
mark_inset(ax[0], axins[2], loc1=1, loc2=2, fc="none", ec="0.5", zorder=-1)


axins_disk.set_yscale('log')
axins_disk.set_ylim(1e-16, 4e-14)
axins_disk.set_xlim(920, 5300)

# plot nirspec bands on axins_disk
gratings = dict(g140h=(900, 1900),
                g235h=(1650, 3180),
                g395h=(2890, 5290),
                )
gratings_text = dict(g140h=6e-16,
                     g235h=6e-16,
                     g395h=7e-15)
colors = ['navy', 'green', 'brown']
# use a faded grey for the bands

for band, color in zip(gratings.keys(), colors):
    axins_disk.axvspan(gratings[band][0], gratings[band][1], color=color, alpha=0.12, lw=0)
    xc = gratings[band][0] + (gratings[band][1] - gratings[band][0])/2
    if band == 'g140h':
        xc -= 90
    axins_disk.text(xc, gratings_text[band], band.upper(), color=color, fontsize=10,
                    ha='center', va='center', fontweight='bold')


fig_name = path / 'twx_figs/fig1_spec.pdf'
fig.savefig(fig_name, bbox_inches='tight')
plt.close()
# close all
plt.close('all')
print(f'Saved figure to {fig_name}')
