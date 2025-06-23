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
import pathlib
from retrieval_base.retrieval import Retrieval
import retrieval_base.auxiliary_functions as af
from retrieval_base.config import Config

from fig1_insets import create_insets

path = af.get_path(return_pathlib=True)
path_figures = pathlib.Path('/home/dario/phd/twa2x_paper/figures')
config_file = 'config_jwst.txt'
target = 'TWA28'
# run = None
# run = 'lbl12_G1G2G3_fastchem_0'
w_set='NIRSpec'

runs = dict(
    TWA27A='freeslab_lbl10_G1G2G3_0',
    TWA28='freeslab_lbl10_G1G2G3_0',
            )

dw = 90
xc = [1110, 2290, 4510]
y_factor = 0.96e14
inset_regions = [[(xc[0]-dw, xc[0]+dw), (1.2e-14*y_factor, 2.45e-14*y_factor)],
                 [(xc[1]-dw, xc[1]+dw), (4.6e-15*y_factor, 8.7e-15*y_factor)],
                 [(xc[2]-dw, xc[2]+dw), (8.5e-16*y_factor, 1.46e-15*y_factor)]]
fig, ax, axins = create_insets(inset_regions)


def load_data(target, run):
    cwd = os.getcwd()
    if target not in cwd:
        os.chdir(f'{path}/{target}')
        print(f'Changed directory to {target}')

    conf = Config(path=path, target=target, run=run)(config_file)        
        
    m_spec = af.pickle_load(f'{conf.prefix}data/bestfit_m_spec_NIRSpec.pkl')
    d_spec = af.pickle_load(f'{conf.prefix}data/d_spec_NIRSpec.pkl')
    
    cov = af.pickle_load(f'{conf.prefix}data/bestfit_Cov_NIRSpec.pkl')
    # print(f'len(cov) = {len(cov)}')
    # print(f'cov.shape = {cov.shape}')
    err = np.array([cov[cov_i,0].get_err(mask=d_spec.mask_isfinite[cov_i]) for cov_i in range(len(cov))])
    d_spec.err = err
    print(d_spec.err.shape)

    m_spec.flux = m_spec.flux.squeeze()
    
    m_spec.wave = d_spec.wave 
    m_spec.flux_bb = m_spec.blackbody_disk(**m_spec.blackbody_disk_args).squeeze()
    d_spec.squeeze()
    return d_spec, m_spec

d_specs, m_specs = {}, {}
for target in runs.keys():
    d_specs[target], m_specs[target] = load_data(target, runs[target])

colors = dict(TWA28={'data':'k', 
                    #  'model':'orange',
                    'model':'#e89c4b',
                     },
              TWA27A={'data':'#733b27', 
                    #   'model':'#0a74da',
                      'model':'seagreen',
                      })

lw = 0.9
def plot_chunk(d_spec, m_spec, idx=0, relative_residuals=False, colors=None, ls='-',
               plot_bb=False):
    
    nans = np.isnan(d_spec.flux[idx])
    ax[0].plot(d_spec.wave[idx], d_spec.flux[idx], color=colors['data'], lw=lw, alpha=0.8, ls='-')
    ax[0].plot(d_spec.wave[idx], np.where(nans, np.nan, m_spec.flux[idx]), color=colors['model'], lw=lw, alpha=0.8, ls=ls)
    
    res = (d_spec.flux[idx] - m_spec.flux[idx]) / d_spec.flux[idx]
    ax[1].plot(d_spec.wave[idx], np.where(nans, np.nan, res), color=colors['model'], 
               lw=lw, alpha=0.8,
            #    ls=ls,
            ls='',
               marker='o',
               markersize=0.7,
               markevery=10,
               )
               


fig, ax, axins = create_insets(inset_regions, residuals_plot=True)
# add an inset showing the logscale y axis with the disk flux
axins_disk = ax[0].inset_axes([0.5, 0.36, 0.48, 0.58])

# n_orders = d_specs['TWA28'].n_orders
n_orders = len(d_specs['TWA28'].wave)
print(n_orders)
for idx in range(n_orders):
    for t, target in enumerate(runs.keys()):
        d_spec, m_spec = d_specs[target], m_specs[target]
        plot_chunk(d_spec, m_spec, idx=idx, colors=colors[target])
        
        for r, region in enumerate(inset_regions):
            x1, x2 = region[0]
            mask = (d_spec.wave[idx] > x1) & (d_spec.wave[idx] < x2)
            if mask.sum() > 0:
                nans = np.isnan(d_spec.flux[idx][mask])
                # axins[r].plot(d_spec.wave[idx][mask], d_spec.flux[idx][mask], color=colors[target]['data'], lw=lw, alpha=0.8)
                # axins[r].fill_between(d_spec.wave[idx][mask], d_spec.flux[idx][mask] - d_spec.err[idx][mask], d_spec.flux[idx][mask] + d_spec.err[idx][mask], 
                #                       color=colors[target]['data'], alpha=0.2, lw=0.5)
                axins[r].errorbar(d_spec.wave[idx][mask], d_spec.flux[idx][mask], yerr=d_spec.err[idx][mask], 
                                 color=colors[target]['data'], lw=lw, alpha=0.8,
                                 marker='o',
                                 markersize=0.7,
                                 markevery=10,
                                 zorder=-1
                                 )
                axins[r].plot(d_spec.wave[idx][mask], np.where(nans, np.nan, m_spec.flux[idx][mask]), color=colors[target]['model'], lw=lw, alpha=0.8)
                
        nans = np.isnan(d_spec.flux[idx])
        axins_disk.plot(d_spec.wave[idx], d_spec.flux[idx], color=colors[target]['data'], lw=lw*0.6, alpha=0.8)
        axins_disk.plot(d_spec.wave[idx], np.where(nans, np.nan, m_spec.flux[idx]), color=colors[target]['model'], lw=lw*0.6, alpha=0.8)
        axins_disk.plot(d_spec.wave[idx], m_spec.flux_bb[idx] * d_spec.flux_unit_factor, color=colors[target]['model'], lw=lw*1.7, alpha=0.8, ls='--')
        # axins_disk.plot(d_spec.wave[idx],m_spec.flux[idx] -  m_spec.flux_bb[idx], color=colors[target]['model'], lw=lw, alpha=0.8, ls='--')
        
ax[0].set_ylim(1e-16*y_factor, 2.5e-14*y_factor)
ax[0].set_xlim(920, 5300)
ax[1].axhline(0, color='k', lw=0.5)
# make ylims for residuals symmetric
ylim = ax[1].get_ylim()
ylim_s = max(abs(ylim[0]), abs(ylim[1]))
# ax[1].set_ylim(-ylim_s, ylim_s)
ax[1].set_ylim(-0.15, 0.15)
# add label to the y axis
y_label = r'$F_{\lambda}$' '  / 10$^{14}$ ' 'erg ' r'$\text{s}^{-1} \text{cm}^{-2} \text{nm}^{-1}$'
ax[0].set_ylabel(y_label)
ax[1].set_ylabel(r'$\Delta F_{\lambda} / F_{\lambda}$')

# axins_disk.set_ylabel(y_label)
axins_disk.set_xlabel(r'Wavelength / nm')
# add common xlabel 
fig.text(0.5, -0.53, r'Wavelength / nm', ha='center', va='center')


legend_elements_twa28 = [
    Line2D([0], [0], color=colors['TWA28']['data'], lw=lw*1.6, label='Data', markevery=10, markersize=0.5),
    Line2D([0], [0], color=colors['TWA28']['model'], lw=lw*1.6, label='Model', ls='-', markevery=10, markersize=0.5),
    Line2D([0], [0], color=colors['TWA28']['model'], lw=lw*2.0, label='BB', ls='--', markevery=10, markersize=0.5)
]

legend_elements_twa27a = [
    Line2D([0], [0], color=colors['TWA27A']['data'], lw=lw*1.6, label='Data', markevery=10, markersize=0.5),
    Line2D([0], [0], color=colors['TWA27A']['model'], lw=lw*1.6, label='Model', ls='-', markevery=10, markersize=0.5),
    Line2D([0], [0], color=colors['TWA27A']['model'], lw=lw*2.0, label='BB', ls='--', markevery=10, markersize=0.5)
]

# Add legends to the plot with shorter handles

legend1 = ax[0].legend(handles=legend_elements_twa27a, title='TWA 27A', loc=(0.13, 0.64), fontsize=10, frameon=False,
                       title_fontproperties={'weight': 'bold', 'size': 11}, handlelength=1.3)
legend2 = ax[0].legend(handles=legend_elements_twa28, title='TWA 28', loc=(0.26, 0.64), fontsize=10, frameon=False,
                       title_fontproperties={'weight': 'bold', 'size': 11}, handlelength=1.3)

# Add the legends to the axes
ax[0].add_artist(legend1)
ax[0].add_artist(legend2)

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

mark_inset(ax[0], axins[0], loc1=1, loc2=2, fc="none", ec="0.5", zorder=-1)
mark_inset(ax[0], axins[1], loc1=1, loc2=2, fc="none", ec="0.5", zorder=-1)
mark_inset(ax[0], axins[2], loc1=1, loc2=2, fc="none", ec="0.5", zorder=-1)

# add text indicating position of lines in each axins
lines = {'Na': [(1141, 1144, 1.4, 1.2, 'Na', 0)],
         'K': [(1177, 1190, 8.8, 7.7, 'Na', 0)],
        #  'CO': [(2294, 2362, 5.3e-15, 4.65e-15, 'CO', 1)],
         '12CO': [(2294, 2298, 5.3, 4.69, 'Na', 1)],
         '13CO': [(2345, 2349, 6.9, 7.4, 'Na', 1)],

         }  # x-position, y-position for text

# Function to create L-shaped annotation
def add_arrow(ax, x, y, x_text, y_text, text, lw=1.0, alpha=0.4, fontsize=10,
              style='Na'):
    assert style in ['Na', 'CO'], f'style must be Na or CO, not {style}'
    
    if style == 'Na':
        # Vertical line
        ax.annotate(
            text,  # Text label
            xy=(x, y),  # Point to annotate
            xytext=(x_text, y_text),
            arrowprops=dict(
                arrowstyle='-[',
                color='black',
                lw=lw,
                alpha=alpha,
                connectionstyle='angle,angleA=0,angleB=90,rad=0'
            ),
            ha='center',  # Horizontal alignment of text
            va='center',  # Vertical alignment of text
            fontsize=fontsize,
            color='black'
        )
    elif style == 'CO':
        
        ax.annotate(
        '',  # Text label
        xy=(x, y),  # Point to annotate
        xytext=(x_text, y_text), 
        arrowprops=dict(arrowstyle='-', color='black', lw=lw,
                        alpha=alpha,
                        connectionstyle='bar,angle=0,fraction=0'),
        
        ha='center',  # Horizontal alignment of text
        va='center',  # Vertical alignment of text
        fontsize=10,
        color='black'
    )
        ax.text(x*0.995, y_text*1.05, text, fontsize=fontsize, color='black', ha='center', va='center')
            
def draw_L(ax, x_start, y_start, x_mid, y_mid, x_end, y_end, close=False,
           text=None, text_loc='left', line_args={}, text_args={}):
    
    color = line_args.pop('color', 'black')
    linewidth = line_args.pop('linewidth', 2)
    alpha = line_args.pop('alpha', 0.8)
    ax.plot([x_start, x_mid], [y_start, y_mid], color=color, linewidth=linewidth, alpha=alpha)  # Horizontal segment
    ax.plot([x_mid, x_end], [y_mid, y_end], color=color, linewidth=linewidth, alpha=alpha)      # Vertical segment
    if close:
        # add another vertical segment to complete [
        ax.plot([x_end, x_end], [y_end, y_start], color=color, linewidth=linewidth, alpha=alpha)
        
    if text is not None:
        assert text_loc in ['left', 'center', 'right']
        x_text = {'left': x_start, 'center': x_mid + (x_end-x_mid)/2, 'right': x_end}[text_loc]
        y_text = y_mid
        xpad = text_args.pop('xpad', 0.01)
        ypad = text_args.pop('ypad', 0.01)
        ax.text(x_text+xpad, y_text+ypad, text, color='black', ha='center', va='center', 
                transform=ax.transData, **text_args)
    return ax

def add_underline(ax, text, x, y, width=None, pad=0.1, **text_kwargs):
    """Add text with a solid line below it
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw to
    text : str
        The text to draw
    x : float
        The x position of the text
    y : float
        The y position of the text
    width : float, optional
        The width of the line. If None, uses the text width
    pad : float, optional
        The padding between text and line in points
    **text_kwargs
        Additional arguments passed to ax.text()
    """
    # Draw the text
    text_obj = ax.text(x, y, text, ha='center', va='bottom', **text_kwargs)
    
    # Get the bbox
    fig = ax.get_figure()
    fig.canvas.draw()
    bbox = text_obj.get_window_extent()
    
    # Convert bbox to data coordinates
    bbox_data = bbox.transformed(ax.transData.inverted())
    
    # Draw the line
    line_y = bbox_data.y0 - pad * (bbox_data.y1 - bbox_data.y0)
    if width is None:
        x0 = bbox_data.x0
        x1 = bbox_data.x1
    else:
        x0 = x - width/2
        x1 = x + width/2
    
    ax.plot([x0, x1], [line_y, line_y], color=text_kwargs.get('color', 'black'),
            linewidth=text_kwargs.get('linewidth', 1.0),
            alpha=text_kwargs.get('alpha', 1.0))
    
    return text_obj

draw_L(axins[0], 1041, 1.52, 1041, 1.44, 1080, 1.44,close=True,
       text='VO', text_loc='center', text_args={'fontsize': 10, 'xpad': 0.00, 'ypad': -0.08},
       line_args={'alpha': 1.0, 'linewidth': 0.8, 'color': 'gray'})

draw_L(axins[0], 1134, 1.35, 1134, 1.27, 1146, 1.27,close=True,
       text='Na', text_loc='center', text_args={'fontsize': 10, 'xpad': 0.00, 'ypad': -0.06},
       line_args={'alpha': 1.0, 'linewidth': 0.8, 'color': 'gray'})
draw_L(axins[0], 1168, 1.30, 1168, 1.22, 1183, 1.22,close=True,
       text='K', text_loc='center', text_args={'fontsize': 10, 'xpad': 13, 'ypad': -0.02},
       line_args={'alpha': 1.0, 'linewidth': 0.8, 'color': 'gray'})

draw_L(axins[1], (2203+2212)/2, 0.65, (2203+2212)/2, 0.6, 2214, 0.60,close=False,
       text='Na', text_loc='center', text_args={'fontsize': 10, 'xpad': 12, 'ypad': 0.0},
       line_args={'alpha': 1.0, 'linewidth': 0.8, 'color': 'gray'})
# draw_L(axins[1], 2203, 0.65, 2203, 0.60, 2212, 0.60,close=True,
#        line_args={'alpha': 1.0, 'linewidth': 0.8, 'color': 'gray'})

draw_L(axins[1], 2293.5, 0.525, 2293.5, 0.46, 2370, 0.46,close=False,
       text=r'$^{12}$CO', text_loc='center', text_args={'fontsize': 10, 'xpad': -23, 'ypad': 0.015},
       line_args={'alpha': 1.0, 'linewidth': 0.8, 'color': 'gray'})

draw_L(axins[1], 2344.5, 0.695, 2344.5, 0.74, 2376, 0.74,close=False,
       text=r'$^{13}$CO', text_loc='center', text_args={'fontsize': 10, 'xpad': -4, 'ypad': 0.015},
       line_args={'alpha': 1.0, 'linewidth': 0.8, 'color': 'gray'})

y_co_mband = 0.137
draw_L(axins[2], 4440, y_co_mband, 4450, y_co_mband, 4500, y_co_mband,close=False,
       text='CO', text_loc='center', text_args={'fontsize': 10, 'xpad': 40, 'ypad': -0.001},
       line_args={'alpha': 1.0, 'linewidth': 0.8, 'color': 'gray'})
draw_L(axins[2], 4530, y_co_mband*1, 4530, y_co_mband*1, 4590, y_co_mband*1,close=False,
       text='',
       line_args={'alpha': 1.0, 'linewidth': 0.8, 'color': 'gray'})
# Add annotations for each line
# for line in lines:
#     for x1, x2, y1, y2, style, axins_id in lines[line]:
#         # Parameters for the L-shaped arrow
#         add_arrow(axins[axins_id], x1, y1, x2, y2, line, style=style)

axins_disk.set_yscale('log')
axins_disk.set_ylim(1e-16*y_factor, 4e-14*y_factor)
axins_disk.set_xlim(920, 5300)

# plot nirspec bands on axins_disk
gratings = dict(g140h=(900, 1900),
                g235h=(1650, 3180),
                g395h=(2890, 5290),
                )
gratings_text = dict(g140h=6e-16 * y_factor,
                     g235h=6e-16 * y_factor,
                     g395h=7e-15 * y_factor)
colors = ['navy', 'green', 'brown']
# use a faded grey for the bands

for band, color in zip(gratings.keys(), colors):
    axins_disk.axvspan(gratings[band][0], gratings[band][1], color=color, alpha=0.12, lw=0)
    xc = gratings[band][0] + (gratings[band][1] - gratings[band][0])/2
    # if band == 'g140h':
    #     xc -= 90
    #     # add_underline(axins_disk, band.upper(), xc, gratings_text[band], 
    #     #              color=color, fontsize=10, fontweight='bold', width=100)
    # else:
    xc_offset = -90 if band == 'g140h' else 0
    axins_disk.text(xc+xc_offset, gratings_text[band], band.upper(), color=color, fontsize=10,
                    ha='center', va='center', fontweight='bold')

# add a,b,c,d,e labels to each panel for easy reference
panels = ['a', 'b', 'c', 'd', 'e', 'f']
import matplotlib.patheffects as pe
pe = pe.withStroke(linewidth=2, foreground='w')
ax[0].text(0.02, 0.08, 'a', transform=ax[0].transAxes, fontsize=12, fontweight='bold',
            ha='left', va='top', path_effects=[pe])
ax[1].text(0.02, 0.35, 'b', transform=ax[1].transAxes, fontsize=12, fontweight='bold',
            ha='left', va='top', path_effects=[pe])
axins_disk.text(0.02, 0.15, 'c', transform=axins_disk.transAxes, fontsize=12, fontweight='bold',
            ha='left', va='top', path_effects=[pe])

for i, axx in enumerate([axins[0], axins[1], axins[2]]):
    xtext = 0.05
    ytext = 0.1
    axx.text(xtext, ytext, panels[i+3], transform=axx.transAxes, fontsize=12, fontweight='bold',
            ha='left', va='top', path_effects=[pe])

# axins[1].axvline(2345, color='red', lw=0.5)
# fig_name = path / 'twx_figs/fig1_spec.pdf'
fig_name = path_figures / 'fig1_spec_full_range.pdf'

fig.savefig(fig_name, bbox_inches='tight')
plt.close()
# close all
plt.close('all')
print(f'Saved figure to {fig_name}')
