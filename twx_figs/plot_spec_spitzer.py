import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
from retrieval_base.retrieval import Retrieval
import retrieval_base.auxiliary_functions as af
from retrieval_base.config import Config


path = af.get_path(return_pathlib=True)
path_figures = pathlib.Path('/home/dario/phd/twa2x_paper/figures')

config_file = 'config_jwst.txt'
w_set='NIRSpec'


def load_data(target, run):
    cwd = os.getcwd()
    if target not in cwd:
        os.chdir(f'{path}/{target}')
        print(f'Changed directory to {target}')

    conf = Config(path=path, target=target, run=run)(config_file)        
        
    m_spec = af.pickle_load(f'{conf.prefix}data/bestfit_m_spec_NIRSpec.pkl')
    m_spec.flux = m_spec.flux.squeeze()
    d_spec = af.pickle_load(f'{conf.prefix}data/d_spec_NIRSpec.pkl')
    
    Cov = af.pickle_load(f'{conf.prefix}data/bestfit_Cov_NIRSpec.pkl')
    LogLike = af.pickle_load(f'{conf.prefix}data/bestfit_LogLike_NIRSpec.pkl')
    err = np.nan * np.ones_like(d_spec.flux)
    
    for i in range(d_spec.n_orders):
        for j in range(d_spec.n_dets):
            # mask_i = d_spec.mask_isfinite[i,]
            mask_ij = d_spec.mask_isfinite[i,j]

            if Cov is not None:
                err_ij = Cov[i,j].get_err(mask=mask_ij)
            else:
                err_ij = d_spec.err[i,j]
                    
            beta_ij = LogLike.beta[i,j]
            err_ij *= beta_ij # optimal uncertainty scaling
            err[i,j,:] = err_ij
        
    flux_factor = conf.config_data['NIRSpec'].get('flux_unit_factor', 1.0)
    
    d_spec.err = err
    d_spec.squeeze()
    m_spec.flux.squeeze()

    m_spec.flux /= flux_factor
    d_spec.flux /= flux_factor
    d_spec.err /= flux_factor
    return d_spec, m_spec

def fig_ax():
    fig, ax = plt.subplots(2,1, figsize=(14,5), sharex=True, gridspec_kw={'height_ratios':[3,1]},
                           tight_layout=True)
    return fig, ax

def plot_chunk(d_spec, m_spec, ax=None, idx=0, relative_residuals=False, colors=None, offset=0.0, ls='-', lw=1.4, new_fig=False,
               color_residuals=None, ylim_p=None):
    
    new_ax = (ax is None)
    if new_ax:
        fig, ax = plt.subplots(2,1, figsize=(10,5), sharex=True)
    else:
        assert len(ax) == 2, f'ax must be a list of 2 elements, not {len(ax)}'
        
    wave = d_spec.wave[idx]
    flux = d_spec.flux[idx] + offset
    err = d_spec.err[idx]
    nans = np.isnan(flux)
    m_flux = m_spec.flux[idx] + offset
    m_flux_nans = np.where(~nans, np.nan, m_flux)
    
    ax[0].plot(wave, flux, color=colors['data'], lw=lw, alpha=0.8, ls=ls)
    ax[0].plot(wave, flux, color=colors['data'], ls='none', marker='o', ms=1, alpha=0.8)
    ax[0].fill_between(wave, flux - err, flux + err, color=colors['data'], alpha=0.2, lw=0.0)
    ax[0].plot(wave, m_flux, color=colors['model'], lw=lw, alpha=0.8, ls=ls)
    # ax[0].plot(wave, m_flux_nans, color='red', lw=lw, alpha=0.8, ls=ls)
    
    res = flux - m_flux
    if relative_residuals:
        res = res / flux
        err = err / flux
        
    MAD = np.nanmedian(np.abs(res))
    
    color_residuals = colors['model'] if color_residuals is None else color_residuals
    ax[1].plot(wave, res, color=color_residuals, lw=lw, alpha=0.4)
    ax[1].scatter(wave, res, color=color_residuals, s=1, alpha=0.8)
    ax[1].fill_between(wave, -err, err, color=colors['model'], alpha=0.3, lw=0.0)
    ax[1].axhline(0.0,color=colors['data'], lw=0.7)
    # if new_ax:
    
    ax[1].set_xlabel('Wavelength / nm')
    # ax[0].set_ylabel('Flux / erg/s/cm2/nm') 
    if ylim_p is not None:
        p = np.nanpercentile(flux, ylim_p)
        ax[0].set_ylim(p[0], p[1])
    
    # res_label = r'$\Delta F / F$' if relative_residuals else r'$\Delta F / erg/s/cm^2/nm$'
    res_label = '(Flux - Model)\n/ Flux' if relative_residuals else r'$\Delta F / erg/s/cm^2/nm$'
    ax[1].set_ylabel(res_label)
    
    if new_fig:
        
    
        ylim = - MAD*6.0, MAD*6.0
        ax[1].set_ylim(ylim)
        ax[1].text(0.02, 0.95, f'MAD = {MAD*100.0:.1f} %', transform=ax[1].transAxes,
                ha='left', va='top', fontsize=12)
        xlim = np.nanmin(wave[~nans])-2.0, np.nanmax(wave[~nans])+2.0
        ax[0].set_xlim(xlim)
        ax[1].set_xlim(xlim)
    
    # ax[1].axhline(0.0,color=colors['model'], lw=0.7)
        # ax.set_title(f'Chunk {idx}')
        # plt.show()
    return ax, wave, flux, err

colors = dict(TWA28={'data':'k', 'model':'#D55E00'},
              TWA27A={'data':'k', 'model':'#009E73'})


# runs = dict(TWA28='lbl11_G2G3_fastchem_GP_0',
#             TWA27A='lbl11_G2G3_fastchem_GP_0',
#             )
runs = dict(TWA28='freeslab_lbl10_G2G3_1',
            TWA27A='freeslab_lbl10_G2G3_2',
            )

d_specs, m_specs = {}, {}
for target in runs.keys():
    d_specs[target], m_specs[target] = load_data(target, runs[target])


def plot_idx(idx, fig=None, ax=None, ylim_p=None, ylim=None):  
    new_fig = False
    if fig is None or ax is None:
        fig, ax = fig_ax()
        new_fig = True
        ax[0].set_ylim(ylim[0], ylim[1])
    for t, target in enumerate(runs.keys()):
        d_spec, m_spec = d_specs[target], m_specs[target]
    # assert len(ax) == 2, f'ax must be a list of 2 elements, not {len(ax)}'
        _, wave, flux, err = plot_chunk(d_spec, m_spec, ax=ax[[t,-1]], relative_residuals=True, 
                                        idx=idx,
                                        colors=colors[target],
                                        new_fig=new_fig,
                                        color_residuals=colors[target]['model'],
                                        ylim_p=ylim_p)
         
# fig, ax = fig_ax()
fig, axes = plt.subplots(4,1, figsize=(10,5), sharex=False, gridspec_kw={'height_ratios':[0.5,3,3,1]})
ax = axes[1:]  # Use axes[1:] for the data plots    


# Clean up the label axes (axes[0])
axes[0].set_xticks([])
axes[0].set_yticks([])
for spine in axes[0].spines.values():
    spine.set_visible(False)

xlim = np.nanmin(d_specs['TWA28'].wave), np.nanmax(d_specs['TWA28'].wave)
ymin = np.nanmin([d_specs[t].flux for t in runs.keys()])
ymax = np.nanmax([d_specs[t].flux for t in runs.keys()])
# ax[0].set_xlim(xlim[0], 15e3)
# ax[0].set_ylim(1e-17, ymax)

for idx in range(d_specs['TWA28'].flux.shape[0]):
    plot_idx(idx, fig=fig, ax=ax)
    
    
run_spitzer = 'spitzer_G2G3'
target = 'TWA28'
prefix = '/home/dario/phd/retrieval_base'

for t, target in enumerate(runs.keys()):
    spitzer = np.load(f'{prefix}/{target}/retrieval_outputs/{run_spitzer}/test_data/spitzer_model.npy')
    d_spec_spitzer = af.pickle_load(f'{prefix}/{target}/retrieval_outputs/{run_spitzer}/test_data/d_spec_NIRSpec.pkl')
    flux_unit_factor = d_spec_spitzer.flux_unit_factor
    wave_full = spitzer[0,:,:].flatten()
    bb_full = spitzer[3,:,:].flatten()

    wave, flux, err, bb, model_flux = spitzer[:,-1,:]
    # model_flux /= d_specs[target].flux_unit_factor
    model_flux /= flux_unit_factor # UPDATE 2025-07-02
    model_flux[:1] = np.nan
    ax[t].plot(wave, flux, color='k', marker='o', ms=2, alpha=0.8, ls='none', label='Observations')
    ax[t].plot(wave, model_flux, color=colors[target]['model'], lw=1.8, alpha=0.8, ls='-', label='Full model')
    ax[t].plot(wave, model_flux - bb, color=colors[target]['model'], lw=1.8, alpha=0.8, ls='--', label='Atmosphere')
    ax[t].plot(wave_full, bb_full, color='brown', lw=1.8, alpha=0.8, ls='--', label='Blackbody')


    # ax[t].plot(wave, bb, color=colors[target]['model'], lw=1.8, alpha=0.8, ls=':')
    ax[-1].plot(wave, (flux - model_flux) / flux, color=colors[target]['model'], marker='o', ms=2, alpha=0.8, ls='none')
    
    ax[t].legend(loc='upper right', frameon=False, ncol=2)
    ax[t].set(yscale='log')
    ax[t].set_ylim(1e-17,2e-14)
# plt.show()

# Draw instrument labels with arrows in the top subplot
nirspec = (xlim[0], 5.3e3)  # Convert to nm
spitzer = (5.3e3, 15e3)
axes[0].set_xlim(xlim[0], 15e3)

# Arrow style
arrow_props = dict(
    arrowstyle='<->', 
    color='k',
    lw=1.5,
    shrinkA=0,
    shrinkB=0
)

# Text style
text_props = dict(
    va='center',  # Changed to center for better alignment
    ha='center',
    fontsize=12,
    weight='bold',
)

# Y position for arrows and text (in axes coordinates)
arrow_y = 0.0
text_y = 0.5

# Draw NIRSpec range in axes[0]
axes[0].annotate('', 
                xy=(nirspec[0], arrow_y), 
                xytext=(nirspec[1], arrow_y),
                arrowprops=arrow_props)
axes[0].text(np.mean(nirspec), text_y, 'JWST/NIRSpec', **text_props)

# Draw Spitzer range in axes[0]
axes[0].annotate('',
                xy=(spitzer[0], arrow_y),
                xytext=(spitzer[1], arrow_y),
                arrowprops=arrow_props)
axes[0].text(np.mean(spitzer), text_y, 'Spitzer/IRS', **text_props)

# Set the x limits for the label subplot
xlim_axes = (xlim[0], 15e3)
axes[-1].set_xlim(xlim_axes)

xscale_log = False
if xscale_log:
    axes[-1].set_xscale('log')
    # change xticks to scientific notation
    xticks = [2000, 3000, 4000, 6000, 10000, 15000]
    axes[-1].set_xticks(xticks)
    axes[-1].set_xticklabels([f'{x:.0f}' for x in xticks])
    
ax[-1].set_ylim(-0.5, 0.5)

yticks = [-0.40 , 0.0, 0.40]
axes[-1].set_yticks(yticks)
axes[-1].set_yticklabels([f'{y:.1f}' for y in yticks])

targets = ['TWA 28', 'TWA 27A']
for i in range(2):
    ax[i].set_xlim(xlim_axes)
    ax[i].set_xticks([])
    ax[i].text(0.10, 0.95, targets[i], transform=ax[i].transAxes, fontsize=12, ha='left', va='top', weight='bold')
    # remove small xticks too
    if xscale_log:
        ax[i].set_xscale('log')
        ax[i].set_xticks([], minor=True)

    

# add xticks back for axes[-1]
# xticks = np.arange(2000, 15000, 1000)
# axes[-1].set_xticks(xticks)

# add common ylabel for axes[1,2]
fig.text(0.06, 0.5, r'Flux / erg s$^{-1}$ cm$^{-2}$ nm$^{-1}$', ha='center', va='center', rotation=90)
fig_name = path_figures /'fig_spec_spitzer.pdf'
plt.savefig(fig_name, bbox_inches='tight')
print(f'Saved figure to {fig_name}')
plt.close()