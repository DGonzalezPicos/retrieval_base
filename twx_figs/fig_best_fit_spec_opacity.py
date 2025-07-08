import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.patheffects as pe
pe_white = [pe.withStroke(linewidth=8.0, foreground='white'), pe.Normal()]

from retrieval_base.retrieval import Retrieval
import retrieval_base.auxiliary_functions as af
from retrieval_base.config import Config
import h5py
from pathlib import Path
path = af.get_path(return_pathlib=True)
path_figures = Path('/home/dario/phd/twa2x_paper/figures')

config_file = 'config_jwst.txt'
w_set='NIRSpec'

# set global font size
plt.rcParams.update({'font.size': 12})


def load_data(target, run):
    cwd = os.getcwd()
    if target not in cwd:
        os.chdir(f'{path}/{target}')
        print(f'Changed directory to {target}')

    conf = Config(path=path, target=target, run=run)(config_file)  
    opacity_labels = {k[4:] : v[0][-1] for k,v in conf.opacity_params.items()}
    opacity_labels = {k:v.replace("\\log\\ ", "") for k,v in opacity_labels.items()} 
        
    m_spec = af.pickle_load(f'{conf.prefix}data/bestfit_m_spec_NIRSpec.pkl')
    m_spec.flux = m_spec.flux.squeeze()
    d_spec = af.pickle_load(f'{conf.prefix}data/d_spec_NIRSpec.pkl')
    
    Cov = af.pickle_load(f'{conf.prefix}data/bestfit_Cov_NIRSpec.pkl')
    LogLike = af.pickle_load(f'{conf.prefix}data/bestfit_LogLike_NIRSpec.pkl')
    err = np.nan * np.ones_like(d_spec.flux)
    
    chem = af.pickle_load(f'{conf.prefix}data/bestfit_Chem.pkl')
    
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

    apply_flux_factor = False
    if apply_flux_factor:
        print(f'Dividing flux by factor {flux_factor}')

        m_spec.flux /= flux_factor
        d_spec.flux /= flux_factor
        d_spec.err /= flux_factor
    print(f'Flux factor: {flux_factor}')
    
    d_spec.VMRs = chem.VMRs
    d_spec.pRT_name_dict = chem.pRT_name_dict
    d_spec.pRT_name_dict_r = chem.pRT_name_dict_r
    d_spec.opacity_labels = opacity_labels
    return d_spec, m_spec


def fig_ax():
    fig, ax = plt.subplots(2,1, figsize=(12,5), sharex=True, gridspec_kw={'height_ratios':[3,1]},
                           tight_layout=True)
    return fig, ax

def plot_chunk(d_spec, m_spec, ax=None, idx=0, relative_residuals=False, colors=None, offset=0.0, ls='-', lw=1.4, new_fig=False,
               color_residuals=None, ylim_p=None, target=None):
    
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
    m_flux[nans] = np.nan
    m_flux_nans = np.where(~nans, np.nan, m_flux)
    print(f'idx={idx} cenwave={np.nanmedian(wave)}')
    # if idx in np.arange(18)
    print(f'Setting flux to nan for {idx}')
    # flux[:100] = np.nan
    # m_flux[:100] = np.nan
    if idx in [5,11]: # manual fix for edge of grating issues (wavelength?)
        flux[-29:] = np.nan
        m_flux[-29:] = np.nan
        
    # if idx in [10]:
    #     print(f'Setting flux to nan for {idx}')
    #     flux[-200:] = np.nan
    #     err[-200:] = np.nan
    #     m_flux[-200:] = np.nan
    ax[0].plot(wave, flux, color=colors['data'], lw=lw, alpha=0.8, ls=ls)
    ax[0].plot(wave, flux, color=colors['data'], ls='none', marker='o', ms=1.2, alpha=0.8)
    ax[0].fill_between(wave, flux - err, flux + err, color=colors['data'], alpha=0.2, lw=0.0)
    ax[0].plot(wave, m_flux, color=colors['model'], lw=lw, alpha=0.8, ls=ls, label=target)
    # ax[0].plot(wave, m_flux_nans, color='red', lw=lw, alpha=0.8, ls=ls)
    
    res = flux - m_flux
    if relative_residuals:
        res_norm = res / flux
        err_norm = err / flux
        
    MAD = np.nanmedian(np.abs(res_norm))
    
    color_residuals = colors['model'] if color_residuals is None else color_residuals
    # ax[1].plot(wave, res_norm, color=color_residuals, lw=lw, alpha=0.4)
    # ax[1].scatter(wave, res_norm, color=color_residuals, s=3, alpha=0.8)
    ax[1].scatter(wave, res_norm, color=colors['data'], s=3, alpha=0.8)
    ax[1].fill_between(wave, -err_norm, err_norm, color=colors['model'], alpha=0.3, lw=0.0)
    ax[1].axhline(0.0,color=colors['data'], lw=0.7)
    # if new_ax:
    
    ax[1].set_xlabel('Wavelength / nm')
    # ax[0].set_ylabel('Flux / erg/s/cm2/nm') 
    y_label = r'$F_{\lambda}$' '  / 10$^{14}$ ' 'erg ' r'$\text{s}^{-1} \text{cm}^{-2} \text{nm}^{-1}$'
    ax[0].set_ylabel(y_label)
    if ylim_p is not None:
        p = np.nanpercentile(flux, ylim_p)
        ax[0].set_ylim(p[0], p[1])
    
    res_label = r'$\Delta F / F$' if relative_residuals else r'$\Delta F / erg/s/cm^2/nm$'
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

species_colors = {
    # Major species
    'H2O': '#1f77b4',    # Blue
    # 'CH4': '#9467bd',    # Purple
    'CH4': '#58004f', # dark purple
    'CO2': '#1a5460',    # Green grey
    '12CO': '#2ca02c',     # Green
    '13CO': '#f72585',     # Red
    'C18O': '#ff7f0e',     # Orange
    
    # Metal hydrides
    'FeH': '#9467bd',    # Purple
    'CrH': '#8c564b',    # Brown
    'AlH': '#e377c2',    # Pink
    'NaH': '#7f7f7f',    # Gray
    
    # Metal oxides
    'TiO': '#105e67',    # Dark cyan
    'VO': '#bcbd22',     # Yellow-green
    'AlO': '#ff9896',    # Light red
    'SiO': '#d62728',    # Red
    
    # Other molecules
    'H2S': '#c5b0d5',    # Lavender
    'OH': '#FDC416',     # Light orange
    'HF': 'deepskyblue', # Light blue
    'HCl': '#c49c94',    # Light brown
    
    # Atomic species
    # 'Na': '#ff00cc',
    'Na':'k',
    'K': '#3ade07',      # Magenta
    'Fe': '#e41a1c',     # Red
    'Al': '#377eb8',     # Blue
    'Ti': '#ff4bab',     # Green
    'V': '#984ea3',      # Purple
    'Ca': '#a65628',     # Brown
    'Mg': '#f781bf',     # Pink
    'Si': '#999999'      # Gray
}
species_to_plot = list(species_colors.keys())

colors = dict(TWA28={'data':'k', 
                    #  'model':'orange', 
                    'model':'#D55E00',
                     'crires': 'brown'
                     },
                    # 'crires': '#CC79A7'
              TWA27A={
                    # 'data':'#733b27',
                    'data':'black',
                    #   'model':'#0a74da',
                    'model':'#009E73',
                      },
              )
runs_all = dict(
            TWA28='freeslab_lbl10_G1G2G3_1',
            TWA27A='freeslab_lbl10_G1G2G3_1',
            )

d_specs, m_specs = {}, {}
for target in runs_all.keys():
    
    try:
        d_specs[target], m_specs[target] = load_data(target, runs_all[target])
    except Exception as e:
        print(f'Error loading data for {target}: {e}')
        continue
    
runs = {k:v for k,v in runs_all.items() if k in d_specs.keys()}

def plot_idx(idx, fig=None, ax=None, ylim_p=None, ylim=None, targets=None):  
    new_fig = False
    if fig is None or ax is None:
        fig, ax = fig_ax()
        new_fig = True
        ax[0].set_ylim(ylim[0], ylim[1])
        
    if targets is None:
        targets = runs.keys()
    for target in targets:
        d_spec, m_spec = d_specs[target], m_specs[target]
    # assert len(ax) == 2, f'ax must be a list of 2 elements, not {len(ax)}'
        _, wave, flux, err = plot_chunk(d_spec, m_spec, ax=ax, relative_residuals=True, 
                                        idx=idx,
                                        colors=colors[target],
                                        new_fig=new_fig,
                                        color_residuals=colors[target]['model'],
                                        ylim_p=ylim_p,
                                        target=target)
        
    return wave, flux, err


out_path = Path('/home/dario/phd/retrieval_base/twx_figs/')
out_path_opacities = out_path / 'opacities'

def load_opacities(grating='g140h'):
    # load opacity file
    h5_name = out_path_opacities / f'NIRSpec_{grating}_T2400.h5'
    with h5py.File(h5_name, 'r') as opacity_file:

        opas = opacity_file['opacities'][:]
        opas_wave = opacity_file['wavelength'][:]
        opas_species = list(opacity_file['species'][:].astype(str))

    return opas, opas_wave, opas_species


wave_range_gratings = dict(
                   g140h= (970.0, 1830.0),
                   g235h= (1630.0, 3060.0),
                   g395h= (2840.0, 5300.0),
                   )

opas_grating = {g:load_opacities(g) for g in wave_range_gratings.keys()}

def plot_band(segments, grating, fig=None, ax=None, res_ylim=None, show_ylabel=True, target='TWA28'):
    # starget = runs.keys()[0]
    # cs = custom_settings[band]
    # segments = cs['segments']
    opas, opas_wave, opas_species = opas_grating[grating]

    wmin, wmax = [], []
    ymin, ymax = [], []
    fluxes = []
    for idx in segments:
        wave, flux, err = plot_idx(idx, fig=fig, ax=ax, targets=[target])
        nans = np.isnan(flux)

        wmin.append(np.nanmin(wave[~nans]))
        wmax.append(np.nanmax(wave[~nans]))
        fluxes.append(flux)
    # 
    
    # ymin.append(np.nanmin(flux))
    # ymax.append(np.nanmax(flux))
    
    wmin = np.nanmin(wmin)
    wmax = np.nanmax(wmax)
    # ymin = np.nanmin(ymin)
    # ymax = np.nanmax(ymax)
    median, std = np.nanmedian(fluxes), np.nanstd(fluxes)
    ymin = min(median - 1.2*std, np.nanmin(fluxes))
    ymax = max(median + 1.2*std, np.nanmax(fluxes))
    
    xlim_pad = 0.01*(wmax-wmin)
    ax[0].set_xlim(wmin-xlim_pad, wmax+xlim_pad)
    ax[1].set_xlim(wmin-xlim_pad, wmax+xlim_pad)
    ax[0].set_ylim(ymin*0.4, ymax*1.1)
    # ax[1].set_ylim(-0.15, 0.15)
    if res_ylim is not None:
        ax[1].set_ylim(res_ylim[0], res_ylim[1])
        
    # species_to_plot = species_to_plot_band[band]
    ax_opas = ax[0].twinx()
    # for s, species in enumerate(species):

    vmr_species_median = {k:np.nanmedian(v) for k,v in d_specs[target].VMRs.items()}
    # sort by median vmr, descending    
    vmr_species_median = dict(sorted(vmr_species_median.items(), key=lambda item: item[1], reverse=True))

    wopas_max_thresh = 1e-3
    wopas_dict = {}
    for s, species in enumerate(vmr_species_median.keys()):
        
        line_species = d_specs[target].pRT_name_dict_r[species]
        if species not in species_to_plot:
            
            continue
        if line_species not in opas_species:
            print(f'{species} not in opas_species')
            continue
        vmr_s = np.nanmedian(d_specs[target].VMRs[species])
        s_idx = opas_species.index(line_species)
        # print(f'{species} {s_idx}')
        # color_s = next(deep_palette)
        color_s = species_colors[species]
        wopas = opas[s_idx,:] * vmr_s
        opas_mask = (opas_wave >= wmin) & (opas_wave <= wmax)

        wopas_max = np.nanmax(wopas[opas_mask])
        print(f' {species} {wopas_max:.2e}')
        if wopas_max < wopas_max_thresh:
            # print(f' Skipping {species}... low opacity {wopas_max:.2e}')
            continue
        wopas_dict[species] = wopas
        
    # sort dictionary by max value, descending
    wopas_dict = dict(sorted(wopas_dict.items(), key=lambda item: np.nanmax(item[1]), reverse=True))
        
    for species, wopas in wopas_dict.items():
        label = d_specs[target].opacity_labels[species]
        zorder = -1 if species in ['H2O','AlO'] else 1
        ax_opas.plot(opas_wave[opas_mask], wopas[opas_mask], label=f'{label}', alpha=0.8, color=species_colors[species], zorder=zorder)
        # fill between x-axis and line
        ax_opas.fill_between(opas_wave[opas_mask], 0.0, wopas[opas_mask], color=species_colors[species], alpha=0.2, zorder=zorder)

    # decrease spacing of legend items
    ax_opas.legend(ncol=len(wopas_dict), handlelength=1.0, fontsize=12,
                   framealpha=0.4, 
                   )

    ax_opas.set(yscale='log')
    if show_ylabel:
        ax_opas.set_ylabel('Weighted opacity' +r' / cm$^2$ g$^{-1}$')
    ax_opas.set_ylim(1e-5, 1e12)
    


# nb1 = 4 # 18/2
# # nb1 = 0
# nb2 = 9
# # nb2 = 4
# nb = nb2-nb1
def main():
    """Main function to generate spectral opacity figures for both targets"""
    
    # Define output path
    twx_paper = Path('/home/dario/phd/twa2x_paper/figures')
    
    # Define spectral regions (nb1, nb2) to plot
    spectral_regions = [(0, 4), (4, 9)]
    
    # Define gratings for each spectral segment
    gratings = ['g140h']*3 + ['g235h']*3 + ['g395h']*3
    
    # Define custom y-limits for each spectral segment  
    custom_ylims = [
        (0.66, 2.66),
        (0.45, 2.30),
        (0.30, 1.70),
        (0.15, 1.60),
        (0.05, 0.86),  # 12CO bandhead
        (0.12, 0.40), 
        (0.16, 0.30),
        (0.05, 0.25),
        (0.06, 0.13),
    ]
    
    # Offset factors for TWA27A y-limits
    # offset_ylims = np.linspace(0.62, 1.0, len(custom_ylims))[::-1]
    offset_ylims = np.ones(len(custom_ylims))
    offset_ylims[3:] = np.linspace(0.78, 1.0, len(custom_ylims[3:]))[::-1]
    
    # Loop over each target
    for target in runs_all.keys():
        if target not in d_specs:
            print(f'Skipping {target} - data not loaded')
            continue
            
        print(f'Plotting {target} with run {runs_all[target]}')
        
        # Loop over spectral regions
        for nb1, nb2 in spectral_regions:
            nb = nb2 - nb1
            nb_range = np.arange(nb1, nb2)
            
            print(f'  Creating figure for spectral region {nb1}-{nb2}')
            
            # Create figure with subplots
            fig, axes = plt.subplots(nb*2, 1, 
                                    figsize=(12, 3.5*nb), 
                                    sharex=False,
                                    gridspec_kw={'height_ratios': [3, 1]*nb,
                                                'hspace': 0.3})
            
            

            # Separate spectrum and residual axes
            ax_spec = axes[0::2]
            ax_res = axes[1::2]
            
            ax_spec[0].text(0.02, 0.92 if nb1 == 0 else 0.72,
                            f'{target.replace("TWA", "TWA ")}', 
                         transform=ax_spec[0].transAxes,
                            ha='left', va='top', fontsize=14,
                            color='black',
                            weight='bold',
                            bbox=dict(facecolor='none', 
                                      alpha=1.0, edgecolor='k',
                                    
                            )
            )
            
            # Apply y-limit offsets for TWA27A
            target_ylims = custom_ylims.copy()
            if target == 'TWA27A':
                for i, nb_i in enumerate(nb_range):
                    target_ylims[nb_i] = (custom_ylims[nb_i][0] * offset_ylims[nb_i], 
                                         custom_ylims[nb_i][1] * offset_ylims[nb_i])
            
            # Plot each spectral segment
            for i, nb_i in enumerate(nb_range):
                
                show_ylabel = (i == (nb2-nb1)//2)
                
                # Plot the spectral band with opacities
                plot_band(segments=[nb_i*2, nb_i*2+1], 
                         grating=gratings[nb_i], 
                         fig=fig, 
                         ax=[ax_spec[i], ax_res[i]], 
                         show_ylabel=show_ylabel, 
                         target=target)
                
                # Remove x-axis labels for all but the last subplot
                if nb_i < nb2-1:
                    ax_spec[i].set_xlabel('')
                    ax_res[i].set_xlabel('')
                
                # Remove y-axis labels for non-middle subplots
                if not show_ylabel:
                    ax_spec[i].set_ylabel('')
                    ax_res[i].set_ylabel('')
                
                # Set custom y-limits for this segment
                ax_spec[i].set_ylim(target_ylims[nb_i][0], target_ylims[nb_i][1])
            
            
            
            # Save figure
            pdf_name = twx_paper / f'fig_spec_opacities_{nb1}_{nb2}_{target}.pdf'
            fig.savefig(pdf_name, bbox_inches='tight')
            print(f'  Saved {pdf_name}')
            plt.close(fig)
    
    print('All figures generated successfully!')


if __name__ == '__main__':
    main()



