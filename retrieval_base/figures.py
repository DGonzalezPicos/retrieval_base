import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
from scipy.ndimage import generic_filter, gaussian_filter1d
from scipy.optimize import curve_fit

import os
import copy
import corner

import petitRADTRANS.nat_cst as nc

import retrieval_base.auxiliary_functions as af
from retrieval_base.spline_model import SplineModel

# make borders thicker
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['xtick.minor.width'] = 1.5
mpl.rcParams['ytick.minor.width'] = 1.5


def fig_order_subplots(n_orders, ylabel, xlabel=r'Wavelength (nm)'):

    fig, ax = plt.subplots(
        figsize=(10,2.8*n_orders), nrows=n_orders, 
        gridspec_kw={'hspace':0.22, 'left':0.1, 'right':0.95, 
                     'top':(1-0.02*7/n_orders), 'bottom':0.035*7/n_orders, 
                     }
        )
    if n_orders == 1:
        ax = np.array([ax])

    ax[n_orders//2].set(ylabel=ylabel)
    ax[-1].set(xlabel=xlabel)

    return fig, ax

def fig_flux_calib_2MASS(wave, 
                         calib_flux, 
                         calib_flux_wo_tell_corr, 
                         transm, 
                         poly_model, 
                         wave_2MASS, 
                         transm_2MASS, 
                         tell_threshold=0.2, 
                         order_wlen_ranges=None, 
                         prefix=None, 
                         w_set='', 
                         ):

    fig, ax = plt.subplots(
        figsize=(10,4), nrows=2, sharex=True, 
        gridspec_kw={'hspace':0, 'height_ratios':[1,0.5], 
                     'left':0.1, 'right':0.95, 'top':0.92, 'bottom':0.15, 
                     }
        )
    
    # poly_model /= np.nanmedian(poly_model)
    # poly_model *= np.nanmedian(calib_flux)
    if order_wlen_ranges is not None:
        # Plot zoom-ins of the telluric correction
        n_orders = order_wlen_ranges.shape[0]
        labels = [r'$F_\mathrm{CRIRES}$',
                  r'$F_\mathrm{CRIRES}/T_\mathrm{CRIRES}$',
                  r'$T_\mathrm{CRIRES}$',]
        for i in range(n_orders):
            # Only plot within a wavelength range
            wave_min = order_wlen_ranges[i,:].min() - 0.5
            wave_max = order_wlen_ranges[i,:].max() + 0.5

            mask_wave = np.arange(i*3*2048, (i+1)*3*2048, dtype=int)
            ax[0].plot(wave[mask_wave], calib_flux_wo_tell_corr[mask_wave], c='k', lw=0.5, alpha=0.4, 
                    label=labels[0] if i == 0 else None
                    )
            ax[0].plot(wave[mask_wave], calib_flux[mask_wave], c='k', lw=0.5, 
                    label=labels[1] if i == 0 else None
                    )
            
            ax[1].plot(wave[mask_wave], transm[mask_wave], c='k', lw=0.5, 
                       label=labels[2] if i == 0 else None)
            ax[1].plot(wave[mask_wave], poly_model[mask_wave], c='magenta', lw=1)
            
    ax[0].set(#ylim=(0, 1.5*np.nanpercentile(calib_flux, q=95)), 
              ylabel=r'$F_\lambda\ (\mathrm{erg\ s^{-1}\ cm^{-2}\ nm^{-1}})$', 
              )
    ax[0].legend(loc='upper left')
    
    
    if isinstance(tell_threshold, np.ndarray):
        # ax[1].plot(wave, poly_model, c='magenta', lw=1)
        ax[1].plot(wave, tell_threshold, c='gray', lw=1, ls='--')
    else:
        ax[1].axhline(tell_threshold, c='gray', lw=1, ls='--')
    ax[1].plot(wave_2MASS, transm_2MASS, c='r', lw=1, label=r'$T_\mathrm{2MASS}$')
    ax[1].set(xlim=(wave.min()-20, wave.max()+20), xlabel=r'Wavelength (nm)', 
              ylim=(0,1.1), ylabel=r'Transmissivity'
              )
    ax[1].legend(loc='upper left')

    if prefix is not None:
        plt.savefig(prefix+f'plots/flux_calib_tell_corr_{w_set}.pdf')
    #plt.show()
    plt.close(fig)

    if order_wlen_ranges is not None:
        # Plot zoom-ins of the telluric correction
        n_orders = order_wlen_ranges.shape[0]

        fig, ax = fig_order_subplots(
            n_orders, 
            ylabel=r'$F_\lambda\ (\mathrm{erg\ s^{-1}\ cm^{-2}\ nm^{-1}})$'
            )

        for i in range(n_orders):
            # Only plot within a wavelength range
            wave_min = order_wlen_ranges[i,:].min() - 0.5
            wave_max = order_wlen_ranges[i,:].max() + 0.5

            mask_wave = np.arange(i*3*2048, (i+1)*3*2048, dtype=int)

            ax[i].plot(
                wave[mask_wave], calib_flux_wo_tell_corr[mask_wave], 
                #(calib_flux_wo_tell_corr/poly_model)[mask_wave], 
                c='k', lw=0.5, alpha=0.4
                )
            ax[i].plot(wave[mask_wave], calib_flux[mask_wave], c='k', lw=0.5)

            ax[i].set(xlim=(wave_min, wave_max))
        
        if prefix is not None:
            plt.savefig(prefix+f'plots/tell_corr_zoom_ins_{w_set}.pdf')
        #plt.show()
        plt.close(fig)

def fig_sigma_clip_old(wave, flux, flux_wo_clip, sigma_clip_bounds, order_wlen_ranges, sigma, prefix=None, w_set=''):

    # Plot zoom-ins of the sigma-clipping procedure
    n_orders = order_wlen_ranges.shape[0]

    fig, ax = fig_order_subplots(
        n_orders, 
        ylabel=r'$F_\lambda\ (\mathrm{erg\ s^{-1}\ cm^{-2}\ nm^{-1}})$'
        )
    
    for i in range(n_orders):
        # Only plot within a wavelength range
        wave_min = order_wlen_ranges[i,:].min() - 0.5
        wave_max = order_wlen_ranges[i,:].max() + 0.5
        
        mask_wave = np.arange(i*3*2048, (i+1)*3*2048, dtype=int)

        ax[i].plot(wave[mask_wave], flux_wo_clip[mask_wave], c='r', lw=0.3)
        ax[i].plot(wave[mask_wave], flux[mask_wave], c='k', lw=0.5)

        ax[i].fill_between(
            wave[mask_wave], y1=sigma_clip_bounds[0,i], y2=sigma_clip_bounds[2,i], 
            fc='C0', alpha=0.4, ec='none', label=f'{sigma}'+r'$\sigma$'
            )
        ax[i].plot(wave[mask_wave], sigma_clip_bounds[1,i], c='C0', lw=0.5)

        ax[i].set(xlim=(wave_min, wave_max))

    ax[-1].legend()
    
    if prefix is not None:
        plt.savefig(prefix+f'plots/sigma_clip_zoom_ins_{w_set}.pdf')
    #plt.show()
    plt.close(fig)
    
def fig_sigma_clip(d_spec, clip_mask, fig_name=None):
    
    assert d_spec.flux.shape == clip_mask.shape, f'Shapes of d_spec.flux and flux_clip do not match \
        ({d_spec.flux.shape} vs. {clip_mask.shape})'
    ylabel = r'$F_\lambda\ (\mathrm{erg\ s^{-1}\ cm^{-2}\ nm^{-1}})$'
    fig, ax = fig_order_subplots(d_spec.n_orders, ylabel=ylabel)
    lw = 0.8
    for i in range(d_spec.n_orders):
        for j in range(d_spec.n_dets):
            
            mask = clip_mask[i,j]
            f_clip = np.where(mask, d_spec.flux[i,j], np.nan)
            f_clean  = np.where(~mask, d_spec.flux[i,j], np.nan) 
            ax[i].plot(d_spec.wave[i,j], f_clip, c='r', lw=lw)
            # if overplot_array is not None:
            ax[i].plot(d_spec.wave[i,j], f_clean, c='k', lw=lw, alpha=0.4)
        
        xlim = (d_spec.wave[i,:].min()-0.5, d_spec.wave[i,:].max()+0.5)
        ax[i].set(xlim=xlim)

    if fig_name is not None:
        fig_name = fig_name if fig_name is not None else prefix+f'plots/spec_to_fit_{w_set}.pdf'
        plt.savefig(fig_name)
        print(f' Figure saved as {fig_name}')
        #plt.show()
        plt.close(fig)
    return fig, ax

def fig_spec_to_fit(d_spec, prefix=None, w_set=''):

    ylabel = r'$F_\lambda\ (\mathrm{erg\ s^{-1}\ cm^{-2}\ nm^{-1}})$'
    if d_spec.high_pass_filtered:
        ylabel = r'$F_\lambda$ (high-pass filtered)'

    fig, ax = fig_order_subplots(d_spec.n_orders, ylabel=ylabel)

    for i in range(d_spec.n_orders):
        for j in range(d_spec.n_dets):
            ax[i].plot(d_spec.wave[i,j], d_spec.flux[i,j], c='k', lw=0.5)
        
        # ax[i].set(xlim=(d_spec.order_wlen_ranges[i].min()-0.5, 
        #                 d_spec.order_wlen_ranges[i].max()+0.5)
        #           )
        xlim = (d_spec.wave[i,:].min()-0.5, d_spec.wave[i,:].max()+0.5)
        ax[i].set(xlim=xlim)

    if prefix is not None:
        plt.savefig(prefix+f'plots/spec_to_fit_{w_set}.pdf')
    #plt.show()
    plt.close(fig)

def fig_bestfit_model(
        d_spec, 
        m_spec, 
        LogLike, 
        Cov, 
        xlabel='Wavelength (nm)', 
        bestfit_color='C1', 
        ax_spec=None, 
        ax_res=None, 
        prefix=None, 
        w_set=''
        ):

    if (ax_spec is None) and (ax_res is None):
        # Create a new figure
        is_new_fig = True
        n_orders = d_spec.n_orders

        fig, ax = plt.subplots(
            figsize=(10,2.5*n_orders*2), nrows=n_orders*3, 
            gridspec_kw={'hspace':0, 'height_ratios':[1,1/3,1/5]*n_orders, 
                        'left':0.1, 'right':0.95, 
                        'top':(1-0.02*7/(n_orders*3)), 
                        'bottom':0.035*7/(n_orders*3), 
                        }
            )
    else:
        is_new_fig = False

    ylabel_spec = r'$F_\lambda$'+'\n'+r'$(\mathrm{erg\ s^{-1}\ cm^{-2}\ nm^{-1}})$'
    if d_spec.high_pass_filtered:
        ylabel_spec = r'$F_\lambda$ (high-pass filtered)'

    # Use the same ylim, also for multiple axes
    ylim_spec = (np.nanmean(d_spec.flux)-4*np.nanstd(d_spec.flux), 
                 np.nanmean(d_spec.flux)+4*np.nanstd(d_spec.flux)
                )
    ylim_res = (1/3*(ylim_spec[0]-np.nanmean(d_spec.flux)), 
                1/3*(ylim_spec[1]-np.nanmean(d_spec.flux))
                )
    N_knots = LogLike.N_knots # number of spline knots for spectrum

    for i in range(d_spec.n_orders):

        if is_new_fig:
            # Spectrum and residual axes
            ax_spec = ax[i*3]
            ax_res  = ax[i*3+1]

            # Remove the temporary axis
            ax[i*3+2].remove()

            # Use a different xlim for the separate figures
            xlim = (d_spec.wave[i,:].min()-0.5, 
                    d_spec.wave[i,:].max()+0.5)
        else:
            xlim = (d_spec.wave.min()-0.5, 
                    d_spec.wave.max()+0.5)

        ax_spec.set(xlim=xlim, xticks=[],
                    # ylim=ylim_spec,
                    )
        ax_res.set(xlim=xlim, ylim=ylim_res)

        for j in range(d_spec.n_dets):
        
            mask_ij = d_spec.mask_isfinite[i,j]
            if np.sum(mask_ij) == 0:
                continue
            # if mask_ij.any():
            # Show the observed and model spectra
            ax_spec.plot(
                d_spec.wave[i,j], d_spec.flux[i,j], 
                c='k', lw=0.5, label='Observation'
                )

            label = 'Best-fit model ' + \
                    r'$(\chi^2_\mathrm{red}$ (w/o $\sigma$-model)$=' + \
                    '{:.2f}'.format(LogLike.chi_squared_red) + \
                    r')$'
                    
            # PLot model (check if spline decomposition used during retrieval)
            # if hasattr(LogLike, 'phi'):
            #         m_flux_spline = SplineModel(N_knots=LogLike.N_knots, spline_degree=3)(m_spec.flux[i,j])
            #         m_flux = LogLike.phi[i,j] @ m_flux_spline
                    
            # else:
                
            #     f = LogLike.phi[i,j]
            #     m_flux = m_spec.flux[i,j] * f
            
            m_flux = LogLike.m[i,j] 
                    
            ax_spec.plot(
                d_spec.wave[i,j], m_flux, 
                c=bestfit_color, lw=1, label=label
                )
            
            if getattr(m_spec, "N_veiling", 0) > 0:
                # print(f' N_knots = {N_knots}')
                m_v = LogLike.phi[i,j,N_knots:] @ m_spec.M_veiling
                m_v[~mask_ij] = np.nan
                ax_spec.plot(d_spec.wave[i,j], m_v, c='magenta', lw=1, label='Veiling')

                # m_pRT = m_spec.flux[i,j][None,:]
                m_pRT = SplineModel(N_knots=N_knots, spline_degree=3)(m_spec.flux[i,j]) if N_knots > 1 else m_spec.flux[i,j][None,:]
                m_pRT[:,~mask_ij] = np.nan
                ax_spec.plot(d_spec.wave[i,j], LogLike.phi[i,j,:N_knots] @ m_pRT, c='orange', lw=1, label='pRT')
            if hasattr(m_spec, 'veiling_model'):
                ax_spec.plot(d_spec.wave[i,j], m_spec.veiling_model[i,j]* np.mean(LogLike.phi[i,j]), 
                             c='magenta', lw=1, label='Veiling model')
                # ax_spec.plot(d_spec.wave[i,j], m_spec.pRT_model[i,j], c='orange', lw=1, label='pRT model')
            if m_spec.flux_envelope is not None:
                ax_spec.plot(
                    d_spec.wave[i,j], m_spec.flux_envelope[3,i,j], c='C0', lw=1
                    )

            # if mask_ij.any():

            # Plot the residuals
            # res_ij = d_spec.flux[i,j] - LogLike.phi[i,j]*m_spec.flux[i,j]
            res_ij = d_spec.flux[i,j] - m_flux
            ax_res.plot(d_spec.wave[i,j], res_ij, c='k', lw=0.5)
            ax_res.plot(
                [d_spec.wave[i,j].min(), d_spec.wave[i,j].max()], 
                [0,0], c=bestfit_color, lw=1
            )

            if m_spec.flux_envelope is not None:
                print(f'[fig_bestfig_model] flux_envelope --> Not implemented')
                ax_res.plot(
                    d_spec.wave[i,j], m_spec.flux_envelope[3,i,j] - LogLike.phi[i,j]*m_spec.flux[i,j], 
                    c='C0', lw=1
                    )

            # Show the mean error
            mean_err_ij = np.mean(Cov[i,j].err)
            ax_res.errorbar(
                d_spec.wave[i,j].min()-0.2, 0, yerr=1*mean_err_ij, 
                fmt='none', lw=1, ecolor='k', capsize=2, color='k', 
                label=r'$\langle\sigma_{ij}\rangle$'
                )

            # Get the covariance matrix
            cov = Cov[i,j].get_dense_cov()
            
            # Scale with the optimal uncertainty-scaling
            cov *= LogLike.s[i,j]**2

            # Get the mean error from the trace
            mean_scaled_err_ij = np.mean(np.diag(np.sqrt(cov)))

            ax_res.errorbar(
                d_spec.wave[i,j].min()-0.4, 0, yerr=1*mean_scaled_err_ij, 
                fmt='none', lw=1, ecolor=bestfit_color, capsize=2, color=bestfit_color, 
                #label=r'$\s_{ij}\langle\sigma_{ij}\rangle$'
                label=r'$\s_{ij}\cdot\langle\mathrm{diag}(\sqrt{\Sigma_{ij}})\rangle$'
                )

            if i==0 and j==0:
                ax_spec.legend(
                    loc='upper right', ncol=2, fontsize=8, handlelength=1, 
                    framealpha=0.7, handletextpad=0.3, columnspacing=0.8
                    )

    # Set the labels for the final axis
    ax_spec.set(ylabel=ylabel_spec)
    ax_res.set(xlabel=xlabel, ylabel='Res.')

    if is_new_fig and (prefix is not None):
        plt.savefig(prefix+f'plots/bestfit_spec_{w_set}.pdf')
        plt.close(fig)
    else:
        return ax_spec, ax_res

def fig_cov(LogLike, Cov, d_spec, cmap, prefix=None, w_set=''):

    all_cov = np.zeros(
        (d_spec.n_orders, d_spec.n_dets, 
         d_spec.n_pixels, d_spec.n_pixels)
        )
    vmax = np.zeros((d_spec.n_orders, d_spec.n_dets))
    for i in range(d_spec.n_orders):
        for j in range(d_spec.n_dets):
            
            # Only store the valid pixels
            mask_ij = d_spec.mask_isfinite[i,j]

            # Get the covariance matrix
            cov = Cov[i,j].get_dense_cov()

            # Scale with the optimal uncertainty scaling
            cov *= LogLike.s[i,j]**2

            # Insert the masked rows into the covariance matrix
            indices = np.arange(0, d_spec.n_pixels, 1)[~mask_ij]
            for idx in indices:
                cov = np.insert(cov, idx, np.zeros(mask_ij.sum()), axis=0)
            for idx in indices:
                cov = np.insert(cov, idx, np.zeros(d_spec.n_pixels), axis=1)

            # Add to the complete array
            all_cov[i,j,:,:] = cov

            # Store the median of the diagonal
            vmax[i,j] = np.median(np.diag(cov))

    # Use a single range in the matshows
    vmin, vmax = 0, 0.3*np.max(vmax)


    fig, ax = plt.subplots(
        figsize=(10*d_spec.n_dets/3, 3.5*d_spec.n_orders), 
        nrows=d_spec.n_orders, ncols=d_spec.n_dets, 
        gridspec_kw={
            'wspace':0.1, 'hspace':0.1, 
            'left':0.08, 'right':0.95, 
            'top':0.95, 'bottom':0.08
            }
        )
    if d_spec.n_orders == 1:
        ax = np.array([ax])

    for i in range(d_spec.n_orders):
        for j in range(d_spec.n_dets):

            extent = [
                d_spec.wave[i,j].min(), d_spec.wave[i,j].max(), 
                d_spec.wave[i,j].max(), d_spec.wave[i,j].min(), 
                ]
            ax[i,j].matshow(
                all_cov[i,j], aspect=1, extent=extent, cmap=cmap, 
                interpolation='none', vmin=vmin, vmax=vmax
                )
            ticks = np.linspace(d_spec.wave[i,j].min()+0.5, d_spec.wave[i,j].max()-0.5, num=4)
            ax[i,j].set_xticks(
                ticks, labels=['{:.0f}'.format(t_i) for t_i in ticks]
                )
            ax[i,j].set_yticks(
                ticks, labels=['{:.0f}'.format(t_i) for t_i in ticks], 
                rotation=90, va='center'
                )
            ax[i,j].tick_params(
                axis='x', which='both', bottom=False, top=True, labelbottom=False
                )
            ax[i,j].grid(True, alpha=0.1)

    ax[-1,1].set(xlabel='Wavelength (nm)')
    ax[d_spec.n_orders//2,0].set(ylabel='Wavelength (nm)')

    if prefix is not None:
        plt.savefig(prefix+f'plots/cov_matrices_{w_set}.pdf')
        plt.close(fig)

    return all_cov


def fig_PT(PT,
            ax=None, 
            ax_grad=None,
            fig=None,
            xlim=None, 
            xlim_grad=None,
            bestfit_color='brown',
            envelopes_color='brown',
            int_contr_em_color='k',
            text_color='gray',
            # weigh_alpha=True,
            show_photosphere=True,
            show_knots=True,
            show_text=True,
            plot_sonora=False,
            fig_name=None,
    ):

    is_new_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7,7))
        is_new_fig = True
        
    assert hasattr(PT, 'temperature_envelopes'), 'No temperature envelopes found'
    
    p = PT.pressure
    # if hasattr(PT, 'int_contr_em'):
    if len(PT.int_contr_em) > 0:
        ax_twin = ax.twiny()

        int_contr_em_color = np.atleast_1d(int_contr_em_color)
        for i, w_set in enumerate(PT.int_contr_em.keys()):
            # Plot the integrated contribution emission
            print(' - Plotting integrated contribution emission')
            ax_twin.plot(
                PT.int_contr_em[w_set], p, 
                # c=int_contr_em_color[i],
                bestfit_color,
                lw=2, alpha=0.4,
                )
            peaf_icf = np.nanmax(PT.int_contr_em[w_set])
            print(f' - Peak integrated contribution emission: {peaf_icf:.2f} at {p[np.argmax(PT.int_contr_em[w_set])]:.2e} bar')

        
            # if weigh_alpha:
            #     af.weigh_alpha(PT.int_contr_em, p, np.linspace(0,10000,p.size), ax, alpha_min=0.5, plot=True)
            # define photosphere as region where PT.int_contr_em > np.quantile(PT.int_contr_em, 0.9)
            if show_photosphere and getattr(PT, 'temperature_envelopes', None) is not None:
                photosphere = PT.int_contr_em[w_set] > np.quantile(PT.int_contr_em[w_set], 0.95)
                P_phot = np.mean(p[photosphere])
                T_phot = np.mean(PT.temperature_envelopes[3][photosphere])
                T_phot_err = np.std(PT.temperature_envelopes[3][photosphere])
                # print(f' - Photospheric temperature: {T_phot:.1f} +- {T_phot_err:.1f} K')
                # make empty marker
                ax.scatter(T_phot, P_phot, c='red',
                            marker='o', 
                            s=50, 
                            alpha=0.5,
                            zorder=10,
                            label=f'{T_phot:.0f} $\pm$ {T_phot_err:.0f} K')
                
                ax.legend(loc='upper right', fontsize=12)

        # remove xticks
        ax_twin.set_xticks([])
        ax_twin.spines['top'].set_visible(False)
        ax_twin.spines['bottom'].set_visible(False)
        ax_twin.set(
            # xlabel='Integrated contribution emission',
            xlim=(0,np.max([ice for ice in PT.int_contr_em.values()])*1.5),
            )
    if hasattr(PT, 'log_P_knots') and show_knots:
        
        for i, log_P_knot in enumerate(PT.log_P_knots):
            ax.axhline(10**log_P_knot, c=text_color, lw=1,
                    ls='-' if i==len(PT.log_P_knots)//2 else '--',
                    alpha=0.8,
                    zorder=0)
            
    
    if PT.temperature_envelopes is not None:     # Plot the PT confidence envelopes
        for i in range(3):
            ax.fill_betweenx(
                y=p, x1=PT.temperature_envelopes[i], 
                x2=PT.temperature_envelopes[-i-1], 
                color=envelopes_color, ec='none', 
                alpha=0.3,
                )
        # Plot the median PT
        ax.plot(PT.temperature_envelopes[3], p, c=bestfit_color, lw=2,)
        xlim = (0, PT.temperature_envelopes[-1].max()*1.06) if xlim is None else xlim
        
    else:
        ax.plot(PT.temperature, p, c=bestfit_color, lw=2)
        xlim = (0, PT.temperature.max()*1.06) if xlim is None else xlim
        
    if hasattr(PT, 'sonora') and plot_sonora:
        try:
            seo = SonoraElfOwl(teff=PT.sonora.get('teff', 2400), log_g=PT.sonora.get('log_g', 3.5))
            seo.load_PT().get_dlnT_dlnP()
            ax_PT = seo.plot_PT(ax=ax, color='magenta', label=f'Sonora\nT={seo.teff}K\nlog g ={seo.log_g:.1f}')
            if ax_grad is not None:
                ax_grad.plot(seo.dlnT_dlnP, seo.pressure, color='magenta')
        except:
            print(' - Could not plot Sonora PT profile')

    # if hasattr(PT, "dlnT_dlnP_envelopes") and ax_grad is not None:
    if ax_grad is not None and hasattr(PT, "dlnT_dlnP_array"):
        if hasattr(PT, "dlnT_dlnP_envelopes"):
            for i in range(3):
                ax_grad.fill_betweenx(
                    y=p, 
                    x1=PT.dlnT_dlnP_envelopes[i],
                    x2=PT.dlnT_dlnP_envelopes[-i-1],
                    color=envelopes_color, ec='none', 
                    alpha=0.3,
                    )
                
        else:
            ax_grad.plot(PT.dlnT_dlnP_array, p, c=bestfit_color, lw=2)
            
        ax_grad.set(xlabel=r'$\nabla_T$',
            ylim=(p.max(), p.min()), yscale='log',
            # xlim=xlim,
            yticks=[],
            )

    ax.set(xlabel='Temperature / K', ylabel='Pressure / bar',
            ylim=(p.max(), p.min()), yscale='log',
            # xlim=(0, None),
            xlim=xlim,
            )
    
    if fig_name is not None:
        fig.savefig(fig_name)
        print(f' - Saved {fig_name}')
    plt.close(fig)

    if is_new_fig:
        return fig, ax
    if ax_grad is not None:
        return ax, ax_grad
    return ax

def old_fig_PT(PT, 
           pRT_atm, 
           #integrated_contr_em=None, 
           #integrated_contr_em_per_order=None, 
           #integrated_opa_cloud=None, 
           ax_PT=None, 
           envelope_colors=None, 
           posterior_color='C0', 
           bestfit_color='C1', 
           ylabel=r'$P\ \mathrm{(bar)}$', 
           yticks=np.logspace(-6, 2, 9), 
           xlim=(1,3500), 
           show_ln_L_penalty=False, 
           prefix=None, 
           contr_em_color={'J1226':'b', 'K2166':'r'}, 
           opa_cloud_color={'J1226':'b', 'K2166':'r'}, 
           ):
    
    if ax_PT is None:
        fig, ax_PT = plt.subplots(
            figsize=(4.5,4.5), 
            gridspec_kw={'left':0.16, 'right':0.94, 
                         'top':0.87, 'bottom':0.15
                         }
            )

        is_new_fig = True
    else:
        is_new_fig = False

    if PT.temperature_envelopes is not None:
        # Plot the PT confidence envelopes
        for i in range(3):
            ax_PT.fill_betweenx(
                y=PT.pressure, x1=PT.temperature_envelopes[i], 
                x2=PT.temperature_envelopes[-i-1], 
                color=envelope_colors[i+1], ec='none', 
                )

        # Plot the median PT
        ax_PT.plot(
            PT.temperature_envelopes[3], PT.pressure, 
            c=posterior_color, lw=1
        )
    
    # Plot the best-fitting PT profile and median
    if show_ln_L_penalty:
        label = r'$\ln\ \mathrm{L\ penalty}=' + \
                '{:.0f}'.format(np.sign(PT.ln_L_penalty)*10) + '^{' + \
                '{:.2f}'.format(np.log10(np.abs(PT.ln_L_penalty))) + '}$'
    else:
        label = None
    
    ax_PT.plot(
        PT.temperature, PT.pressure, c=bestfit_color, lw=1, label=label
        )
    from retrieval_base.PT_profile import PT_profile_free_gradient
    if hasattr(PT, 'T_knots') and not isinstance(PT, PT_profile_free_gradient):
        ax_PT.plot(
            PT.T_knots, PT.P_knots, c=bestfit_color, ls='', marker='o', markersize=3
            )

    if show_ln_L_penalty:
        ax_PT.legend(
            loc='upper right', handlelength=0.5, 
            handletextpad=0.5, framealpha=0.7
            )

    try:
        SONORA_temperature = np.loadtxt(prefix+'data/SONORA_temperature.dat')
        SONORA_RCB = np.loadtxt(prefix+'data/SONORA_RCB.dat').flatten()[0]
        
        ax_PT.plot(SONORA_temperature, PT.pressure, c='k', lw=1)
        ax_PT.plot(
            np.interp(SONORA_RCB, xp=PT.pressure, fp=SONORA_temperature), 
            SONORA_RCB, 'ko'
            )
    except:
        pass
        
    ax_PT.set(
        xlabel=r'$T\ \mathrm{(K)}$', xlim=xlim, 
        ylabel=ylabel, yscale='log', yticks=yticks
        )
    ax_PT.set_ylim(PT.pressure.min(), PT.pressure.max())
    ax_PT.invert_yaxis()

    for w_set in pRT_atm.keys():
        integrated_contr_em = pRT_atm[w_set].int_contr_em
        integrated_contr_em_per_order = None
        integrated_opa_cloud = pRT_atm[w_set].int_opa_cloud

        # Add the integrated emission contribution function
        ax_contr = ax_PT.twiny()
        fig_contr_em(
            ax_contr, 
            integrated_contr_em, 
            integrated_contr_em_per_order, 
            PT.pressure, 
            bestfit_color=contr_em_color[w_set]
            )
        
        if (integrated_opa_cloud == 0).all():
            continue
        
        # Add the integrated cloud opacity
        ax_opa_cloud = ax_PT.twiny()
        fig_opa_cloud(
            ax_opa_cloud, 
            integrated_opa_cloud, 
            PT.pressure, 
            xlim=(1e2, 1e-8), 
            color=opa_cloud_color[w_set]
            )
    
    # Save or return the axis
    if is_new_fig and (prefix is not None):
        fig.savefig(prefix+'plots/PT_profile.pdf')
        plt.close(fig)
    else:
        return ax_PT

def fig_contr_em(ax_contr, integrated_contr_em, integrated_contr_em_per_order, pressure, bestfit_color='C1'):
    
    if integrated_contr_em_per_order is not None:
        
        if len(integrated_contr_em_per_order) != 1:        
            # Plot the emission contribution functions per order
            #cmap_per_order = mpl.cm.get_cmap('coolwarm')
            cmap_per_order = mpl.colors.LinearSegmentedColormap.from_list(
                'cmap_per_order', colors=['b', 'r']
                )
            
            for i in range(len(integrated_contr_em_per_order)):
                color_i = cmap_per_order(i/(len(integrated_contr_em_per_order)-1))
                ax_contr.plot(
                    integrated_contr_em_per_order[i], pressure, 
                    c=color_i, alpha=0.5, lw=1, 
                    )

    ax_contr.plot(
        integrated_contr_em, pressure, 
        c=bestfit_color, ls='--', alpha=0.7
        )
    ax_contr.set(xlim=(0, 1.1*np.nanmax(integrated_contr_em)))

    ax_contr.tick_params(
        axis='x', which='both', top=False, labeltop=False
        )

    return ax_contr

def fig_opa_cloud(ax_opa_cloud, integrated_opa_cloud, pressure, xlim=(1e0, 1e-10), color='grey'):

    ax_opa_cloud.plot(
        integrated_opa_cloud, pressure, c=color, lw=1, ls=':', alpha=0.5
        )
    
    # Set the color of the upper axis-spine
    ax_opa_cloud.tick_params(
        axis='x', which='both', top=True, labeltop=True, colors='0.5'
        )
    ax_opa_cloud.spines['top'].set_color('0.5')
    ax_opa_cloud.xaxis.label.set_color('0.5')

    ax_opa_cloud.set(
        xlabel=r'$\kappa_\mathrm{cloud}\ (\mathrm{cm^2\ g^{-1}})$', 
        xlim=xlim, xscale='log', 
        )
    
def fig_VMR(Chem,
            ax=None,
            fig=None,
            species_to_plot=[],
            pressure=[],
            ylabel=r'$P\ \mathrm{(bar)}$',
            ls='-', # deprecated
            xlim=(1e-12, 1e-2),
            showlegend=True,
            fig_name=None,
            ):
    
    is_new_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7,7), tight_layout=True)
        is_new_fig = True
    
    MMW = Chem.mass_fractions['MMW']
    for species_i in species_to_plot:
        mass_i  = Chem.read_species_info(species_i, info_key='mass')
        color_i = Chem.read_species_info(species_i, info_key='color')
        label_i = Chem.read_species_info(species_i, info_key='label')
        line_species_i = Chem.read_species_info(species_i, info_key='pRT_name')
        ls_i = ':' if line_species_i.endswith('_high') else '-'
        lw_i = 2.5 if line_species_i.endswith('_high') else 1.5
        if line_species_i not in Chem.mass_fractions.keys():
            print(f'No mass fraction for {species_i}')
            continue
        vmr_i = Chem.mass_fractions[line_species_i] * (MMW / mass_i)
        # vmr_i = Chem.VMRs[line_species_i]
        if vmr_i is None:
            print(f'No VMR for {species_i}')
            continue
        # print(f' - Plotting {species_i} ({line_species_i})')

        label_i = label_i if showlegend else None
        if hasattr(Chem, 'VMRs_envelopes'):
            ax.fill_betweenx(
                y=pressure, x1=Chem.VMRs_envelopes[species_i][0], 
                x2=Chem.VMRs_envelopes[species_i][-1], 
                color=color_i, ec='none', alpha=0.4,
                )
            ax.plot(Chem.VMRs_envelopes[species_i][1], pressure, label=label_i, ls=ls_i, color=color_i, lw=lw_i)
        else:
            ax.plot(vmr_i, pressure, label=label_i, ls=ls_i, color=color_i, lw=lw_i)
        
        
    ax.set(xscale='log', yscale='log', xlabel='VMR',
           ylabel=ylabel, 
           xlim=xlim,
            ylim=(np.max(pressure), np.min(pressure)),
            )
    # print(f' [fig_VMR] showlegend = {showlegend}')
    if showlegend:
        # ncol = 1 + len(species_to_plot)//2
        # ax.legend(ncol=ncol, loc=(0.00, 1.01+0.08*(ncol-3)), frameon=False)
        ax.legend(ncol=2, loc='upper left', frameon=False)
    if (fig_name is not None):
        fig.savefig(fig_name)
        print(f'--> Saved {fig_name}')
        plt.close(fig)
    return ax

def fig_hist_posterior(posterior_i, 
                       param_range_i, 
                       param_quantiles_i, 
                       param_key_i, 
                       posterior_color='C0', 
                       title=None, 
                       bins=20, 
                       prefix=None, 
                       ):

    for _ in range(2):

        title = title + r'$=' + '{:.2f}'.format(param_quantiles_i[1])
        title = title + '^{' + '+{:.2f}'.format(param_quantiles_i[2]-param_quantiles_i[1]) + '}'
        title = title + '_{' + '-{:.2f}'.format(param_quantiles_i[1]-param_quantiles_i[0]) + '}$'
        
        fig, ax_hist = plt.subplots(figsize=(3,3))
        # Plot the posterior of parameter i as a histogram
        ax_hist.hist(
            posterior_i, bins=bins, range=param_range_i, 
            color=posterior_color, histtype='step', 
            )

        # Indicate the 68% confidence interval and median
        ax_hist.axvline(param_quantiles_i[0], c=posterior_color, ls='--')
        ax_hist.axvline(param_quantiles_i[1], c=posterior_color, ls='-')
        ax_hist.axvline(param_quantiles_i[2], c=posterior_color, ls='--')
        
        ax_hist.set(yticks=[], xlim=param_range_i, title=title)

        # Save the histogram
        if prefix is not None:
            # Make the histograms directory
            if not os.path.exists(prefix+'plots/hists'):
                os.makedirs(prefix+'plots/hists')
                
            fig.savefig(prefix+f'plots/hists/{param_key_i.replace("/", "_")}.pdf')
            plt.close(fig)

        if not param_key_i.startswith('log_'):
            break

        else:
            # Plot another histogram with linear values
            param_key_i = param_key_i.replace('log_', '')
            posterior_i = 10**posterior_i

            if param_key_i == 'C_ratio':
                posterior_i = 1/posterior_i
                title = r'$\mathrm{^{12}C/^{13}C}$'                

            param_quantiles_i = af.quantiles(posterior_i, q=[0.16,0.5,0.84])
            param_range_i = (
                4*(param_quantiles_i[0]-param_quantiles_i[1])+param_quantiles_i[1], 
                4*(param_quantiles_i[2]-param_quantiles_i[1])+param_quantiles_i[1]
                )

            # Update the mathtext title
            title = title.replace('\\log\\ ', '')

def fig_residual_ACF(d_spec, 
                     m_spec, 
                     LogLike, 
                     Cov, 
                     rv=np.arange(-500,500+1e-6,1), 
                     bestfit_color='C1', 
                     prefix=None, 
                     w_set=''
                     ):

    # Create a spectrum residual object
    d_spec_res = copy.deepcopy(d_spec)
    d_spec_res.flux = (d_spec.flux - m_spec.flux*LogLike.phi[:,:,None])

    # Cross-correlate the residuals with itself
    rv, ACF, _, _ = af.CCF(
        d_spec=d_spec_res, m_spec=d_spec_res, 
        m_wave_pRT_grid=None, m_flux_pRT_grid=None, 
        rv=rv, apply_high_pass_filter=False, 
        )
    del d_spec_res

    n_orders, n_dets = d_spec.n_orders, d_spec.n_dets

    fig, ax = plt.subplots(
        figsize=(10,3*n_orders), 
        nrows=n_orders, ncols=n_dets,
        sharey=True, 
        gridspec_kw={
            'wspace':0.1, 'hspace':0.25, 
            'left':0.1, 'right':0.95, 
            'top':1-0.03*7/n_orders, 
            'bottom':0.03*7/n_orders, 
            }
    )
    if n_orders == 1:
        ax = np.array([ax])

    for i in range(n_orders):
        for j in range(n_dets):
            
            mask_ij = d_spec.mask_isfinite[i,j]
            wave_ij = d_spec.wave[i,j,mask_ij]
            delta_wave_ij = wave_ij[None,:] - wave_ij[:,None]

            # Plot the auto-correlation of residuals
            ax[i,j].plot(rv, ACF[i,j], lw=0.5, c='k')
            ax[i,j].plot(rv, ACF[i,j]*0, lw=0.1, c='k')
            ax[i,j].set(xlim=(rv.min(), rv.max()))

            # Get the covariance matrix
            cov = Cov[i,j].get_dense_cov()

            # Scale with the optimal uncertainty-scaling
            cov *= LogLike.s[i,j]**2

            # Take the mean along each diagonal
            ks = np.arange(0, len(cov), 1)
            collapsed_cov = np.zeros_like(ks, dtype=float)
            collapsed_delta_wave = np.zeros_like(ks, dtype=float)

            for l, k in enumerate(ks):
                collapsed_cov[l] = np.mean(np.diag(cov, k))
                collapsed_delta_wave[l] = np.mean(np.diag(delta_wave_ij, k))

            collapsed_cov = np.concatenate(
                (collapsed_cov[::-1][:-1], collapsed_cov)
                )
            collapsed_delta_wave = np.concatenate(
                (-collapsed_delta_wave[::-1][:-1], collapsed_delta_wave)
                )

            collapsed_cov *= mask_ij.sum()

            ax_delta_wave_ij = ax[i,j].twiny()
            ax_delta_wave_ij.plot(
                collapsed_delta_wave, 
                collapsed_cov*np.nanmax(ACF[i,j])/np.nanmax(collapsed_cov), 
                c=bestfit_color, lw=1
                )
            ax_delta_wave_ij.set(
                xlim=(wave_ij.mean() * rv.min()/(nc.c*1e-5),
                      wave_ij.mean() * rv.max()/(nc.c*1e-5))
                )

            if (i == 0) and (j == 1):
                ax_delta_wave_ij.set(
                    xlabel=r'$\Delta\lambda\ \mathrm{(nm)}$', 
                    )

    ax[n_orders//2,0].set(ylabel='Auto-correlation')
    ax[-1,1].set(xlabel=r'$v_\mathrm{rad}\ \mathrm{(km\ s^{-1})}$')

    if prefix is not None:
        fig.savefig(prefix+f'plots/auto_correlation_residuals_{w_set}.pdf')

    #plt.show()
    plt.close(fig)

def plot_ax_CCF(ax, 
                d_spec, 
                m_spec, 
                pRT_atm, 
                m_spec_wo_species=None, 
                pRT_atm_wo_species=None, 
                LogLike=None, 
                Cov=None, 
                rv=np.arange(-1000,1000+1e-6,3.), 
                rv_to_exclude=(-100,100), 
                color='k', 
                label=None,
                prefix=None,
                species_h=None,
                ):

    if pRT_atm_wo_species is not None:
        #pRT_atm_wo_species_flux_pRT_grid = pRT_atm_wo_species.flux_pRT_grid_only.copy()
        pRT_atm_wo_species_flux_pRT_grid = pRT_atm_wo_species.flux_pRT_grid.copy()
    else:
        pRT_atm_wo_species_flux_pRT_grid = None

    rv, CCF, d_ACF, m_ACF = af.CCF(
        d_spec=d_spec, 
        m_spec=m_spec, 
        m_wave_pRT_grid=pRT_atm.wave_pRT_grid.copy(), 
        m_flux_pRT_grid=pRT_atm.flux_pRT_grid.copy(), 
        m_spec_wo_species=m_spec_wo_species, 
        m_flux_wo_species_pRT_grid=pRT_atm_wo_species_flux_pRT_grid, 
        LogLike=LogLike, 
        Cov=Cov, 
        rv=rv, 
        )

    # Convert to signal-to-noise functions
    CCF_SNR, m_ACF_SNR, excluded_CCF_SNR = af.CCF_to_SNR(
        rv, CCF.sum(axis=(0,1)), ACF=m_ACF.sum(axis=(0,1)), rv_to_exclude=rv_to_exclude
        )
    # Store the CCF_SNR and m_ACF_SNR at each RV for later use
    if species_h is not None:
        array = np.array([rv, CCF_SNR, m_ACF_SNR]).T
        assert prefix is not None, 'No prefix provided'
        np.save(prefix+f'data/CCF_{species_h}.npy', array)
        print(f'- Saved CCF_{species_h}.npy')

    ax.axvline(0, lw=1, c='k', alpha=0.2)
    ax.axhline(0, lw=0.1, c='k')
    ax.axvspan(
        rv_to_exclude[0], rv_to_exclude[1], 
        color='k', alpha=0.1, ec='none'
        )

    # Plot the cross- and auto-correlation signal-to-noises
    ax.plot(rv, m_ACF_SNR, c=color, lw=1, ls='--', alpha=0.5)
    ax.plot(rv, CCF_SNR, c=color, lw=1, label=label)
    #ax.plot(rv, excluded_CCF_SNR, c=color, lw=0.5, ls='-', alpha=0.5)
    #ax.fill_between(rv, y1=CCF_SNR, y2=excluded_CCF_SNR, color=color, alpha=0.5)

    SNR_rv0 = CCF_SNR[rv==0][0]
    ax.annotate(
        r'S/N$=$'+'{:.1f}'.format(SNR_rv0), xy=(0, SNR_rv0), 
        xytext=(0.55, 0.7), textcoords=ax.transAxes, color=color, 
        arrowprops={'arrowstyle':'-', 'lw':0.5, 
                    'color':color, 'alpha':0.5
                    }
        )

    ax.legend(
        loc='upper right', handlelength=1, framealpha=0.7, 
        handletextpad=0.5, columnspacing=0.8
        )
    ax.set(ylim=(min([1.2*CCF_SNR.min(), 1.2*m_ACF_SNR.min(), -5]), 
                 max([1.2*CCF_SNR.max(), 1.2*m_ACF_SNR.max(), 7])
                 )
           )
    return ax

def fig_species_contribution(d_spec, 
                             m_spec, 
                             m_spec_species, 
                             pRT_atm, 
                             pRT_atm_species, 
                             Chem, 
                             LogLike, 
                             Cov, 
                             species_to_plot, 
                             rv_CCF=np.arange(-1000,1000+1e-6,5), 
                             rv_to_exclude=(-100,100), 
                             bin_size=25, 
                             prefix=None, 
                             w_set='', 
                             ):

    if not os.path.exists(prefix+'plots/species'):
        os.makedirs(prefix+'plots/species')

    fig_CCF, ax_CCF = plt.subplots(
        figsize=(5,2*(len(species_to_plot)+1)), 
        nrows=len(species_to_plot)+1, 
        sharex=True, sharey=False, 
        gridspec_kw={
            'hspace':0.05, 'left':0.13, 'right':0.95, 
            'top':0.97, 'bottom':0.05
            }
        )
    h = 0

    # Plot the cross-correlation of the complete model
    plot_ax_CCF(
        ax_CCF[0], 
        d_spec, 
        m_spec, 
        pRT_atm, 
        LogLike=LogLike, 
        Cov=Cov, 
        rv=rv_CCF, 
        rv_to_exclude=rv_to_exclude, 
        color='k', 
        label='Complete'
        )
        
    print('** figures.fig_species_contribution**')
    print(f'- species to plot: {species_to_plot}')
    for species_h in list(Chem.species_info.keys()):
        
        if species_h not in species_to_plot:
            continue
        print(f'--> Plotting {species_h}')
        # Check if the line species was included in the model
        line_species_h = Chem.read_species_info(species_h, info_key='pRT_name')
        if line_species_h in Chem.line_species:

            h += 1

            # Read the ModelSpectrum class for this species
            m_spec_h  = m_spec_species[species_h]
            pRT_atm_h = pRT_atm_species[species_h]
            
            # Residual between data and model w/o species_i
            d_res = d_spec.flux/LogLike.phi[:,:,None] - m_spec_h.flux

            low_pass_d_res = np.nan * np.ones_like(d_res)
            low_pass_d_res[d_spec.mask_isfinite] = gaussian_filter1d(d_res[d_spec.mask_isfinite], sigma=300)
            d_res -= low_pass_d_res

            # Residual between complete model and model w/o species_i
            m_res = m_spec.flux - m_spec_h.flux
            #m_res = m_spec_h.flux_only

            low_pass_m_res = np.nan * np.ones_like(m_res)
            low_pass_m_res[d_spec.mask_isfinite] = gaussian_filter1d(m_res[d_spec.mask_isfinite], sigma=300)
            m_res -= low_pass_m_res

            # Read the color and label
            color_h = Chem.read_species_info(species_h, info_key='color')
            label_h = Chem.read_species_info(species_h, info_key='label')

            # Plot the cross-correlation of species_h
            plot_ax_CCF(
                ax_CCF[h], 
                d_spec, 
                m_spec, 
                pRT_atm, 
                m_spec_wo_species=m_spec_h, 
                pRT_atm_wo_species=pRT_atm_h, 
                LogLike=LogLike, 
                Cov=Cov, 
                rv=rv_CCF, 
                rv_to_exclude=rv_to_exclude, 
                color=color_h, 
                label=label_h,
                prefix=prefix,
                species_h=species_h,
                )

            # Use a common ylim for all orders
            ylim = (np.nanmean(d_res) - 5*np.nanstd(d_res), 
                    np.nanmean(d_res) + 4*np.nanstd(d_res)
                    )

            fig, ax = fig_order_subplots(
                d_spec.n_orders, 
                ylabel='Residuals\n'+r'$\mathrm{(erg\ s^{-1}\ cm^{-2}\ nm^{-1})}$'
                )

            for i in range(d_spec.n_orders):
                
                ax[i].axhline(0, lw=0.1, c='k')
                
                for j in range(d_spec.n_dets):

                    label = r'$d-m_\mathrm{w/o\ ' + label_h.replace('$', '') + r'}$'
                    if species_h in ['13CO', 'NH3']:
                        alpha = 0.3
                        label_1 = None
                        label_2 = label + f' (binned to {bin_size} pixels)'
                    else:
                        alpha = 1.0
                        label_1 = label

                    ax[i].plot(
                        d_spec.wave[i,j], d_res[i,j], 
                        c='k', lw=0.5, alpha=alpha, label=label_1
                        )
                    
                    if species_h in ['13CO', 'NH3', 'C18O', 'C17O']:
                        mask_ij = d_spec.mask_isfinite[i,j]
                        binned_d_res_ij = np.nan * np.ones_like(d_res[i,j])
                        #binned_d_res_ij[mask_ij] = gaussian_filter1d(d_res[i,j,mask_ij], sigma=bin_size)
                        binned_d_res_ij = generic_filter(d_res[i,j], np.nanmedian, size=bin_size)

                        ax[i].plot(
                            d_spec.wave[i,j], binned_d_res_ij, c='k', lw=0.5, label=label_2
                            )
                    else:
                        pass
                        '''
                        mask_ij = d_spec.mask_isfinite[i,j]
                        low_pass_d_res_ij = np.nan * np.ones_like(d_res[i,j])
                        low_pass_d_res_ij[mask_ij] = gaussian_filter1d(d_res[i,j,mask_ij], sigma=300)
                        ax[i].plot(
                            d_spec.wave[i,j], low_pass_d_res_ij, c='0.5', lw=1
                            )
                        '''

                    ax[i].plot(
                        d_spec.wave[i,j], m_res[i,j], c=color_h, lw=1, 
                        label=r'$m_\mathrm{only\ '+label_h.replace('$', '')+r'}$'
                        )

                    if (i == 0) and (j == 0):
                        ax[i].legend(
                            loc='upper right', ncol=2, fontsize=10, handlelength=1, 
                            framealpha=0.7, handletextpad=0.3, columnspacing=0.8
                            )
                ax[i].set(ylim=ylim)

            if prefix is not None:
                fig.savefig(prefix+f'plots/species/{species_h}_spec.pdf')
            plt.close(fig)

    ax_CCF[-1].set(
        xlabel=r'$v_\mathrm{rad}\ \mathrm{(km\ s^{-1})}$', 
        xlim=(rv_CCF.min(), rv_CCF.max()), 
        #yticks=np.arange(-6,30,3), 
        #ylim=ax_CCF[-1].get_ylim()
        )
    ax_CCF[len(species_to_plot)//2].set(ylabel='S/N')

    if prefix is not None:
        fig_CCF.savefig(prefix+f'plots/species/CCF_{w_set}.pdf')
    plt.close(fig_CCF)
    
    
def fig_prior_check(ret, w_set, fig_name=None):
    
    assert hasattr(ret, 'd_spec'), 'Retrieval object does not have d_spec attribute.'
    d_spec = ret.d_spec[w_set]
    n_orders = d_spec.n_orders
    fig = plt.figure(figsize=(16, 10))
    # create a gridspec object
    gs = fig.add_gridspec(n_orders, 2, width_ratios=[3,1], wspace=0.02, hspace=0.25, bottom=0.08, top=0.94, left=0.06, right=0.94)
    ax_PT = fig.add_subplot(gs[:,1])
    # move yticks and label from ax_PT to right side
    ax_PT.yaxis.tick_right()
    ax_PT.yaxis.set_label_position('right')
    ax_spec = [fig.add_subplot(gs[order,0]) for order in range(n_orders)]
    
    colors = ['C0', 'C1', 'C2']
    theta = [0.0, 0.5, 1.0] # lower edge, center, upper edge
    for i, theta_i in enumerate(theta):        
        
        ret.Param(theta_i * np.ones(len(ret.Param.param_keys)))
        sample = {k:ret.Param.params[k] for k in ret.Param.param_keys}
        print(sample)
        ln_L = ret.PMN_lnL_func()
        print(f'ln(L) = {ln_L:.2e}\n')
        
        for order in range(d_spec.n_orders):
            for det in range(d_spec.n_dets):
                mask_ij = d_spec.mask_isfinite[order, det]
                if mask_ij.sum() == 0:
                    continue
                x = d_spec.wave[order, det]
                
                label = f'ln(L)={ln_L:.2e}' if (order+det) == 0 else None
                assert hasattr(ret.LogLike[w_set], 'phi'), 'LogLike object does not have phi attribute.'
                # m_flux_spline = SplineModel(N_knots=ret.LogLike[w_set].N_knots, spline_degree=3)(ret.m_spec[w_set].flux[order, det])
                # m_flux = ret.LogLike[w_set].phi[order, det] @ m_flux_spline
                m_flux = ret.LogLike[w_set].m[order,det]
                print(f' phi ({i}, {order}, {det}) = {ret.LogLike[w_set].phi[order, det]}')
                    
                    
                ax_spec[order].plot(x, m_flux, color=colors[i], alpha=0.85, label=label)
                # f = ret.LogLike[w_set].phi[order, det]
                s = ret.LogLike[w_set].s[order, det] # not using this...
                # print(f' f={f:.2e}')
                # ax_spec[order].plot(x, ret.m_spec[w_set].flux[order, det] * f, color=colors[i], alpha=0.85, label=label)
                ax_PT.plot(ret.PT.temperature, ret.PT.pressure, color=colors[i], alpha=0.85)
                if i == 0:
                    ax_spec[order].plot(x, d_spec.flux[order, det], color='k', alpha=0.3)
                if hasattr(ret.m_spec[w_set], 'veiling_model'):
                    ax_spec[order].plot(x, ret.m_spec[w_set].veiling_model[order,det] * np.mean(ret.LogLike[w_set].phi[order,det]),
                                        c=colors[i], lw=1, ls='--', alpha=0.7,
                                 label='Veiling model' if (order+det) == 0 else None)
        
        ax_spec[-1].set(xlabel='Wavelength (nm)')
    
    ax_PT.set(xlabel='Temperature (K)', ylabel='Pressure (bar)',
                yscale='log', ylim=(np.max(ret.PT.pressure), np.min(ret.PT.pressure)))
    # add one ylabel for all subplots
    fig.text(0.02, 0.5, 'Flux (erg s$^{-1}$ cm$^{-2}$ nm$^{-1}$)', va='center', rotation='vertical')
    if fig_name is not None:
        fig.savefig(fig_name)
        print(f'Figure saved as {fig_name}')
        plt.close(fig)
    return fig, ax_PT, ax_spec

def fig_free_parameter(ret, free_parameter,
                       fixed_parameters={},
                       N_points=4, 
                       w_set='K2166',
                       cmap='viridis',
                       fig_name=None):
    ''' Generate spectra with fixed parameters and varying one parameter 
    Same layout as fig_prior_check '''
    assert hasattr(ret, 'd_spec'), 'Retrieval object does not have d_spec attribute.'
    d_spec = ret.d_spec[w_set]
    n_orders = d_spec.n_orders
    
    if len(fixed_parameters) == 0:
        # fixed_parameters = {k:0.5 for k in ret.Param.param
        bestfit_params, _ = ret.PMN_analyze()
        fixed_parameters = dict(zip(ret.Param.param_keys, bestfit_params))
    
    for key_i in list(fixed_parameters.keys()):
        if key_i.startswith('log_'):
            linear_key = key_i.replace('log_', '')
            fixed_parameters[linear_key] = 10**fixed_parameters[key_i]
    # print(f'Fixed parameters: {fixed_parameters}')
    fig = plt.figure(figsize=(14, 8))
    # create a gridspec object
    gs = fig.add_gridspec(n_orders, 2, width_ratios=[3,1], wspace=0.02, hspace=0.0, bottom=0.08, top=0.94, left=0.06, right=0.94)
    ax_PT = fig.add_subplot(gs[:,1])
    # move yticks and label from ax_PT to right side
    ax_PT.yaxis.tick_right()
    ax_PT.yaxis.set_label_position('right')
    ax_spec = [fig.add_subplot(gs[order,0]) for order in range(n_orders)]
    # add three rows with the same size spanning the width up to ax_PT
    # ax_spec = [fig.add_subplot(gs[i,0]) for i in range(0, n_orders*2, 2)]
    
    # Use central value from priors as default
    ret.Param(0.5 * np.ones(len(ret.Param.param_keys)))
    sample = {k:ret.Param.params[k] for k in ret.Param.param_keys}
    
    assert free_parameter in sample.keys(), f'Parameter {free_parameter} not found in free parameters.'
    bounds = ret.Param.param_priors[free_parameter]
    free_parameter_range = np.linspace(bounds[0], bounds[1], N_points)
    print(f' Varying {free_parameter} from {bounds[0]} to {bounds[1]}')
    # colors = plt.cm.viridis(np.linspace(0, 1, len(free_parameter_range)))
    colors = getattr(plt.cm, cmap)(np.linspace(0, 1.0, len(free_parameter_range)))
    # # check if free
    temperature_ref = np.ones_like(ret.PT.pressure)
    
    ret.CB.active = True # to compute the emission contribution function
    for i, theta_i in enumerate(free_parameter_range):        
        
        params_copy = fixed_parameters.copy()
        params_copy[free_parameter] = theta_i
        ret.evaluate_model(list(params_copy.values()))
        ln_L = ret.PMN_lnL_func()
        print(f'ln(L) = {ln_L:.2e}\n')
        # chi2 = ret.LogLike[w_set].chi_squared_red
        
        temperature = ret.PT.temperature
        ret.CB.active = False
        if np.any(temperature_ref != temperature) or free_parameter=='log_g':
            print(f' Plotting temperature for {free_parameter}={theta_i}')
            # ax_PT.plot(ret.PT.temperature, ret.PT.pressure, color=colors[i], alpha=0.85)
            # ret.PT.plot
            ax_PT.plot(ret.PT.temperature, ret.PT.pressure, color=colors[i], alpha=0.85)
            if free_parameter.startswith('dlnT_dlnP') and not free_parameter.endswith('RCE'):
                print(f' Adding hline for {free_parameter}')
                knot_id = int(free_parameter[-1])
                ax_PT.axhline(ret.PT.P_knots[::-1][knot_id], color='magenta', alpha=0.50, ls='-')
            
            if hasattr(ret.pRT_atm[w_set], 'int_contr_em'):

                # Add the integrated emission contribution function
                ax_contr = ax_PT.twiny()
                fig_contr_em(
                    ax_contr, 
                    ret.pRT_atm[w_set].int_contr_em, 
                    None, # integrated_contr_em_per_order, 
                    ret.PT.pressure, 
                    bestfit_color=colors[i]
                    )
        
            # if hasattr
            # ax_twin = ax_PT.twinx()
            
            temperature_ref = np.copy(temperature)
            ret.CB.active = True

        for order in range(d_spec.n_orders):
            for det in range(d_spec.n_dets):
                mask_ij = d_spec.mask_isfinite[order, det]
                x = d_spec.wave[order, det]
                
                label = f'ln(L)={ln_L:.3e}' if (order+det) == 0 else None
                # if hasattr(ret.LogLike[w_set], 'phi'):
                #     m_flux_spline = SplineModel(N_knots=ret.LogLike[w_set].N_knots, spline_degree=3)(ret.m_spec[w_set].flux[order, det])
                #     m_flux = ret.LogLike[w_set].phi[order, det] @ m_flux_spline
                    
                # else:
                #     phi = ret.LogLike[w_set].phi[order, det]
                #     m_flux = ret.m_spec[w_set].flux[order, det] * phi
                    
                m_flux = ret.LogLike[w_set].m[order,det]
                ax_spec[order].plot(x, m_flux, color=colors[i], alpha=0.85, label=label)
                
                if i == 0:
                    ax_spec[order].plot(x, d_spec.flux[order, det], color='k', alpha=0.3)
        
        ax_spec[-1].set(xlabel='Wavelength (nm)')
    im = ax_spec[0].scatter([], [], c=[], cmap=cmap,
                       vmin=free_parameter_range.min(), 
                       vmax=free_parameter_range.max())
    cbar = fig.colorbar(im, ax=[ax_spec[0], ax_PT], 
                        label = free_parameter,
                        aspect=40,
                        orientation='horizontal', 
                        location='top',                        
                        pad=0.05)

    ax_PT.set(xlabel='Temperature (K)', ylabel='Pressure (bar)',
                yscale='log', ylim=(np.max(ret.PT.pressure), np.min(ret.PT.pressure)))
    # copy legend from ax_spec[0] and show it on ax_PT
    ax_PT.legend(*ax_spec[0].get_legend_handles_labels(), loc='upper right', framealpha=0.7, fontsize=16)

    # add one ylabel for all subplots
    fig.text(0.02, 0.5, 'Flux (erg s$^{-1}$ cm$^{-2}$ nm$^{-1}$)', va='center', rotation='vertical')
    if fig_name is not None:
        fig.savefig(fig_name)
        print(f'Figure saved as {fig_name}')
        plt.close(fig)
    return fig, ax_PT, ax_spec

def fig_free_parameter_residuals(ret, free_parameter,
                       fixed_parameters={},
                       N_points=4, 
                       w_set='K2166',
                       cmap='viridis',
                       fig_name=None):
    ''' Generate spectra with fixed parameters and varying one parameter
    Same layout as fig_prior_check, but with spectra and residuals on each page '''
    
    assert hasattr(ret, 'd_spec'), 'Retrieval object does not have d_spec attribute.'
    d_spec = ret.d_spec[w_set]
    n_orders = d_spec.n_orders
    
    if len(fixed_parameters) == 0:
        bestfit_params, _ = ret.PMN_analyze()
        fixed_parameters = dict(zip(ret.Param.param_keys, bestfit_params))
    
    for key_i in list(fixed_parameters.keys()):
        if key_i.startswith('log_'):
            linear_key = key_i.replace('log_', '')
            fixed_parameters[linear_key] = 10**fixed_parameters[key_i]
    
    if fig_name is not None:
        pdf_pages = PdfPages(fig_name)
    
    # Set default parameters
    ret.Param(0.5 * np.ones(len(ret.Param.param_keys)))
    sample = {k: ret.Param.params[k] for k in ret.Param.param_keys}
    
    assert free_parameter in sample.keys(), f'Parameter {free_parameter} not found in free parameters.'
    bounds = ret.Param.param_priors[free_parameter]
    free_parameter_range = np.linspace(bounds[0], bounds[1], N_points)
    print(f'Varying {free_parameter} from {bounds[0]} to {bounds[1]}')
    free_parameter_label = ret.Param.param_mathtext[free_parameter]

    colors = getattr(plt.cm, cmap)(np.linspace(0, 1.0, len(free_parameter_range)))
    
    for order in range(n_orders):
        fig, (ax_spec, ax_resid) = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw={'height_ratios': [3, 1]}, 
                                                sharex=True, tight_layout=True)
        
        for i, theta_i in enumerate(free_parameter_range):
            params_copy = fixed_parameters.copy()
            params_copy[free_parameter] = theta_i
            ret.evaluate_model(list(params_copy.values()))
            ln_L = ret.PMN_lnL_func()
            print(f'ln(L) = {ln_L:.2e}')
            
            for det in range(d_spec.n_dets):
                mask_ij = d_spec.mask_isfinite[order, det]
                x = d_spec.wave[order, det]
                m_flux = ret.LogLike[w_set].m[order, det]
                
                ax_spec.plot(x, m_flux, color=colors[i], alpha=0.75, label=f'ln(L)={ln_L:.3e}')
                
                if i == 0:
                    ax_spec.plot(x, d_spec.flux[order, det], color='k', alpha=0.3, label='Data')
                
                residuals = d_spec.flux[order, det] - m_flux
                ax_resid.plot(x, residuals, color=colors[i], alpha=0.75)

        ax_spec.set_ylabel('Flux (erg s$^{-1}$ cm$^{-2}$ nm$^{-1}$)')
        ax_resid.set_ylabel('Residuals')
        ax_resid.set_xlabel('Wavelength (nm)')
        ax_resid.axhline(0, color='k', lw=0.5, alpha=0.5)
        
        # Legend for the first plot
        if order == 0:
            ax_spec.legend(loc='upper right', framealpha=0.7, fontsize=10)
        
        # Add colorbar for parameter variation
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=free_parameter_range.min(), vmax=free_parameter_range.max()))
        # place cbar on top of the spec     
        cbar = plt.colorbar(sm, ax=ax_spec, orientation='horizontal', pad=0.02, label=free_parameter_label,
                            aspect=80, location='top')

        if fig_name is not None:
            pdf_pages.savefig(fig)
            plt.close(fig)
    
    if fig_name is not None:
        pdf_pages.close()
        print(f'Figures saved in {fig_name}')
    return fig

def fig_veiling_factor(
                d_spec,
                LogLike,
                color='orange',
                fig_name=None,
                       ):
    
    phi = LogLike.phi
    N_knots = LogLike.N_knots # first N_knots are the spectrum
    # take mean of the veiling factor per order
    phi_veiling = phi[:,:,N_knots:]
    # replace values of 1.0 with nan (skipped detectors)
    phi_veiling = np.where(phi_veiling == 1.0, np.nan, phi_veiling)
    phi_order_mean = np.nanmean(phi_veiling, axis=(1,2))
    phi_order_std = np.nanstd(phi_veiling, axis=(1,2))
    
    wave_order = np.mean(d_spec.wave, axis=(1,2))
    
    # fit linear model to the veiling factor with uncertainty
    # x = np.arange(len(phi_order_mean))
    x = wave_order
    y = phi_order_mean
    yerr = phi_order_std
    def model(x, a, b):
        return a*x + b
    popt, pcov = curve_fit(model, x, y, sigma=yerr)
    R2 = np.corrcoef(y, model(x, *popt))[0,1]**2 # R-squared value    
    
    
    fig, ax = plt.subplots(figsize=(9,5), tight_layout=True)
    ax.errorbar(wave_order, phi_order_mean,
                yerr=phi_order_std, 
                fmt='o',
                color=color)
    if R2 > 0.5:
        ax.plot(x, model(x, *popt), 'b--', label=f'y={popt[0]:.2f}x + {popt[1]:.2f}\nR$^2$={R2:.2f}')
        ax.legend()

    ax.set(xlabel='Wavelength / nm', ylabel='Veiling factor ' + r'$r_{\rm k}$')
    if fig_name is not None:
        fig.savefig(fig_name)
        print(f'Figure saved as {fig_name}')
        plt.close(fig)
    return None


def fig_chemistry(
                  Chem,
                  fig=None,
                  species_to_plot=None,
                  color='k',
                  smooth=None,
                  fontsize=14,
                  fig_name=None,
                  ):
    
    assert hasattr(Chem, 'VMRs_posterior'), 'No VMRs_posterior found'
    
    species_to_plot = list(Chem.VMRs_posterior.keys()) if species_to_plot is None else species_to_plot
    assert len(species_to_plot) > 1, 'No species to plot'
    if '12_13CO' in species_to_plot:
        # species_to_plot.remove('12CO')
        species_to_plot.remove('13CO')
    if 'C16_18O' in species_to_plot:
        species_to_plot.remove('C18O')
        
    if 'H2_16_18O' in species_to_plot:
        species_to_plot.remove('H2O_181')
        
        
    if 'C/O' not in species_to_plot and 'C/O' in Chem.VMRs_posterior.keys():    
        species_to_plot.append('C/O')
        
    if 'Fe/H' not in species_to_plot and 'Fe/H' in Chem.VMRs_posterior.keys():
        species_to_plot.append('[Fe/H]')
        
    samples = []
    labels = []
    for species in species_to_plot:
        if species not in ['12_13CO', 'C16_18O', 'C/O', 'Fe/H']:
            samples.append(np.log10(Chem.VMRs_posterior[species]))
            labels.append(f'log {species}')
            
        else:
            samples.append(Chem.VMRs_posterior[species])
            labels.append(species)
    
    samples = np.array(samples).T
    # rearrange samples array to have C/O, Fe/H, 12_13CO first
    # samples = np.roll(samples, 3, axis=-1)
    # labels = np.roll(labels, 3)
    # replace with latex labels
    samples_dict = dict(zip(labels, samples.T))
    # change keys for latex labels for 12CO/13CO and H2O/H2O_181
    if '12_13CO' in samples_dict.keys():
        samples_dict['$^{12}$CO/$^{13}$CO\n'] = samples_dict.pop('12_13CO')
        
    if 'C16_18O' in samples_dict.keys():
        samples_dict['$^{16}$CO/$^{18}$CO\n'] = samples_dict.pop('C16_18O')
        
    if 'log H2_16_18O' in samples_dict.keys():
        samples_dict['log H$_2$$^{16}$O/H$_2$$^{18}$O\n'] = samples_dict.pop('log H2_16_18O')
    if 'Fe/H' in samples_dict.keys():
        samples_dict['[Fe/H]'] = samples_dict.pop('Fe/H')
        
    
    samples = np.array(list(samples_dict.values())).T
    labels = list(samples_dict.keys())
    # ensure C/O and Fe/H are the first two labels and samples
    new_samples = np.copy(samples)
    new_labels = np.copy(labels)
    
    first_labels =['C/O', '[Fe/H]', '$^{12}$CO/$^{13}$CO\n', 
                   '$^{16}$CO/$^{18}$CO\n',
                   'log H$_2$$^{16}$O/H$_2$$^{18}$O\n']
    
    for i, label in enumerate(first_labels):
        if label in labels:
            idx = labels.index(label)
            new_samples = np.insert(new_samples, i, samples[:,idx], axis=1)
            new_labels = np.insert(new_labels, i, label)
            
            # delete old
            delete_idx = (idx+1) if (idx+1) > len(labels) else -1
            new_samples = np.delete(new_samples, delete_idx, axis=1)
            new_labels = np.delete(new_labels, delete_idx)
            
    samples, labels = new_samples, new_labels
        
    
    # get quantiles for ranges
    quantiles = np.array(
            [af.quantiles(samples[:,i], q=[0.16,0.5,0.84]) \
             for i in range(samples.shape[1])]
             )
        
    ranges = np.array(
        [(4*(q_i[0]-q_i[1])+q_i[1], 4*(q_i[2]-q_i[1])+q_i[1]) \
            for q_i in quantiles]
        )
    # pop elements that have identical min and max
    idx_to_pop = []
    for i, r in enumerate(ranges):
        if r[0] == r[1]:
            idx_to_pop.append(i)
    for i in idx_to_pop:
        print(f' - Removing {labels[i]} from plot with identical min and max: {ranges[i]}')
        ranges = np.delete(ranges, i, axis=0)
        labels = np.delete(labels, i)
        samples = np.delete(samples, i, axis=1)
    
    
    
    # samples = np.log10(np.array([Chem.VMRs_posterior[species_i] for species_i in species_to_plot]))
    print(f'Shape of samples array: {samples.shape}')
    fig = corner.corner(
        samples, 
        labels=labels, 
        title_kwargs={'fontsize': fontsize},
        labelpad=0.25*samples.shape[0]/17,
        bins=20,
        max_n_ticks=3,
        show_titles=True,
        range=ranges,
        
        quantiles=[0.16,0.84],
        title_quantiles=[0.16,0.5,0.84],
        
        color=color,
        linewidths=0.5,
        hist_kwargs={'color':color,
                        'linewidth':0.5,
                        'density':True,
                        'histtype':'stepfilled',
                        'alpha':0.5,
                        },
        
        fill_contours=True,
        smooth=smooth,
        fig=fig,
        )
    if fig_name is not None:
        fig.savefig(fig_name)
        print(f' - Saved {fig_name}')
    plt.close(fig)
    return fig

    
def fig_spline_model(
        d_spec, 
        m_spec, 
        LogLike, 
        Cov, 
        xlabel='Wavelength (nm)', 
        bestfit_color='C1', 
        ax_spec=None, 
        ax_res=None, 
        prefix=None, 
        w_set=''
        ):

    if (ax_spec is None) and (ax_res is None):
        # Create a new figure
        is_new_fig = True
        n_orders = d_spec.n_orders

        fig, ax = plt.subplots(
            figsize=(10,2.5*n_orders*2), nrows=n_orders*3, 
            gridspec_kw={'hspace':0, 'height_ratios':[1,1/3,1/5]*n_orders, 
                        'left':0.1, 'right':0.95, 
                        'top':(1-0.02*7/(n_orders*3)), 
                        'bottom':0.035*7/(n_orders*3), 
                        }
            )
    else:
        is_new_fig = False

    ylabel_spec = r'$F_\lambda$'+'\n'+r'$(\mathrm{erg\ s^{-1}\ cm^{-2}\ nm^{-1}})$'
    if d_spec.high_pass_filtered:
        ylabel_spec = r'$F_\lambda$ (high-pass filtered)'

    # Use the same ylim, also for multiple axes
    ylim_spec = (np.nanmean(d_spec.flux)-4*np.nanstd(d_spec.flux), 
                 np.nanmean(d_spec.flux)+4*np.nanstd(d_spec.flux)
                )
    ylim_res = (1/3*(ylim_spec[0]-np.nanmean(d_spec.flux)), 
                1/3*(ylim_spec[1]-np.nanmean(d_spec.flux))
                )
    N_knots = LogLike.N_knots # number of spline knots for spectrum
    if N_knots <= 1:
        print(f' N_knots must be > 1 to plot the spline continuum model')
        return None
        

    for i in range(d_spec.n_orders):

        if is_new_fig:
            # Spectrum and residual axes
            ax_spec = ax[i*3]
            ax_res  = ax[i*3+1]

            # Remove the temporary axis
            ax[i*3+2].remove()

            # Use a different xlim for the separate figures
            xlim = (d_spec.wave[i,:].min()-0.5, 
                    d_spec.wave[i,:].max()+0.5)
        else:
            xlim = (d_spec.wave.min()-0.5, 
                    d_spec.wave.max()+0.5)

        ax_spec.set(xlim=xlim, xticks=[],
                    # ylim=ylim_spec,
                    )
        ax_res.set(xlim=xlim, ylim=ylim_res)

        for j in range(d_spec.n_dets):
        
            mask_ij = d_spec.mask_isfinite[i,j]
            if np.sum(mask_ij) == 0:
                continue
            # if mask_ij.any():
            # Show the observed and model spectra
            ax_spec.plot(
                d_spec.wave[i,j], d_spec.flux[i,j], 
                c='k', lw=0.5, label='Observation'
                )

            label = 'Best-fit model ' + \
                    r'$(\chi^2_\mathrm{red}$ (w/o $\sigma$-model)$=' + \
                    '{:.2f}'.format(LogLike.chi_squared_red) + \
                    r')$'
                    
            # PLot model (check if spline decomposition used during retrieval)
            # if hasattr(LogLike, 'phi'):
            #         m_flux_spline = SplineModel(N_knots=LogLike.N_knots, spline_degree=3)(m_spec.flux[i,j])
            #         m_flux = LogLike.phi[i,j] @ m_flux_spline
                    
            # else:
                
            #     f = LogLike.phi[i,j]
            #     m_flux = m_spec.flux[i,j] * f
            
            m_flux = LogLike.m[i,j] 
            # phi_flat = np.ones_like(LogLike.phi[i,j])
            # m_flux_spline = SplineModel(N_knots=N_knots, spline_degree=3)(m_spec.flux[i,j])
            # m_flux_flat = phi_flat @ m_flux_spline
            m_flux_flat = m_spec.flux[0,i,j,:]
            
            spline_cont = m_flux_flat / m_flux
                    
            ax_spec.plot(
                d_spec.wave[i,j], m_flux, 
                c=bestfit_color, lw=1, label=label
                )
            ax_spec.plot(
                d_spec.wave[i,j], m_flux_flat, 
                c='magenta', lw=1, label='Best-fit model (flat)'
                )
            ax_spec.plot(
                d_spec.wave[i,j], spline_cont, 
                c='deepskyblue', lw=3, label='Spline continuum', alpha=0.5,
                )
            
            # Plot the residuals
            # res_ij = d_spec.flux[i,j] - LogLike.phi[i,j]*m_spec.flux[i,j]
            res_ij = d_spec.flux[i,j] - m_flux
            res_ij_flat = d_spec.flux[i,j] - m_flux_flat
            ax_res.plot(d_spec.wave[i,j], res_ij, c='k', lw=0.5)
            ax_res.plot(d_spec.wave[i,j], res_ij_flat, c='magenta', lw=0.5)
            ax_res.plot(
                [d_spec.wave[i,j].min(), d_spec.wave[i,j].max()], 
                [0,0], c=bestfit_color, lw=1
            )

            # Show the mean error
            mean_err_ij = np.mean(Cov[i,j].err)
            ax_res.errorbar(
                d_spec.wave[i,j].min()-0.2, 0, yerr=1*mean_err_ij, 
                fmt='none', lw=1, ecolor='k', capsize=2, color='k', 
                label=r'$\langle\sigma_{ij}\rangle$'
                )

            # Get the covariance matrix
            cov = Cov[i,j].get_dense_cov()
            
            # Scale with the optimal uncertainty-scaling
            cov *= LogLike.s[i,j]**2

            # Get the mean error from the trace
            mean_scaled_err_ij = np.mean(np.diag(np.sqrt(cov)))

            ax_res.errorbar(
                d_spec.wave[i,j].min()-0.4, 0, yerr=1*mean_scaled_err_ij, 
                fmt='none', lw=1, ecolor=bestfit_color, capsize=2, color=bestfit_color, 
                #label=r'$\s_{ij}\langle\sigma_{ij}\rangle$'
                label=r'$\s_{ij}\cdot\langle\mathrm{diag}(\sqrt{\Sigma_{ij}})\rangle$'
                )

            if i==0 and j==0:
                ax_spec.legend(
                    loc='upper right', ncol=2, fontsize=8, handlelength=1, 
                    framealpha=0.7, handletextpad=0.3, columnspacing=0.8
                    )

    # Set the labels for the final axis
    ax_spec.set(ylabel=ylabel_spec)
    ax_res.set(xlabel=xlabel, ylabel='Res.')

    if is_new_fig and (prefix is not None):
        plt.savefig(prefix+f'plots/bestfit_spline_model_{w_set}.pdf')
        plt.close(fig)
    else:
        return ax_spec, ax_res

    

# if __name__ == '__main__':