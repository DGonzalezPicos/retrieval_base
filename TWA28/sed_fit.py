""" Rebin JWST spectrum and BT-Settl model to the same wavelength grid and
fit SED properties with a disk model"""
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os
import xarray as xr
import time
import seaborn as sns
import corner
import json
from memory_profiler import profile
from pympler import asizeof

from scipy.signal import medfilt
# from spectres import spectres

os.environ['OMP_NUM_THREADS'] = '1'
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
import pymultinest


from retrieval_base.spectrum_jwst import SpectrumJWST
from retrieval_base.btsettl import BTSettl
from retrieval_base import auxiliary_functions as af
from retrieval_base.resample import Resample

class SED:
    
    path_suffix = 'dario/phd' if 'dario' in os.environ['HOME'] else 'dgonzalezpi'
    base_path = pathlib.Path(f'/home/{path_suffix}')    
    path = base_path /'retrieval_base'
    target = 'TWA28'
    
    bestfit_color = 'brown'
    
    flux_factor = 1e15 # multiply by constant to avoid numerical issues
    
    def __init__(self, gratings=[], run='test_1', evaluation=False):
        
        self.gratings = gratings
        assert all([g in ['g140h-f100lp', 'g235h-f170lp', 'g395h-f290lp'] for g in gratings]), 'Invalid grating'
        
        self.files = [f'jwst/{self.target}_{g}.fits' for g in gratings]
        
        self.run = run
        self.update_path(self.path)
        self.evaluation = evaluation
        self.prefix = str(self.run_path  / 'pmn_')
        self.cb_count = -1
        
    def load_spec(self, Nedge=50, Nbin=None):
        
        self.spec = SpectrumJWST(Nedge=Nedge).load_gratings(self.files)
        self.spec.reshape(self.spec.n_orders, 1)
        # spec.fix_wave_nans() # experimental...
        self.spec.sigma_clip_reshaped(use_flux=False, 
                                    sigma=3, 
                                    width=31, 
                                    max_iter=5,
                                    fun='median', 
                                    debug=False)
        self.spec.squeeze()
        self.spec.flux_scaling(self.flux_factor)
        self.wave_full = self.spec.wave.copy()

        nans = np.isnan(self.wave_full) | np.isnan(self.spec.flux)
        self.wave_full[nans] = np.nan
        # self.Nbin = Nbin
        # if self.Nbin is not None:
        #     self.spec.rebin(nbin=self.Nbin)
            
        return self
    
    def load_spitzer(self, 
                     file='1102-3430.txt', 
                     wmax=33.0, 
                     nbin=0,
                     sigma_clip=3.0,
                     sigma_width=5):
        spitzer_file = self.path / self.target / 'spitzer' / file
        wave, flux, err, flag = np.loadtxt(spitzer_file, skiprows=0).T
        mask = flag > 0
        print(f' Number of flagged points: {np.sum(mask)}')

        # mask points beyond 33 micron
        mask_wave = wave > wmax
        # flux[mask_wave] = np.nan
        wave = wave[~mask_wave]
        flux = flux[~mask_wave]
        err = err[~mask_wave]
        
        
        ## Unit conversion ##
        wave_cm = wave * 1e-4  # [microns] -> [cm]
        # convert Jy to [erg cm^{-2} s^{-1} Hz^{-1}]
        flux *= 1e-23
        # convert [erg cm^{-2} s^{-1} Hz^{-1}] -> [erg cm^{-2} s^{-1} cm^{-1}]
        flux *= 2.998e10 / wave_cm**2 # wave in cm
        # Convert [erg cm^{-2} s^{-1} cm^{-1}] -> [erg cm^{-2} s^{-1} nm^{-1}]
        flux *= 1e-7
        err = err * (1e-23) * (2.998e10 / wave_cm**2) * 1e-7
        
        flux *= self.flux_factor # NEW (to avoid numerical issues)
        err *= self.flux_factor # NEW (to avoid numerical issues)


        # clip 3 sigma outliers
        if sigma_clip > 0:
            flux_medfilt = medfilt(flux, kernel_size=sigma_width)
            mask_clip = np.abs(flux - flux_medfilt) > sigma_clip*err
            # flux[mask_clip] = np.nan
            wave = wave[~mask_clip]
            flux = flux[~mask_clip]
            err = err[~mask_clip]
    
        self.wave_full = af.make_array([*self.wave_full, wave*1e3])
        
        
        # self.nbin_spitzer = nbin
        # if self.nbin_spitzer > 0:
        #     # wave, flux, err = self.rebin_spitzer(wave, flux, err, nbin=nbin)
        #     wave, flux, err = af.rebin(wave, flux, self.nbin_spitzer, err=err)
        if getattr(self, 'wave_step', None) is not None:
            print(f' Resampling Spitzer spectrum with wave_step = {self.wave_step}')
            new_wave = np.arange(np.nanmin(wave)*1e3, np.nanmax(wave)*1e3, self.wave_step)
            # flux, err = spectres(new_wave, wave*1e3, flux, spec_errs=err, fill=np.nan)
            flux, cov = Resample(wave=wave*1e3, flux=flux, flux_err=err)(new_wave)
            err = np.sqrt(np.diag(cov))
            # wave = new_wave
            assert np.all(np.isfinite(flux)), 'NaNs in resampled Spitzer flux'
            assert np.all(np.isfinite(err)), 'NaNs in resampled Spitzer error'
            
        # concatenate data along axis 0  for wave, flux, err
        # assert len(self.spec.wave.shape) == 2, f'Invalid shape {self.spec.wave.shape}, should be (n_orders, n_wave)'
        # self.spec.wave = af.make_array([*self.spec.wave, wave*1e3])
        # self.spec.flux = af.make_array([*self.spec.flux, flux])
        # self.spec.err = af.make_array([*self.spec.err, err])
        print(f' Adding {np.sum(~np.isnan(flux))} points from Spitzer')
        self.spec.wave.append(new_wave)
        self.spec.flux.append(flux)
        self.spec.err.append(err)
        
        # self.spec.n_orders += 1
        return self
    
    def update_path(self, path):
        
        self.path = pathlib.Path(path)
        if 'retrieval_base' not in self.path.parts:
            self.path = self.path / 'retrieval_base'
            
        self.run_path = self.path / self.target / 'SED_runs' / self.run
        self.run_path.mkdir(parents=True, exist_ok=True)
        self.figs_path = self.run_path / 'figs'
        self.figs_path.mkdir(parents=True, exist_ok=True)
        self.prefix = str(self.run_path  / 'pmn_')
        return self
        
    
    def mask_wave(self, wmin=None, wmax=None, debug=True):
        
        nans_in = np.isnan(self.spec.wave)
        if wmin is not None:
            self.spec.wave[self.spec.wave < wmin] = np.nan
            self.wave_full[self.wave_full < wmin] = np.nan
            
        if wmax is not None:
            self.spec.wave[self.spec.wave > wmax] = np.nan
            self.wave_full[self.wave_full > wmax] = np.nan
            
        nans_out = np.isnan(self.spec.wave)
        if debug:
            print(f' nans in = {nans_in.sum()}')
            print(f' nans out = {nans_out.sum()}')
            print(f' total nans wave = {np.isnan(self.spec.wave).sum() / self.spec.wave.size:.2f}')
        return self
    
    # @profile
    def init_BTSettl(self,
                     create_grid=False,
                     wmin=800, 
                     wmax=12500,
                     wave_unit='nm',
                     file=None):
    
       
        self.bt = BTSettl()
        # if create_grid:
        if file is None or not pathlib.Path(file).exists():
            self.bt.load_full_dataset()
            
            wave = np.arange(wmin, wmax, 0.5)
            self.bt.prepare_grid(teff_range=[2100,2900], 
                                    logg_range=[3.0, 4.5], 
                                    out_res=3000,
                                    wave=wave,
                                    file_out=file,
            )
        
        # self.bt.load_dataset(self.bt.path / 'BTSETTL_NIRSPec.nc')
        assert pathlib.Path(file).exists(), f'Invalid file {file}'
        self.bt.load_dataset(file)
        self.bt.set_interpolator()
        return self

    
    def rebin(self, Nbin):
            
        assert Nbin > 0, 'Invalid Nbin'
        self.Nbin = Nbin
        self.spec.rebin(nbin=self.Nbin)
        return self
    
    def resample(self, wave_step=0.1):
        
        # assert wave_step > np.nanmean(np.diff(self.wave_full)), f'Invalid dwave {wave_step}'
        self.wave_step = wave_step
        self.spec.resample(wave_step)
        
        nan_frac = [np.isnan(f).sum() / f.size for f in self.spec.flux]
        assert all([f < 1.0 for f in nan_frac]), f'Invalid nan fraction {nan_frac}'
        return self
        
    def plot(self, ax=None, **kwargs):
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    
        # if len(self.spec.flux.shape) > 1:
        if isinstance(self.spec.flux, np.ndarray):
            for i in range(self.spec.n_orders):
                
                if self.Nbin is not None:
                    ax.scatter(self.spec.wave[i,], self.spec.flux[i,], **kwargs)
                    if hasattr(self, 'err'):
                        ax.errorbar(self.spec.wave[i,], self.spec.flux[i,], yerr=self.spec.err[i,], fmt='o', **kwargs)
                else:
                    ax.plot(self.spec.wave[i,], self.spec.flux[i,], **kwargs)
        
        return ax
    
    # define prior function for pymultinest
    def set_params(self, free_params, constant_params={}):
        """free_params is a dictionary containing the bounds of the prior of each free parameter"""
        self.free_params = free_params
        self.free_params_keys = list(free_params.keys())
        self.ndim = len(self.free_params_keys)
        
        self.constant_params = constant_params
        
        # merge both dictionaries
        self.params = {**self.free_params, **self.constant_params}
        # save dictionary as json file
        with open(self.run_path / 'params.json', 'w') as f:
            
            json.dump(self.params, f, indent=4)
        print(f' Saved {self.run_path / "params.json"}')
        
        
        # essential_keys = ['teff', 'logg', 'R_p', 'd_pc']
        # assert all([key in self.params.keys() for key in essential_keys]), f'Missing essential keys {essential_keys}'
        
        self.add_disk = ('T_d' in self.params.keys()) or ('R_d' in self.params.keys())
        
        self.n_dof = sum([np.sum(~np.isnan(f)) for f in self.spec.flux]) - self.ndim
        print(f' Number of degrees of freedom = {self.n_dof}')
        return self
    
    def prior(self, cube=None, ndim=None, nparams=None):
        """prior function for pymultinest"""
        
        for i, key in enumerate(self.free_params_keys):
            a, b = self.free_params[key]
            cube[i] = a + (b - a) * cube[i]
            self.params[key] = cube[i]
            # log to linear for keys that star with 'log_'
            if key.startswith('log_'):
                self.params[key[4:]] = 10**cube[i]
            
        
        return cube

    def prior_check(self, n=4, random=False, inset_xlim=None, xscale='linear', yscale='linear'):
        """Check prior function"""
        
        fig, ax = plt.subplots(2,1, figsize=(14, 4), sharex=True, gridspec_kw={'height_ratios': [2, 1]},
                               tight_layout=True)
        self.plot(ax=ax[0], color='k', alpha=0.2)
        ax[0].set_xscale(xscale)
        ax[0].set_yscale(yscale)
        # ax[1].set_yscale(yscale)
        
        # color palette from seaborn
        colors = sns.color_palette('deep')
        for i in range(n):
            self.prior(cube=np.random.rand(self.ndim), ndim=self.ndim, nparams=None)
            start = time.time()
            lnL = self.loglike()
            end = time.time()

            print(f' Paramaters: {self.params}')
            print(f' lnL = {lnL:.2e} (t={end-start:.2f} s)')
            print(f' Uncertainty scaling sqrt(s2) = {np.sqrt(self.s2)}')
            
            # self.bt.plot(ax=ax[0], color=colors[i], alpha=0.8, lw=1., s=10)

            res = [f - m for f, m in zip(self.spec.flux, self.bt.flux)]
            
            chi2_v = self.chi2 / self.n_dof
            ax_spec = [ax[0]]
            if inset_xlim is not None:
                ax_inset = fig.add_axes([0.6, 0.6, 0.25, 0.25])
                ax_inset.set_xlim(inset_xlim)
                ax_inset.set_xlim(inset_xlim)
                # merge list of lists into 1d numpy array
                w_flat, f_flat = ([] for _ in range(2))
                for w, f in zip(self.spec.wave, self.spec.flux):
                    w_flat = np.concatenate([w_flat, w])
                    f_flat = np.concatenate([f_flat, f])
                # print(w_flat)
                mask_inset = (w_flat > inset_xlim[0]) & (w_flat < inset_xlim[1])
                assert np.sum(mask_inset) > 0, 'No points in inset'
                ylim = (0.8*np.nanmin(f_flat[mask_inset]), 1.2*np.nanmax(f_flat[mask_inset]))
                ax_inset.set_ylim(ylim)
                ax_spec.append(ax_inset)
                # indicate inset
                ax[0].indicate_inset_zoom(ax_inset)
                
                
            for j in range(len(res)):
                for axi in ax_spec:
                    axi.plot(self.spec.wave[j], self.spec.flux[j], color=colors[i], alpha=0.2)
                    axi.plot(self.spec.wave[j], self.m[j], color=colors[i], alpha=0.8)
                    axi.plot(self.spec.wave[j], self.m[j] - self.bt.flux_disk[j], color=colors[i],ls='--', alpha=0.3)
                
                ax[1].scatter(self.spec.wave[j], res[j], color=colors[i], s=10)
                # add text top right corner with same color showing chi2_nu
            ax[0].text(0.98, 0.98-0.1*i, f'$\chi^2_\\nu$={chi2_v:.2f}',
                        transform=ax[0].transAxes, ha='right', va='top', color=colors[i])
                
            
        ax[1].axhline(0, color='k', ls='-', lw=0.4)

        # make ylims of ax[1] symmetric
        # if yscale == 'linear':
        ylim = np.nanmax(np.abs(ax[1].get_ylim()))
        ax[1].set_ylim(-ylim, ylim)
        xlim = (np.nanmin(self.spec.wave[0])*0.95, 1.01*np.nanmax(self.spec.wave[-1]))
        ax[0].set_xlim(xlim)
        flux_factor_label = r' $\times 10^{-15}$' if self.flux_factor == 1e15 else ''
        ax[0].set_ylabel('Flux' + flux_factor_label+ r' / erg s$^{-1}$ cm$^{-2}$ nm$^{-1}$')
        ax[0].set(title='Prior check')
        ax[-1].set(xlabel='Wavelength / nm', ylabel='Residuals')
        # plt.show()
        fig.savefig(self.figs_path / 'prior_check.pdf')
        print(f'Saved {self.figs_path / "prior_check.pdf"}')
        plt.close(fig)
        return None
        
                    
        
    
    def loglike(self, cube=None, ndim=None, nparams=None):
        """log likelihood function for pymultinest"""

        # self.bt = self.bt.copy()
        self.bt.get_spec(teff=self.params['teff'], logg=self.params['logg'])
        self.bt.scale_flux(R_jup=self.params['R_p'], d_pc=self.params['d_pc'])
        self.bt.apply_flux_factor(self.flux_factor)

        assert np.sum(np.isnan(self.bt.flux)) == 0, 'NaNs in BTSettl flux'
       
        self.bt.resample(new_wave=self.spec.wave)
        
        if self.add_disk:
            self.bt.blackbody_disk(T=self.params['T_d'], 
                                   R=self.params.get('R_d', 10**self.params['log_R_d']),
                                   d=self.params['d_pc'], 
                                    add_flux=True,
                                    flux_factor=self.flux_factor)
            
        self.bt.apply_flux_scaling(a_j=self.params.get('a_j', 1.0),
                                   a_h=self.params.get('a_h', 1.0),
                                   a_hk=self.params.get('a_hk', 1.0),
        )
            
        n_chunks = len(self.spec.wave)
        # print(f' n_chunks = {n_chunks}')
        
        s2, self.m = ([] for i in range(2))
        chi2 = 0.
        lnL = 0.
    
        for i in range(n_chunks):
            
            w = self.spec.wave[i]
            f = self.spec.flux[i]
            e = self.spec.err[i]
            # forward model
            m_i = self.bt.flux[i]
            self.m.append(m_i)
            nans = np.isnan(w) | np.isnan(f) | np.isnan(e) | np.isnan(m_i)
            n = np.sum(~nans)
            if n == 0:
                print(f' No valid points in chunk {i}')
                continue
            
            w = w[~nans]
            f = f[~nans]
            e = e[~nans]
            m_i = m_i[~nans]
            
            # add uncertainty scaling
            e2 = e**2
            logdet_cov = np.sum(np.log(e2))

            chi2_i_0 = np.sum((f - m_i)**2 / e2)
            s2_i = min(20**2, chi2_i_0 / n) # TODO: test this clipping
            
            chi2 += chi2_i_0
            chi2_i = chi2_i_0 / s2_i
            
            
            lnL_i = -0.5 * n * np.log(2*np.pi)
            lnL_i += -0.5 * logdet_cov
            lnL_i += -0.5 * n * np.log(s2_i)
            lnL_i += -0.5 * chi2_i
            lnL += lnL_i
            
            s2.append(s2_i)
            
        self.chi2 = chi2
        self.s2 = np.array(s2)
        self.lnL = lnL
        
        if np.isfinite(self.lnL):
            return self.lnL
        else:
            return -np.inf
    
    
    def PMN_analyzer(self, return_posterior=False):
        '''Load the MultiNest output and save attributes:
        - bestfit_params
        - posterior (if requested)
        '''
        
         # Set-up analyzer object
    
        
        outputfiles_basename = self.prefix
        # if not os.path.exists(file_pmn):
        #     # file_pmn = 'results/'+self.conf.prefix+'pmn_.txt'
        #     outputfiles_basename = self.conf.prefix.split('/')

        analyzer = pymultinest.Analyzer(
            n_params=self.ndim, 
            outputfiles_basename=outputfiles_basename
            )
        stats = analyzer.get_stats()
       
        self.bestfit_params = np.array(stats['modes'][0]['maximum a posterior'])

        if return_posterior:
            # Load the equally-weighted posterior distribution
            return self.bestfit_params, analyzer.get_equal_weighted_posterior()[:,:-1]
        return self.bestfit_params
    
    def PMN_callback(self, 
                    n_samples, 
                    n_live, 
                    n_params, 
                    live_points, 
                    posterior, 
                    stats,
                    max_ln_L, 
                    ln_Z, 
                    ln_Z_err, 
                    nullcontext
                    ):
        
        self.cb_count += 1
        if self.evaluation:
            self.bestfit_params_values, posterior = self.PMN_analyzer(return_posterior=True)
            # self.get_PT_mf_envelopes(posterior)
        else:
            self.bestfit_params_values = posterior[np.argmax(posterior[:,-2]),:-2]
            posterior = posterior[:,:-2]
            
        self.bestfit_params = {key: self.bestfit_params_values[i] for i, key in enumerate(self.free_params_keys)}
        for k,v in self.bestfit_params.items():
            print(f' {k} = {v:.2f}')
            self.params[k] = v
            
        self.loglike()
        # if self.evaluation:
        # print(f' Best-fit parameters: {self.bestfit_params}')
        print(f' chi2 = {self.chi2:.2f}')
        print(f' error scaling sqrt(s2) = {np.sqrt(self.s2)}')
        
        # plot best-fit spectrum
        
        # cornerplot
        Q = np.array(
                [np.quantile(posterior[:,i], q=[0.16,0.5,0.84]) \
                for i in range(posterior.shape[1])]
                )
            
        ranges = np.array(
            [(4*(q_i[0]-q_i[1])+q_i[1], 4*(q_i[2]-q_i[1])+q_i[1]) \
                for q_i in Q]
            )
        # labels = [r'$T_{\rm eff}$', r'$\log g$', r'$R_p$', r'$T_d$', r'$R_d$']
        labels = self.free_params_keys
        fontsize = 16
        smooth = 1.0 if self.evaluation else 0.0
        fig = plt.figure(figsize=(14, 16))
        fig = corner.corner(posterior, 
                            labels=labels, 
                            title_kwargs={'fontsize': fontsize},
                            labelpad=0.25*posterior.shape[0]/17,
                            bins=20,
                            max_n_ticks=3,
                            show_titles=True,
                            range=ranges,
                            
                            quantiles=[0.16,0.84],
                            title_quantiles=[0.16,0.5,0.84],
                            
                            color=self.bestfit_color,
                            linewidths=0.5,
                            hist_kwargs={'color':self.bestfit_color,
                                            'linewidth':0.5,
                                            'density':True,
                                            'histtype':'stepfilled',
                                            'alpha':0.5,
                                            },
                            
                            fill_contours=True,
                            smooth=smooth,
                            # truths=truths,
                            # truth_color=self.true_color,
                            fig=fig,
                            )
        
        # add bestfit spectrum to figure
        l, b, w, h = [0.49,3.2,0.50,0.20]

        ax_res_dim  = [l, b*(h+0.03), w, 0.97*h/5]
        ax_spec_dim = [l, ax_res_dim[1]+ax_res_dim[3], w, 2*h/4]
        ax_spec_log_dim = [l, ax_spec_dim[1]+ax_spec_dim[3]+0.015, w, 2*h/4]
        
        ax_spec = fig.add_axes(ax_spec_dim)
        ax_spec_log = fig.add_axes(ax_spec_log_dim)
        ax_res = fig.add_axes(ax_res_dim)
        # self.plot(ax=ax_spec, color='k', alpha=0.2)
        ax_spec_list = [ax_spec, ax_spec_log]
        for i in range(self.spec.n_orders):
            
            for j, ax_spec_j in enumerate(ax_spec_list):
                ax_spec_j.plot(self.spec.wave[i], self.spec.flux[i], color='k', alpha=0.2)
                ax_spec_j.plot(self.spec.wave[i], self.m[i], color='brown', alpha=0.8, 
                            label=f'BT-Settl + Disk ($\chi^2$={self.chi2:.2f})' if i==0 else None)
                
                ax_spec_j.plot(self.spec.wave[i], self.m[i] - self.bt.flux_disk[i], 
                            color='dodgerblue', label='BT' if i==0 else None,
                            alpha=0.8, ls='--')
                ax_spec_j.plot(self.spec.wave[i], self.bt.flux_disk[i], color='darkorange', alpha=0.8, ls=':',
                            label='Disk' if i==0 else None)
            ax_res.scatter(self.spec.wave[i], self.spec.flux[i] - self.m[i], color='k', s=10)
       
        xlim = (0.98*np.nanmin(self.spec.wave[0]), 1.01*np.nanmax(self.spec.wave[-1]))
        # if xlim[1] > 10e3:
            # log-log plot
        ax_spec_log.set_xscale('log')
        ax_spec_log.set_yscale('log')
            # ax_res.set_xscale('log')
            # ax_res.set_yscale('log')
        # else:
        ylim = np.nanmax(np.abs(ax_res.get_ylim()))
        ax_res.set_ylim(-ylim, ylim)
        ax_res.axhline(0, color='k', ls='-', lw=0.4)

            
        ax_res.set_xlabel(r'Wavelength / nm')
        ax_res.set_ylabel(r'Residuals')
        
        ax_spec.legend(frameon=False)
        flux_factor_label = r' $\times 10^{-15}$' if self.flux_factor == 1e15 else ''
        # ax_spec.set_ylabel('Flux' + flux_factor_label+ r'/ erg s$^{-1}$ cm$^{-2}$ nm$^{-1}$')
        # common ylabel for top two axes
        fig.text(ax_spec_dim[0]-0.05, ax_spec_dim[1]+0.11, 'Flux' + flux_factor_label+ r'/ erg s$^{-1}$ cm$^{-2}$ nm$^{-1}$',
                va='center', rotation='vertical')
    
        # corner.overplot_lines(fig, bestfit_params, color=self.bestfit_color, lw=0.5)
        fig_label = 'final' if self.evaluation else f'{self.cb_count}'
        outfig = self.figs_path / f'summary_{fig_label}.pdf'
        fig.savefig(outfig)
        print(f'Saved {outfig}\n')
        plt.close(fig)
        
    def set_PMN_hyperparameters(self,
                                n_live_points=100,
                                evidence_tolerance=50.0,
                                n_iter_before_update=100,
                                sampling_efficiency=0.20,
                                const_efficiency_mode=True
                                ):
        '''Set-up MultiNest hyperparameters'''
        
        self.n_live_points = n_live_points
        self.evidence_tolerance = evidence_tolerance
        self.n_iter_before_update = n_iter_before_update
        self.sampling_efficiency = sampling_efficiency
        self.const_efficiency_mode = const_efficiency_mode
        return self
    
    def PMN_run(self):
        # Pause the process to not overload memory on start-up
        time.sleep(1)
        if not hasattr(self, 'n_live_points'):
            self.set_PMN_hyperparameters()
        
        pymultinest.run(
            LogLikelihood=self.loglike,
            Prior=self.prior,
            n_dims=self.ndim,
            outputfiles_basename=self.prefix,
            resume=False,
            verbose=True,
            const_efficiency_mode=self.const_efficiency_mode,
            sampling_efficiency=self.sampling_efficiency,
            n_live_points=self.n_live_points,
            evidence_tolerance=self.evidence_tolerance,
            n_iter_before_update=self.n_iter_before_update,
            dump_callback=self.PMN_callback,
            )
        
    def PMN_eval(self):
        self.evaluation = True
        self.PMN_callback(n_samples=None, 
                    n_live=None, 
                    n_params=None, 
                    live_points=None, 
                    posterior=None, 
                    stats=None,
                    max_ln_L=None, 
                    ln_Z=None, 
                    ln_Z_err=None, 
                    nullcontext=None
        )
        return self
    
    def save(self, file):
        # pickle save
        af.pickle_save(file, self)
        return self
    
    def list_memory_allocation(self):
        print(f"Memory usage of object {self}: {asizeof.asizeof(self) / (1024**2):.2f} MB")
        memory_dict = {k:asizeof.asizeof(v) for k,v in self.__dict__.items() if not k.startswith('__')}
        # sort by memory usage
        memory_dict = dict(sorted(memory_dict.items(), key=lambda item: item[1], reverse=True))
        for k, v in memory_dict.items():
            v_MB = v / (1024 ** 2)
            if v_MB < 0.01:
                continue
            print(f'{k}: {v_MB:.2f} MB')
            try:
                memory_dict_attr = {key:asizeof.asizeof(val) for key,val in getattr(self, k).__dict__.items() if not key.startswith('__')}
                memory_dict_attr = dict(sorted(memory_dict_attr.items(), key=lambda item: item[1], reverse=True))
                for key, val in memory_dict_attr.items():
                    val_MB = val / (1024 ** 2)
                    if val_MB < 0.01:
                        continue
                    print(f'  {key}: {val_MB:.2f} MB')
            except:
                continue
            
        return self

      
    
if __name__ == '__main__':
    
    
    path = pathlib.Path('/home/dario/phd/retrieval_base')
    target = 'TWA28'
    cwd = os.getcwd()
    if target not in cwd:
        print(f'Changing directory to {target}')
        os.chdir(target)

    run = 'with_spitzer_8'


    gratings = [
                # 'g140h-f100lp', 
                'g235h-f170lp', 
                'g395h-f290lp',
                ]

    Nedge = 40
    start = time.time()
    sed = SED(gratings, run=run).load_spec(Nedge=Nedge)
    
    # wmin = 0.95 * np.nanmin(sed.spec.wave) # [nm]
    wmin = 1980.0 # [nm]
    wmax = 16.0 * 1e3 # [um] -> [nm]
    bt_model_file = sed.run_path / f'BTSETTL_{wmin:.0f}_{wmax:.0f}.nc'

    sed.init_BTSettl(wmin=wmin,
                    #  wmax=1.05 * np.nanmax(sed.spec.wave)*1e-3,
                    # wmax=33.0, # Spitzer
                    wmax=wmax,
                    # create_grid=False, # deprecated, use `file`
                    file=bt_model_file,
                     wave_unit='nm')
    sed.mask_wave(wmin=wmin)

    end = time.time()
    print(f' Time to load data and model: {end - start:.2f} s')

    # fig, ax = plt.subplots(2, 1, figsize=(14, 4), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)

    # sed.plot(ax=ax[0], color='k', alpha=0.2)
    wave = np.squeeze(sed.spec.wave)

    nans = np.isnan(wave) | np.isnan(sed.spec.flux)
    wave[nans] = np.nan
    # save NIRSpec original wavelength grid
    # np.save('wave_NIRSPec.npy', wave)
    # dwave = np.diff(wave)
    # Nbin = 90
    # delattr(sed.spec, 'err') #FIXME: if commented out, use the standard deviation of each bin as error
    # sed.rebin(Nbin)
    sed.resample(wave_step=100)
    # print(stop)
    sed.spec.scatter_overlapping_points(plot=False)
    sed.spec.apply_error_scaling()
    
    # Nbin_spitzer = 3
    sed.load_spitzer(wmax=wmax*1e-3, sigma_clip=5.0, sigma_width=5)
    # sed.plot(ax=ax[0], color='brown', alpha=0.8, lw=1., s=10)

    # Rp = 2.72
    
    parallax_mas = 16.88 # Gaia DR3
    d_pc = 1e3 / parallax_mas # ~ 59.17 pc
    # d_pc = 59.17 
    
    
    free_params = {
                    'teff': (2200, 2700),
                   'logg': (3.0, 4.5), 
                   'R_p': (2.2, 3.5), 
                    'T_d': (100, 900),
                    'R_d': (1, 100),
                    # 'log_s2_nirspec': (-1, 3),
                    # 'log_s2_spitzer': (-1, 3),
                    }
    
    constant_params = {
        'd_pc': d_pc, 
        'resolution': 2700., 
        # 'Nbin': Nbin,
        # 'Nbin_spitzer': Nbin_spitzer,
        # 'teff': 2382, # Cooper+24, Gaia DR3
        }
    sed.set_params(free_params, constant_params)
    
    sed.prior_check(n=2, random=False)

    run = False
    if run:
        sed.PMN_run()
        sed.PMN_eval()
    # try:
    #     sed.PMN_eval()
    # except:
    #     print('No evaluation')
    # sed.PMN_run()
    
    
