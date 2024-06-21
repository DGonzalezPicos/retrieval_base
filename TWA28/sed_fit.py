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

os.environ['OMP_NUM_THREADS'] = '1'
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
import pymultinest


from retrieval_base.spectrum_jwst import SpectrumJWST, ModelSpectrum
from retrieval_base.btsettl import BTSettl

class SED:
    
    base_path = pathlib.Path('/home/dario/phd')
    path = base_path /'retrieval_base'
    target = 'TWA28'
    
    bestfit_color = 'brown'
    
    def __init__(self, grisms=[], run='test_1', evaluation=False):
        
        self.grisms = grisms
        assert all([g in ['g140h-f100lp', 'g235h-f170lp', 'g395h-f290lp'] for g in grisms]), 'Invalid grism'
        
        self.files = [f'jwst/{target}_{g}.fits' for g in grisms]
        
        self.run = run
        self.run_path = self.path / 'SED_runs' / self.run
        self.run_path.mkdir(parents=True, exist_ok=True)
        self.figs_path = self.run_path / 'figs'
        self.figs_path.mkdir(parents=True, exist_ok=True)
        self.evaluation = evaluation
        self.prefix = str(self.run_path  / 'pmn_')
        self.cb_count = -1
        
                
    def load_spec(self, Nedge=50, Nbin=None):
        
        self.spec = SpectrumJWST(Nedge=Nedge).load_grisms(self.files)
        self.spec.reshape(self.spec.n_orders, 1)
        # spec.fix_wave_nans() # experimental...
        self.spec.sigma_clip_reshaped(use_flux=False, 
                                    sigma=3, 
                                    width=31, 
                                    max_iter=5,
                                    fun='median', 
                                    debug=False)
        self.spec.squeeze()
        self.wave_full = self.spec.wave.copy()

        nans = np.isnan(self.wave_full) | np.isnan(self.spec.flux)
        self.wave_full[nans] = np.nan
        self.Nbin = Nbin
        if self.Nbin is not None:
            self.spec.rebin(nbin=self.Nbin)
            
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
    
    def init_BTSettl(self, wmin=0.7, wmax=5.5, wave_unit='nm'):
        
        self.bt = BTSettl()
        self.bt.load_dataset(self.bt.path / 'BTSETTL_NIRSPec.nc')
        self.bt.set_interpolator()
        return self

    
    def rebin(self, Nbin):
            
        assert Nbin > 0, 'Invalid Nbin'
        self.Nbin = Nbin
        self.spec.rebin(nbin=self.Nbin)
        return self
    

            
    def plot(self, ax=None, **kwargs):
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    
        if len(self.spec.flux.shape) > 1:
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
        
        # essential_keys = ['teff', 'logg', 'R_p', 'd_pc']
        # assert all([key in self.params.keys() for key in essential_keys]), f'Missing essential keys {essential_keys}'
        
        self.add_disk = ('T_d' in self.params.keys()) and ('R_d' in self.params.keys())
        return self
    
    def prior(self, cube=None, ndim=None, nparams=None):
        """prior function for pymultinest"""
        
        for i, key in enumerate(self.free_params_keys):
            a, b = self.free_params[key]
            cube[i] = a + (b - a) * cube[i]
            self.params[key] = cube[i]
            
        
        return cube
    
    def prior_check(self, n=4, random=False):
        """Check prior function"""
        
        fig, ax = plt.subplots(2,1, figsize=(14, 4), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        self.plot(ax=ax[0], color='k', alpha=0.2)
        # color palette from seaborn
        colors = sns.color_palette('deep')
        for i in range(n):
            self.prior(cube=np.random.rand(self.ndim), ndim=self.ndim, nparams=None)
            start = time.time()
            lnL = sed.loglike()
            end = time.time()

            print(f' Paramaters: {sed.params}')
            print(f' lnL = {lnL:.2e} (t={end-start:.2f} s)')
            
            self.bt_copy.plot(ax=ax[0], color=colors[i], alpha=0.8, lw=1., s=10)
            

            res = self.spec.flux - self.bt_copy.flux
            ax[1].scatter(self.spec.wave, res, color=colors[i], s=10)
            
        ax[1].axhline(0, color='k', ls='-', lw=0.4)

        # make ylims of ax[1] symmetric
        ylim = np.nanmax(np.abs(ax[1].get_ylim()))
        ax[1].set_ylim(-ylim, ylim)
        # plt.show()
        fig.savefig(self.figs_path / 'prior_check.pdf')
        print(f'Saved {self.figs_path / "prior_check.pdf"}')
        plt.close(fig)
        return None
        
                    
        
    
    def loglike(self, cube=None, ndim=None, nparams=None):
        """log likelihood function for pymultinest"""

        self.bt_copy = self.bt.copy()
        self.bt_copy.get_spec(teff=self.params['teff'], logg=self.params['logg'])
        self.bt_copy.scale_flux(R_jup=self.params['R_p'], d_pc=self.params['d_pc'])

        assert np.sum(np.isnan(self.bt_copy.flux)) == 0, 'NaNs in BTSettl flux'

        new_bt_flux = np.nan * np.ones_like(self.wave_full)
        for i in range(self.spec.n_orders):
            nans_i = np.isnan(self.wave_full[i,])
            # print(f' nans_i = {nans_i.sum()}')
            new_bt_flux[i,~nans_i] = np.interp(self.wave_full[i,~nans_i], self.bt_copy.wave, self.bt_copy.flux)

        self.bt_copy.flux = new_bt_flux
        self.bt_copy.wave = self.wave_full
        self.bt_copy.rebin(self.params['Nbin'])
        
        if self.add_disk:
            self.bt_copy.blackbody_disk(T=self.params['T_d'], R=self.params['R_d'], d=self.params['d_pc'], add_flux=True)
            
        N = np.array([np.sum(~np.isnan(self.spec.flux[i,])) for i in range(self.spec.n_orders)])
        
        # chi2 = 
        self.m = self.bt_copy.flux # store model
        old_way = True
        if old_way:
            # residuals
            res = self.spec.flux - self.bt_copy.flux
            # chi2 with self.spec.err
            chi2 = np.nansum(res**2 / self.spec.err**2, axis=1)
            # optimal uncertainty scaling to get chi2 = N
            self.s2 = chi2 / N
            # print(f' Error scaling `s` = {np.sqrt(s2)}')
            lnL = np.sum(-0.5 * N * (np.log(2*np.pi) + np.log(self.s2)) - 0.5 * chi2)    
        if np.isnan(lnL):
            lnL = -np.inf
        return lnL
    
    
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
        l, b, w, h = [0.49,3.4,0.50,0.20]

        ax_res_dim  = [l, b*(h+0.03), w, 0.97*h/5]
        ax_spec_dim = [l, ax_res_dim[1]+ax_res_dim[3], w, 4*0.97*h/5]
        ax_spec = fig.add_axes(ax_spec_dim)
        ax_res = fig.add_axes(ax_res_dim)
        self.plot(ax=ax_spec, color='k', alpha=0.2)
        chi2 = np.nansum((self.spec.flux - self.bt_copy.flux)**2 / (self.spec.err)**2)
        if hasattr(self.spec, 'err'):
            ax_spec.errorbar(self.spec.wave.flatten(), self.spec.flux.flatten(), 
                             yerr=(np.sqrt(self.s2)[:,None] * self.spec.err).flatten(),
                             fmt='o', color='k', alpha=0.2, zorder=0) 
            
        self.bt_copy.plot(ax=ax_spec, color='brown', alpha=0.8, lw=1., s=10, label='BT-Settl + Disk' + f' ($\chi^2$={chi2:.2f})')

        if hasattr(self.bt_copy, 'flux_disk'):
            # self.bt_copy.plot(ax=ax_spec, color='darkorange', alpha=0.8, lw=1., s=10)
            ax_spec.plot(self.bt_copy.wave.flatten(), self.bt_copy.flux_disk.flatten(), color='darkorange', alpha=0.8, lw=1., label='Disk')
            ax_spec.plot(self.bt_copy.wave.flatten(), (self.bt_copy.flux - self.bt_copy.flux_disk).flatten(), color='b', alpha=0.5, lw=1., ls='--',
                         label='BT-Settl')
        
        res = self.spec.flux - self.bt_copy.flux
        ax_res.scatter(self.spec.wave, res, color='k', s=10)
        ax_res.axhline(0, color='k', ls='-', lw=0.4)
        ax_res.set_xlabel(r'Wavelength [nm]')
        ax_res.set_ylabel(r'Residuals')
        ylim = np.nanmax(np.abs(ax_res.get_ylim()))
        ax_res.set_ylim(-ylim, ylim)
        ax_spec.legend()
        ax_spec.set_ylabel(r'Flux [erg s$^{-1}$ cm$^{-2}$ nm$^{-1}$]')
        

        
        
        # corner.overplot_lines(fig, bestfit_params, color=self.bestfit_color, lw=0.5)
        fig_label = 'final' if self.evaluation else f'{self.cb_count}'
        outfig = self.figs_path / f'summary_{fig_label}.pdf'
        fig.savefig(outfig)
        print(f'Saved {outfig}\n')
        plt.close(fig)
        
    def set_PMN_hyperparameters(self,
                                n_live_points=100,
                                evidence_tolerance=0.5,
                                n_iter_before_update=200,
                                sampling_efficiency=0.30,
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
      
    
if __name__ == '__main__':
    
    
    path = pathlib.Path('/home/dario/phd/retrieval_base')
    target = 'TWA28'
    cwd = os.getcwd()
    if target not in cwd:
        print(f'Changing directory to {target}')
        os.chdir(target)

    run = 'fixed_teff_2'


    grisms = [
                # 'g140h-f100lp', 
                'g235h-f170lp', 
                'g395h-f290lp',
                ]

    Nedge = 40
    start = time.time()
    sed = SED(grisms, run=run).load_spec(Nedge=Nedge)
    sed.init_BTSettl(wmin=0.95 * np.nanmin(sed.spec.wave)*1e-3, wmax=1.05 * np.nanmax(sed.spec.wave)*1e-3, wave_unit='nm')
    sed.mask_wave(wmin=2250)

    end = time.time()
    print(f' Time to load data and model: {end - start:.2f} s')

    # fig, ax = plt.subplots(2, 1, figsize=(14, 4), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)

    # sed.plot(ax=ax[0], color='k', alpha=0.2)
    wave = np.squeeze(sed.spec.wave)

    nans = np.isnan(wave) | np.isnan(sed.spec.flux)
    wave[nans] = np.nan
    # save NIRSpec original wavelength grid
    # np.save('wave_NIRSPec.npy', wave)
    Nbin = 100
    delattr(sed.spec, 'err')
    sed.rebin(Nbin)
    # sed.plot(ax=ax[0], color='brown', alpha=0.8, lw=1., s=10)

    # Rp = 2.72
    
    parallax_mas = 16.9 # Gaia DR3
    d_pc = 1e3 / parallax_mas # ~ 59.17 pc
    # d_pc = 59.17 
    
    
    free_params = {
                    # 'teff': (2000, 2800),
                   'logg': (2.5, 4.5), 
                   'R_p': (2.3, 3.5), 
                    'T_d': (100, 1000),
                    'R_d': (1, 100)}
    constant_params = {
        'd_pc': d_pc, 
        'resolution': 3000., 
        'Nbin': Nbin,
        'teff': 2382, # Cooper+24, Gaia DR3
        }
    sed.set_params(free_params, constant_params)
    
    sed.prior_check(n=4, random=False)

    run = True
    if run:
        sed.PMN_run()
    # try:
    #     sed.PMN_eval()
    # except:
    #     print('No evaluation')
    # sed.PMN_run()
    sed.PMN_eval()