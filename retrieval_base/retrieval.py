import os
os.environ['OMP_NUM_THREADS'] = '1'
from mpi4py import MPI
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

import numpy as np
import copy

import matplotlib.pyplot as plt
import pymultinest

from .spectrum import DataSpectrum, Photometry
from .parameters import Parameters
from .pRT_model import pRT_model
from .log_likelihood import LogLikelihood
from .PT_profile import get_PT_profile_class
from .chemistry import get_Chemistry_class
from .covariance import get_Covariance_class
from .callback import CallBack

import retrieval_base.figures as figs
import retrieval_base.auxiliary_functions as af
from matplotlib.backends.backend_pdf import PdfPages

def pre_processing(conf, conf_data):

    # Set up the output directories
    af.create_output_dir(conf.prefix, conf.file_params)

    # --- Pre-process data ----------------------------------------------

    # Get instances of the DataSpectrum class 
    # for the target and telluric standard
    d_spec = DataSpectrum(
        wave=None, 
        flux=None, 
        err=None, 
        ra=conf_data['ra'], 
        dec=conf_data['dec'], 
        mjd=conf_data['mjd'], 
        pwv=conf_data['pwv'], 
        file_target=conf_data['file_target'], 
        file_wave=conf_data['file_wave'], 
        slit=conf_data['slit'], 
        wave_range=conf_data['wave_range'], 
        w_set=conf_data['w_set'], 
        )
    
    n_edge_pixels = conf_data.get('n_edge_pixels', 30)
    d_spec.clip_det_edges(n_edge_pixels)
    d_spec.load_molecfit_transm(
            conf_data['file_molecfit_transm'], 
            tell_threshold=conf_data['tell_threshold'],
            T=conf_data['T_std'],
            )

    assert hasattr(d_spec, 'throughput'), 'No throughput found in `d_spec`'
    
    
    
    if hasattr(conf, 'magnitudes_std'):
        print(f'Using magnitudes_std: {conf.magnitudes_std}')
        d_std_spec = DataSpectrum(
            wave=None, 
            flux=None, 
            err=None, 
            ra=conf_data['ra_std'], 
            dec=conf_data['dec_std'], 
            mjd=conf_data['mjd_std'], 
            pwv=conf_data['pwv'], 
            file_target=conf_data['file_std'], 
            file_wave=conf_data['file_std'], 
            slit=conf_data['slit'], 
            wave_range=conf_data['wave_range'], 
            w_set=conf_data['w_set'], 
            )
        d_std_spec.clip_det_edges(n_edge_pixels)
        
        
        photom_2MASS = Photometry(magnitudes=conf.magnitudes_std)
        d_std_spec.load_molecfit_transm(
            conf_data['file_std_molecfit_transm'],
            T=conf_data['T_std'],
            tell_threshold=conf_data['tell_threshold']
            )
        d_std_spec.flux_calib_2MASS(
            photom_2MASS, 
            conf_data['filter_2MASS'], 
            tell_threshold=conf_data['tell_threshold'], 
            prefix=conf.prefix, 
            molecfit=True,
            fig_label='std',
            )
        calib_factor = d_std_spec.calib_factor
        if conf_data.get('off_axis_scale'):
            calib_factor *= conf_data['off_axis_scale']
        
        d_spec.flux_uncorr = (d_spec.flux / d_spec.throughput) * calib_factor
        # Apply flux calibration to data
        avoid_zeros = d_spec.transm*d_spec.throughput > 0.01
        d_spec.flux = np.divide(
            d_spec.flux,
            d_spec.transm*d_spec.throughput,
            where=avoid_zeros
            )
        d_spec.err = np.divide(
            d_spec.err,
            d_spec.transm*d_spec.throughput,
            where=avoid_zeros
            )
        
        # Replace deep telluric lines with nans
        d_spec.flux[d_spec.transm < conf_data['tell_threshold']] = np.nan
        d_spec.update_isfinite_mask(d_spec.flux)

        # Multiply by the calibration factor
        d_spec.flux *= calib_factor
        d_spec.err *= calib_factor
        
        # check if key "file_off_axis_correction.dat" in config_data
        if conf_data.get('file_offaxis_correction'):
            print(f'Applying off-axis correction from {conf_data["file_offaxis_correction"]}')
            # load off-axis correction
            wave_offaxis, corr_offaxis = np.loadtxt(conf_data['file_offaxis_correction'], unpack=True)
            assert np.all(wave_offaxis == d_spec.wave), 'Wave grids do not match'
            # apply correction
            d_spec.flux /= corr_offaxis
            d_spec.err /= corr_offaxis
        # Off-axis blaze function correction
        
        # Plot the flux calibration
        figs.fig_flux_calib_2MASS(
            wave=d_spec.wave, 
            calib_flux=d_spec.flux, 
            calib_flux_wo_tell_corr=d_spec.flux_uncorr, 
            #calib_flux_wo_tell_corr=self.flux*calib_factor, 
            #transm=self.transm/self.throughput, 
            transm=d_spec.transm, 
            poly_model=d_spec.throughput, 
            wave_2MASS=photom_2MASS.transm_curves[conf_data['filter_2MASS']].T[0], 
            transm_2MASS=photom_2MASS.transm_curves[conf_data['filter_2MASS']].T[1], 
            tell_threshold=conf_data['tell_threshold'], 
            order_wlen_ranges=d_spec.order_wlen_ranges, 
            prefix=conf.prefix, 
            w_set=d_spec.w_set, 
            T_star=4200.,
            T_companion=2600.,
            )
        
        
    else:

        # Instance of the Photometry class for the given magnitudes
        photom_2MASS = Photometry(magnitudes=conf.magnitudes)


        # Apply flux calibration using the 2MASS broadband magnitude
        d_spec.flux_calib_2MASS(
            photom_2MASS, 
            conf_data['filter_2MASS'], 
            tell_threshold=conf_data['tell_threshold'], 
            prefix=conf.prefix, 
            molecfit=True,
            )
        del photom_2MASS
    
    # Mask emission lines in the target spectrum
    if len(conf.mask_lines) > 0:
        for key, value in conf.mask_lines.items():
            # wave_flat = d_spec.wave.flatten()
            # mask = (wave_flat > value[0]) & (wave_flat < value[1])
            mask = (d_spec.wave > value[0]) & (d_spec.wave < value[1])
            print(f'Masking {key}: {value[0]} - {value[1]} (n={mask.sum()})')

            d_spec.flux[mask] = np.nan
            d_spec.update_isfinite_mask(d_spec.flux)
            

    # Apply sigma-clipping
    d_spec.sigma_clip_median_filter(
        sigma=3, 
        filter_width=conf_data['sigma_clip_width'], 
        prefix=conf.prefix
        )
    #d_spec.sigma_clip_poly(sigma=3, prefix=conf.prefix)

    # Crop the spectrum
    d_spec.crop_spectrum()

    # Remove the ghost signatures
    d_spec.mask_ghosts(wave_to_mask=conf_data.get('wave_to_mask'))

    # Re-shape the spectrum to a 3-dimensional array
    d_spec.reshape_orders_dets()

    # Apply barycentric correction
    d_spec.bary_corr()

    if conf.apply_high_pass_filter:
        # Apply high-pass filter
        d_spec.high_pass_filter(
            removal_mode='divide', 
            filter_mode='gaussian', 
            sigma=300, 
            replace_flux_err=True
            )

    # Prepare the wavelength separation and average squared error arrays
    d_spec.prepare_for_covariance(
        prepare_err_eff=conf.cov_kwargs['prepare_for_covariance']
        )

    # Plot the pre-processed spectrum
    figs.fig_spec_to_fit(
        d_spec, prefix=conf.prefix, w_set=d_spec.w_set
        )

    # Save the data
    np.save(conf.prefix+f'data/d_spec_wave_{d_spec.w_set}.npy', d_spec.wave)
    np.save(conf.prefix+f'data/d_spec_flux_{d_spec.w_set}.npy', d_spec.flux)
    np.save(conf.prefix+f'data/d_spec_err_{d_spec.w_set}.npy', d_spec.err)
    np.save(conf.prefix+f'data/d_spec_transm_{d_spec.w_set}.npy', d_spec.transm)

    np.save(conf.prefix+f'data/d_spec_flux_uncorr_{d_spec.w_set}.npy', d_spec.flux_uncorr)
    del d_spec.flux_uncorr
    del d_spec.transm, d_spec.transm_err

    # Save as pickle
    af.pickle_save(conf.prefix+f'data/d_spec_{d_spec.w_set}.pkl', d_spec)

    # --- Set up a pRT model --------------------------------------------

    # Create the Radtrans objects
    pRT_atm = pRT_model(
        line_species=conf.line_species, 
        d_spec=d_spec, 
        mode='lbl', 
        lbl_opacity_sampling=conf_data['lbl_opacity_sampling'], 
        cloud_species=conf.cloud_species, 
        # rayleigh_species=['H2', 'He'], 
        # continuum_opacities=['H2-H2', 'H2-He'], 
        rayleigh_species=conf.rayleigh_species,
        continuum_opacities=conf.continuum_opacities,
        log_P_range=conf_data.get('log_P_range'), 
        n_atm_layers=conf_data.get('n_atm_layers'), 
        rv_range=conf.free_params['rv'][0], 
        disk_species=conf.disk_species,
        )

    # Save as pickle
    af.pickle_save(conf.prefix+f'data/pRT_atm_{d_spec.w_set}.pkl', pRT_atm)
    
def prior_check(conf, n=3, random=False, get_contr=False, fig_name=None):
    
    ret = Retrieval(conf=conf, evaluation=False)
    w_set = 'NIRSpec'
    # first evaluation the model at 'n' different parameter values
    if random:
        # random values between 0 and 1
        theta = np.random.rand(n)
    else:
        theta = np.linspace(0, 1, n)
    m_spec_list = []
    logL_list = [] 
    beta_list = []
    
    print(f' --> Evaluating the model at {n} different parameter values')
    print(f' ret.Cov[w_set][0,0].cov.shape = {ret.Cov[w_set][0,0].cov.shape}')
    # plot PT
    fig, (ax_PT, ax_grad) = plt.subplots(1,2, figsize=(10,5), sharey=True)
    
    time_list = []
    for i, theta_i in enumerate(theta):
        start = time.time()

        ret.Param(theta_i * np.ones(len(ret.Param.param_keys)))
        sample = {k:ret.Param.params[k] for k in ret.Param.param_keys}
        print(sample)
        ret.evaluation = get_contr
        ln_L = ret.PMN_lnL_func()
        # assert hasattr(ret.m_spec, 'int_contr_em'), f' No integrated contribution emission found in ret.m_spec'
        print(f'ln_L = {ln_L:.4e}\n')
        end = time.time()
        print(f'Elapsed time: {end-start:.2f} s')
        time_list.append(end-start)
        
        if i == 0:
            print(f' shape data flux = {ret.d_spec[w_set].flux.shape}')
            print(f' shape m_spec.flux = {ret.m_spec[w_set].flux.shape}')
            print(f' shape LogLike.m_flux = {ret.LogLike[w_set].m_flux.shape}')
            print(f' shape LogLike.f = {ret.LogLike[w_set].f.shape}')
            # print(f' shape.ret.Cov.cov_cholesky = {ret.Cov[w_set].cov_cholesky.shape}')
            
        # m_spec_list.append(ret.m_spec[w_set])
        m_spec_list.append(ret.LogLike[w_set].m_flux)
        logL_list.append(ln_L)
        beta_list.append(ret.LogLike[w_set].beta)
        
        if get_contr:
            ret.copy_integrated_contribution_emission()
        # PT_list.append(ret.PT)
        figs.fig_PT(ret.PT, ax=ax_PT, ax_grad=ax_grad, 
                    bestfit_color=f'C{i}', 
                    show_knots=(i==0), 
                    fig=fig,
                    fig_name=str(fig_name).replace('.pdf', '_PT.pdf') if i==(len(theta)-1) else None)
        

    print(f' --> Time per evaluation: {np.mean(time_list):.2f} +- {np.std(time_list):.2f} s')
    # use PDF pages to save multiple plots for each order into one PDF
    with PdfPages(fig_name) as pdf:
        for i in range(ret.d_spec[w_set].n_orders):
            fig, ax = plt.subplots(2,1, figsize=(10,5), sharex=True,
                                   gridspec_kw={'height_ratios':[3,1]})
            ax[0].set(ylabel=f'Flux / {ret.d_spec[w_set].flux_unit}')
            ax[-1].set(xlabel='Wavelength / nm', ylabel='Residuals')
            for j in range(ret.d_spec[w_set].n_dets):

                # Apply mask to model and data, calculate residuals
                mask_ij = ret.d_spec[w_set].mask_isfinite[i,j,:]

                # Number of data points
                N_ij = mask_ij.sum()
                if N_ij == 0:
                    print(f'No data points in order {i}, detector {j}')
                    continue
                print(f' order {i}, detector {j}: N = {N_ij}')


                wave_ij = ret.d_spec[w_set].wave[i,j,:]
                flux_ij = ret.d_spec[w_set].flux[i,j,:]
                ax[0].plot(wave_ij, flux_ij, lw=1, label='data', color='k')
                ax[-1].axhline(0, color='k', ls='-', alpha=0.9)
                for k in range(n):
                    m_flux_ij = m_spec_list[k][i,j,:]
                    logL = logL_list[k]
                    ax[0].plot(wave_ij, m_flux_ij, lw=1, ls='--', label=f'logL = {logL:.3e}')
                    ax[-1].plot(wave_ij, flux_ij - m_flux_ij, lw=1, ls='--', 
                                color=ax[0].get_lines()[-1].get_color())
                    print(f' error scaling factor = {beta_list[k][i,j]:.1f}')
                    
            ax[0].legend()
            pdf.savefig(fig)
            plt.close(fig)
        print(f'--> Saved {fig_name}')
        
        return ret
                       
    


class Retrieval:

    plot_ccf = False
    def __init__(self, conf, evaluation, d_spec=None, tmp_path=None):

        self.conf = conf
        # if MPI active, change self.conf.prefix
        if tmp_path is not None:
            # print(f' rank = {rank}')
            # self.conf.prefix = self.conf.prefix + f'rank{rank}/'
            # get $TMP_PATH variable
            print(f' Updating conf.prefix to {tmp_path}retrieval_outputs/{self.conf.run}/test_')
            self.conf.prefix = f'{tmp_path}retrieval_outputs/{self.conf.run}/test_'
            
        self.conf_output = '/'.join(self.conf.prefix.split('/')[:-1])+'/test_output/'+self.conf.prefix.split('/')[-1]
        self.evaluation = evaluation

        self.d_spec  = {}
        self.pRT_atm = {}
        param_wlen_settings = {}
        
        for w_set in conf.config_data.keys():

            # Load the DataSpectrum and pRT_model classes
            self.d_spec[w_set]  = af.pickle_load(self.conf.prefix+f'data/d_spec_{w_set}.pkl')
            self.pRT_atm[w_set] = af.pickle_load(self.conf.prefix+f'data/pRT_atm_{w_set}.pkl')
            param_wlen_settings[w_set] = [self.d_spec[w_set].n_orders, self.d_spec[w_set].n_dets]
            
            # if len(conf.disk_species) > 0 and not hasattr(self.pRT_atm[w_set], 'disk'):
            #     print(f' [pRT_model] Disk species: {conf.disk_species}')
            #     from retrieval_base.slab_model import Disk
                
            #     self.pRT_atm[w_set].disk_species = conf.disk_species
            #     self.pRT_atm[w_set].disk = Disk(molecules=self.pRT_atm[w_set].disk_species,
            #         # wave_range=(wmin, wmax),
            #         wave_range=(4.1,5.3), # WARNING: manually fixed to only cover the CO lines in G395H
            #         wave_step=None,
            #         grating=None,
            #         path_to_moldata=af.get_path()+'data/hitran',
            #         )


                
        if d_spec is not None:  
            # Use the provided DataSpectrum class
            for w_set in d_spec.keys():
                self.d_spec[w_set] = d_spec[w_set]
                param_wlen_settings[w_set] = [self.d_spec[w_set].n_orders, self.d_spec[w_set].n_dets]

        # print(f' param_wlen_settings = {param_wlen_settings}')

        # Create a Parameters instance
        self.Param = Parameters(
            free_params=self.conf.free_params, 
            constant_params=self.conf.constant_params, 
            PT_mode=self.conf.PT_mode, 
            PT_adiabatic=self.conf.PT_kwargs.get('adiabatic', False),
            n_T_knots=self.conf.PT_kwargs.get('n_T_knots', 6),
            enforce_PT_corr=self.conf.PT_kwargs['enforce_PT_corr'], 
            chem_mode=self.conf.chem_mode, 
            cloud_mode=self.conf.cloud_mode, 
            cov_mode=self.conf.cov_mode, 
            wlen_settings=param_wlen_settings, 
            )
        
        self.Cov     = {}
        self.LogLike = {}
        for w_set in conf.config_data.keys():
            
            # Update the cloud/chemistry-mode
            self.pRT_atm[w_set].cloud_mode = self.Param.cloud_mode
            self.pRT_atm[w_set].chem_mode  = self.Param.chem_mode

            self.Cov[w_set] = np.empty(
                (self.d_spec[w_set].n_orders, self.d_spec[w_set].n_dets), dtype=object
                )
            for i in range(self.d_spec[w_set].n_orders):
                for j in range(self.d_spec[w_set].n_dets):
                    
                    # Select only the finite pixels
                    mask_ij = self.d_spec[w_set].mask_isfinite[i,j]

                    if not mask_ij.any():
                        continue
                    
                    separation_ij, err_eff_ij = None, None
                    if hasattr(self.d_spec[w_set], 'flux_eff'):
                        separation_ij = self.d_spec[w_set].separation[i,j]
                        err_eff_ij = self.d_spec[w_set].err_eff[i,j]
                        

                    self.Cov[w_set][i,j] = get_Covariance_class(
                        self.d_spec[w_set].err[i,j,mask_ij], 
                        self.Param.cov_mode, 
                        separation=separation_ij,
                        err_eff=err_eff_ij,
                        # flux_eff=self.d_spec[w_set].flux_eff[i,j], 
                        **self.conf.cov_kwargs
                        )
            del_attrs = ['err', 'separation', 'err_eff', 'flux_eff']
            for attr in del_attrs:
                if hasattr(self.d_spec[w_set], attr):
                    delattr(self.d_spec[w_set], attr)

            self.LogLike[w_set] = LogLikelihood(
                self.d_spec[w_set], 
                n_params=self.Param.n_params, 
                scale_flux=self.conf.scale_flux, 
                scale_err=self.conf.scale_err, 
                scale_flux_eps=getattr(self.conf, 'scale_flux_eps', 0.05),
                )

        self.PT = get_PT_profile_class(
            self.pRT_atm[w_set].pressure, 
            self.Param.PT_mode, 
            **self.conf.PT_kwargs, 
            )
        if 'sonora' in self.conf.PT_kwargs.keys():
            self.PT.sonora = self.conf.PT_kwargs['sonora']
                
        self.Chem = get_Chemistry_class(
            self.pRT_atm[w_set].line_species, 
            self.pRT_atm[w_set].pressure, 
            self.Param.chem_mode, 
            **self.conf.chem_kwargs, 
            )
        
        self.CB = CallBack(
            d_spec=self.d_spec, 
            evaluation=self.evaluation, 
            n_samples_to_use=2000, 
            prefix=self.conf.prefix, 
            posterior_color='k', 
            bestfit_color='limegreen', 
            species_to_plot_VMR=self.conf.species_to_plot_VMR, 
            species_to_plot_CCF=self.conf.species_to_plot_CCF, 
            )

        if (rank == 0) and self.evaluation and self.plot_ccf:
            self.pRT_atm_broad = {}
            for w_set in conf.config_data.keys():
                
                if os.path.exists(self.conf.prefix+f'data/pRT_atm_broad_{w_set}.pkl'):

                    # Load the pRT model
                    self.pRT_atm_broad[w_set] = af.pickle_load(
                        self.conf.prefix+f'data/pRT_atm_broad_{w_set}.pkl'
                        )
                    continue

                # Create a wider pRT model during evaluation
                self.pRT_atm_broad[w_set] = copy.deepcopy(self.pRT_atm[w_set])
                self.pRT_atm_broad[w_set].get_atmospheres(CB_active=True)

                # Save for convenience
                af.pickle_save(
                    self.conf.prefix+f'data/pRT_atm_broad_{w_set}.pkl', 
                    self.pRT_atm_broad[w_set]
                    )

        # Set to None initially, changed during evaluation
        self.m_spec_species  = None
        self.pRT_atm_species = None
        self.LogLike_species = None

    def PMN_lnL_func(self, cube=None, ndim=None, nparams=None):

        time_A = time.time()

        # Param.params dictionary is already updated
        if self.Param.params.get('temperature') is not None:
            # Read the constant temperatures
            temperature = self.Param.params.get('temperature')
            self.PT.temperature = temperature
        
        else:
            # Retrieve the temperatures
            try:
                temperature = self.PT(self.Param.params)
            except:
                # Something went wrong with interpolating
                # temperature = self.PT(self.Param.params)
                print('Something went wrong with interpolating the PT profile')
                return -np.inf

        # if (temperature.min() < 150) and (self.Param.chem_mode=='fastchem'):
        #     # Temperatures too low for reasonable FastChem convergence
        #     return -np.inf
        
        if temperature.min() < 0:
            # Negative temperatures are rejected
            print(f' ret.PT.temperature = {temperature}')
            print('Negative temperatures...')
            return -np.inf

        # Retrieve the ln L penalty (=0 by default)
        ln_L_penalty = 0
        if hasattr(self.PT, 'ln_L_penalty'):
            ln_L_penalty = self.PT.ln_L_penalty

        # Retrieve the chemical abundances
        if self.Param.chem_mode == 'free':
            assert self.Param.VMR_species is not None, 'No VMR species specified'
            mass_fractions = self.Chem(self.Param.VMR_species, self.Param.params)
        elif self.Param.chem_mode in ['eqchem', 'fastchem', 'SONORAchem']:
            mass_fractions = self.Chem(self.Param.params, temperature)

        if not isinstance(mass_fractions, dict):
            # Non-H2 abundances added up to > 1
            print('Non-H2 abundances added up to > 1')
            return -np.inf

        if self.CB.return_PT_mf:
            # Return temperatures and mass fractions during evaluation
            return (temperature, mass_fractions)

        self.m_spec = {}
        ln_L = ln_L_penalty
        for h, w_set in enumerate(list(self.conf.config_data.keys())):
            
            pRT_atm_to_use = self.pRT_atm[w_set]
            if self.evaluation and self.plot_ccf:
                # Retrieve the model spectrum, with the wider pRT model
                pRT_atm_to_use = self.pRT_atm_broad[w_set]
        
            # Retrieve the model spectrum
            self.m_spec[w_set] = pRT_atm_to_use(
                mass_fractions, 
                temperature, 
                self.Param.params, 
                # get_contr=self.CB.active, 
                get_contr=self.evaluation,
                get_full_spectrum=self.evaluation, 
                )
            
            # Add veiling to the model spectrum (NEW, 2024-05-27)
            # assert np.sum(np.isnan(self.m_spec[w_set].flux)) == 0, 'NaNs in model spectrum before adding veiling'
            if "r_0" in self.Param.params.keys():
                self.m_spec[w_set].add_veiling_power_law(self.Param.params["r_0"],
                                                        self.Param.params.get("alpha", 0.0), # 0.0 = constant
                                                        self.d_spec[w_set].wave,
                                                        np.nanmin(self.d_spec[w_set].wave))
            # assert np.sum(np.isnan(self.m_spec[w_set].flux)) == 0, 'NaNs in model spectrum after adding veiling'
            # Add blackbody flux from disk with radius R_d and temperature T_d (inner disk)
            if "R_d" in self.Param.params.keys():
                self.m_spec[w_set].add_blackbody_disk(R=self.Param.params["R_d"],
                                                    T=self.Param.params["T_d"],
                                                    parallax=self.Param.params["parallax"],
                                                    wave_cm=self.d_spec[w_set].wave*1e-7)
  
            # Spline decomposition
            self.N_knots = self.Param.params.get('N_knots', 1)
            if self.N_knots > 1:
                # print(f'Performing spline decomposition with {self.N_knots} knots...')
                # new shape of the flux array --> [n_knots, n_orders, n_dets, n_pixels]
                self.m_spec[w_set].spline_decomposition(self.N_knots, replace_flux=True)
                # print(f'Median flux of the spline decomposition: {np.nanmedian(self.m_spec[w_set].flux)}')
            else:
                # add a dimension to the flux array --> [1, n_orders, n_dets, n_pixels]
                self.m_spec[w_set].flux = self.m_spec[w_set].flux[None,:,:,:]
            
            if self.conf.cov_mode == 'GP': # New, 2024-07-25
                for i in range(self.d_spec[w_set].n_orders):
                    for j in range(self.d_spec[w_set].n_dets):

                        if not self.d_spec[w_set].mask_isfinite[i,j].any():
                            continue

                        # Update the covariance matrix
                        self.Cov[w_set][i,j](
                            self.Param.params, w_set, 
                            order=i, det=j, 
                            **self.conf.cov_kwargs, 
                            )

            self.m_spec[w_set].fit_radius = ('R_p' in self.Param.param_keys)
            # print(f'Fit radius: {self.m_spec[w_set].fit_radius}')
            
            
            # Retrieve the log-likelihood
            ln_L += self.LogLike[w_set](
                self.m_spec[w_set], 
                self.Cov[w_set], 
                is_first_w_set=(h==0), 
                #ln_L_penalty=ln_L_penalty, 
                evaluation=self.evaluation, 
                )
        
        time_B = time.time()
        self.CB.elapsed_times.append(time_B-time_A)

        return ln_L
    
    def parallel_for_loop(self, func, iterable, **kwargs):

        n_iter = len(iterable)
        n_procs = comm.Get_size()
        
        # Number of iterables to compute per process
        perrank = int(n_iter / n_procs) + 1

        # Lower, upper indices to compute for this rank
        low, high = rank*perrank, (rank+1)*perrank
        if rank == comm.Get_size()-1:
            # Final rank has fewer iterations
            high = n_iter

        # Run the function
        returned = []
        for i in range(low, high):
            if i >= len(iterable):
                break
            returned_i = func(iterable[i], **kwargs)
            returned.append(returned_i)

        # Pause until all processes finished
        comm.Barrier()

        # Combine the outputs
        all_returned = comm.gather(returned, root=0)
        if rank != 0:
            return
        
        if not hasattr(returned_i, '__len__'):
            # Only 1 value returned per process
            
            # Concatenate the lists
            flat_all_returned = []
            for sublist in all_returned:
                for item in sublist:
                    flat_all_returned.append(item)

            return flat_all_returned
        
        # Multiple values returned per process
        flat_all_returned = [[] for _ in range(len(returned_i))]
        for sublist_1 in all_returned:
            for sublist_2 in sublist_1:
                for i, item in enumerate(sublist_2):
                    
                    flat_all_returned[i].append(item)

        return flat_all_returned

    def get_PT_mf_envelopes(self, posterior):

        # Return the PT profile and mass fractions
        self.CB.return_PT_mf = True

        # Objects to store the envelopes in
        self.Chem.mass_fractions_posterior = {}
        self.Chem.unquenched_mass_fractions_posterior = {}

        self.Chem.CO_posterior  = []
        self.Chem.FeH_posterior = []

        self.PT.temperature_envelopes = []
                    
        def func(params_i):

            for j, key_j in enumerate(self.Param.param_keys):
                # Update the Parameters instance
                self.Param.params[key_j] = params_i[j]

                if key_j.startswith('log_'):
                    self.Param.params = self.Param.log_to_linear(self.Param.params, key_j)

                if key_j.startswith('invgamma_'):
                    self.Param.params[key_j.replace('invgamma_', '')] = self.Param.params[key_j]

                if key_j.startswith('gaussian_'):
                    self.Param.params[key_j.replace('gaussian_', '')] = self.Param.params[key_j]

            # Update the parameters
            self.Param.read_PT_params(cube=None)
            self.Param.read_uncertainty_params()
            self.Param.read_chemistry_params()
            self.Param.read_cloud_params()

            # Class instances with best-fitting parameters
            returned = self.PMN_lnL_func()
            
            if isinstance(returned, float):
                # PT profile or mass fractions failed
                return None, None, None, None, None

            # Store the temperatures and mass fractions
            temperature_i, mass_fractions_i = returned
            unquenched_mass_fractions_i = None
            if hasattr(self.Chem, 'unquenched_mass_fractions'):
                unquenched_mass_fractions_i = self.Chem.unquenched_mass_fractions

            # Return the temperature, mass fractions, unquenched, C/O ratio and Fe/H
            return temperature_i, mass_fractions_i, unquenched_mass_fractions_i, self.Chem.CO, self.Chem.FeH
        
        # Compute the mass fractions posterior in parallel
        returned = self.parallel_for_loop(func, posterior)

        if returned is None:
            return
        
        self.PT.temperature_posterior, \
        mass_fractions_posterior, \
        unquenched_mass_fractions_posterior, \
        self.Chem.CO_posterior, \
        self.Chem.FeH_posterior \
            = returned
        
        self.PT.temperature_posterior = np.array(self.PT.temperature_posterior)
        self.Chem.CO_posterior  = np.array(self.Chem.CO_posterior)
        self.Chem.FeH_posterior = np.array(self.Chem.FeH_posterior)

        # Create the lists to store mass fractions per line species
        for line_species_i in mass_fractions_posterior[0].keys():

            self.Chem.mass_fractions_posterior[line_species_i] = []

            if unquenched_mass_fractions_posterior[0] is None:
                continue
            self.Chem.unquenched_mass_fractions_posterior[line_species_i] = []

        # Store the mass fractions posterior in the correct order
        for mf_i, unquenched_mf_i in zip(mass_fractions_posterior, unquenched_mass_fractions_posterior):
            
            # Loop over the line species
            for line_species_i in mf_i.keys():

                self.Chem.mass_fractions_posterior[line_species_i].append(
                    mf_i[line_species_i]
                    )

                if unquenched_mf_i is None:
                    continue
                # Store the unquenched mass fractions
                self.Chem.unquenched_mass_fractions_posterior[line_species_i].append(
                    unquenched_mf_i[line_species_i]
                    )


        # Convert profiles to 1, 2, 3-sigma equivalent and median
        q = [0.5-0.997/2, 0.5-0.95/2, 0.5-0.68/2, 0.5, 
             0.5+0.68/2, 0.5+0.95/2, 0.5+0.997/2
             ]            

        # Retain the pressure-axis
        self.PT.temperature_envelopes = af.quantiles(
            self.PT.temperature_posterior, q=q, axis=0
            )

        self.Chem.mass_fractions_envelopes = {}
        self.Chem.unquenched_mass_fractions_envelopes = {}

        for line_species_i in self.Chem.mass_fractions.keys():

            self.Chem.mass_fractions_posterior[line_species_i] = \
                np.array(self.Chem.mass_fractions_posterior[line_species_i])

            self.Chem.mass_fractions_envelopes[line_species_i] = af.quantiles(
                self.Chem.mass_fractions_posterior[line_species_i], q=q, axis=0
                )
            
            if unquenched_mass_fractions_posterior[0] is None:
                continue

        # Store the unquenched mass fractions
        if hasattr(self.Chem, 'unquenched_mass_fractions'):

            for line_species_i in self.Chem.unquenched_mass_fractions.keys():
                self.Chem.unquenched_mass_fractions_posterior[line_species_i] = \
                    np.array(self.Chem.unquenched_mass_fractions_posterior[line_species_i])

                self.Chem.unquenched_mass_fractions_envelopes[line_species_i] = af.quantiles(
                    self.Chem.unquenched_mass_fractions_posterior[line_species_i], q=q, axis=0
                    )

        self.CB.return_PT_mf = False

    def get_species_contribution(self):

        #self.m_spec_species, self.pRT_atm_species = {}, {}

        self.m_spec_species = dict.fromkeys(self.d_spec.keys(), {})
        self.pRT_atm_species = dict.fromkeys(self.d_spec.keys(), {})

        # Assess the species' contribution
        for species_i in self.Chem.species_info.keys():

            line_species_i = self.Chem.read_species_info(species_i, 'pRT_name')
            if line_species_i not in self.Chem.line_species:
                continue
            # Ignore one species at a time
            self.Chem.neglect_species = dict.fromkeys(self.Chem.neglect_species, False)
            self.Chem.neglect_species[species_i] = True

            # Create the spectrum and evaluate lnL
            self.PMN_lnL_func()

            #self.m_spec.flux_only = flux_only
            for w_set in self.d_spec.keys():
                self.m_spec_species[w_set][species_i]  = copy.deepcopy(self.m_spec[w_set])
                self.pRT_atm_species[w_set][species_i] = copy.deepcopy(self.pRT_atm_broad[w_set])

        # Include all species again
        self.Chem.neglect_species = dict.fromkeys(self.Chem.neglect_species, False)

    def get_all_spectra(self, posterior, save_spectra=False):

        if os.path.exists(self.conf.prefix+'data/m_flux_envelope.npy'):
            
            # Load the model spectrum envelope if it was computed before
            flux_envelope = np.load(self.conf.prefix+'data/m_flux_envelope.npy')

            # Convert envelopes to 1, 2, 3-sigma equivalent and median
            q = [0.5-0.997/2, 0.5-0.95/2, 0.5-0.68/2, 0.5, 
                    0.5+0.68/2, 0.5+0.95/2, 0.5+0.997/2
                    ]            
        
            # Retain the order-, detector-, and wavelength-axes
            flux_envelope = af.quantiles(
                np.array(flux_envelope), q=q, axis=0
                )
                
            return flux_envelope
        
        from tqdm import tqdm
        self.evaluation = False

        flux_envelope = np.nan * np.ones(
            (len(posterior), self.d_spec.n_orders, 
            self.d_spec.n_dets, self.d_spec.n_pixels)
            )
        ln_L_per_pixel_posterior = np.nan * np.ones(
            (len(posterior), self.d_spec.n_orders, 
            self.d_spec.n_dets, self.d_spec.n_pixels)
            )
        chi_squared_per_pixel_posterior = np.nan * np.ones(
            (len(posterior), self.d_spec.n_orders, 
            self.d_spec.n_dets, self.d_spec.n_pixels)
            )

        # Sample envelopes from the posterior
        for i, params_i in enumerate(tqdm(posterior)):

            for j, key_j in enumerate(self.Param.param_keys):
                # Update the Parameters instance
                self.Param.params[key_j] = params_i[j]

            # Update the parameters
            self.Param.read_PT_params(cube=None)
            self.Param.read_uncertainty_params()
            self.Param.read_chemistry_params()
            self.Param.read_cloud_params()

            # Create the spectrum
            self.PMN_lnL_func()

            ln_L_per_pixel_posterior[i,:,:,:]        = np.copy(self.LogLike.ln_L_per_pixel)
            chi_squared_per_pixel_posterior[i,:,:,:] = np.copy(self.LogLike.chi_squared_per_pixel)
            
            if not save_spectra:
                continue

            # Scale the model flux with the linear parameter
            flux_envelope[i,:,:,:] = self.m_spec.flux * self.LogLike.f[:,:,None]

            # Add a random sample from the covariance matrix
            for k in range(self.d_spec.n_orders):
                for l in range(self.d_spec.n_dets):

                    # Get the covariance matrix
                    cov_kl = self.LogLike.cov[k,l].get_dense_cov()

                    # Scale with the optimal uncertainty scaling
                    cov_kl *= self.LogLike.beta[k,l].beta**2

                    # Draw a random sample and add to the flux
                    random_sample = np.random.multivariate_normal(
                        np.zeros(len(cov_kl)), cov_kl, size=1
                        )
                    flux_envelope[i,k,l,:] += random_sample[0]
        
        self.evaluation = True

        np.save(self.conf.prefix+'data/ln_L_per_pixel_posterior.npy', ln_L_per_pixel_posterior)
        np.save(self.conf.prefix+'data/chi_squared_per_pixel_posterior.npy', chi_squared_per_pixel_posterior)
    
        if save_spectra:
            # Save the model spectrum envelope
            np.save(self.conf.prefix+'data/m_flux_envelope.npy', flux_envelope)

            # Convert envelopes to 1, 2, 3-sigma equivalent and median
            q = [0.5-0.997/2, 0.5-0.95/2, 0.5-0.68/2, 0.5, 
                 0.5+0.68/2, 0.5+0.95/2, 0.5+0.997/2
                 ]            

            # Retain the order-, detector-, and wavelength-axes
            flux_envelope = af.quantiles(
                np.array(flux_envelope), q=q, axis=0
                )

            return flux_envelope
        
    def PMN_analyze(self):
        
        # Set-up analyzer object
        try:
            analyzer = pymultinest.Analyzer(
                n_params=self.Param.n_params, 
                outputfiles_basename=self.conf.prefix
                )
            stats = analyzer.get_stats()

        except:
            s = self.conf.prefix # hacky way to check if the outputfiles_basename is a directory
            analyzer = pymultinest.Analyzer(
                n_params=self.Param.n_params, 
                # outputfiles_basename='/'.join(s.split('/')[:-1])+'/output/'+s.split('/')[-1]
                outputfiles_basename=self.conf_output,
                )
            stats = analyzer.get_stats()

        # Load the equally-weighted posterior distribution
        posterior = analyzer.get_equal_weighted_posterior()
        posterior = posterior[:,:-1]

        # Read the parameters of the best-fitting model
        bestfit_params = np.array(stats['modes'][0]['maximum a posterior'])
        # print(f' bestfit_params.shape = {bestfit_params.shape}')
        return bestfit_params, posterior
    
    def evaluate_model(self, bestfit_params):
        # Evaluate the model with best-fitting parameters
        for i, key_i in enumerate(self.Param.param_keys):
            # Update the Parameters instance
            self.Param.params[key_i] = bestfit_params[i]
            # print(f' {key_i}: {bestfit_params[i]}')
            if key_i.startswith('log_'):
                self.Param.params = self.Param.log_to_linear(self.Param.params, key_i)

            if key_i.startswith('invgamma_'):
                self.Param.params[key_i.replace('invgamma_', '')] = self.Param.params[key_i]

        # Update the parameters
        # self.Param.read_PT_params(cube=None)
        # self.Param.read_uncertainty_params()
        # self.Param.read_chemistry_params()
        # self.Param.read_cloud_params()
        # self.read_disk_params()
        self.Param.read_params()

        
        # self.Param.params.update(self.Param.constant_params)
        # check for resolution parameters and place them in a list `res`
        # res_keys = [key for key in list(self.Param.params.keys()) if key.startswith('res_')]
        # if len(res_keys) > 0:
        #     self.Param.params['res'] = [self.Param.params[key] for key in res_keys]
        #     print(f' res: {self.Param.params["res"]}')
        # self.Param.read_resolution_params()
        return self
        
        

    def PMN_callback_func(self, 
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

        self.CB.active = True

        if self.evaluation:

            bestfit_params, posterior = self.PMN_analyze()

            # Get the PT and mass-fraction envelopes
            self.get_PT_mf_envelopes(posterior)
            self.Chem.get_VMRs_posterior()
            
            # Get the model flux envelope
            #flux_envelope = self.get_all_spectra(posterior, save_spectra=False)

        else:

            # Read the parameters of the best-fitting model
            bestfit_params = posterior[np.argmax(posterior[:,-2]),:-2]

            # Remove the last 2 columns
            posterior = posterior[:,:-2]

        if rank != 0:
            return

        # Evaluate the model with best-fitting parameters
        # for i, key_i in enumerate(self.Param.param_keys):
        #     # Update the Parameters instance
        #     self.Param.params[key_i] = bestfit_params[i]
        
        #     if key_i.startswith('log_'):
        #         self.Param.params = self.Param.log_to_linear(self.Param.params, key_i)

        #     if key_i.startswith('invgamma_'):
        #         self.Param.params[key_i.replace('invgamma_', '')] = self.Param.params[key_i]

        # # Update the parameters
        # self.Param.read_PT_params(cube=None)
        # self.Param.read_uncertainty_params()
        # self.Param.read_chemistry_params()
        # self.Param.read_cloud_params()
        self.evaluate_model(bestfit_params)

        if self.evaluation and self.plot_ccf:
            # Get each species' contribution to the spectrum
            self.get_species_contribution()

        # Update class instances with best-fitting parameters
        self.PMN_lnL_func()

        self.CB.active = False

        for w_set in self.conf.config_data.keys():
            self.m_spec[w_set].flux_envelope = None

        pRT_atm_to_use = self.pRT_atm
        if self.evaluation and self.plot_ccf:
            # Retrieve the model spectrum, with the wider pRT model
            pRT_atm_to_use = self.pRT_atm_broad
            #self.m_spec.flux_envelope = flux_envelope
            
        # self.CB.plot_summary = True # WARNING: change back to True after debbuging...
        # Call the CallBack class and make summarizing figures
        self.CB(
            self.Param, self.LogLike, self.Cov, self.PT, self.Chem, 
            self.m_spec, pRT_atm_to_use, posterior, 
            m_spec_species=self.m_spec_species, 
            pRT_atm_species=self.pRT_atm_species
            )

    def PMN_run(self):
        
        # Pause the process to not overload memory on start-up
        # time.sleep(0.1*rank*len(self.d_spec))
        time.sleep(1)

        # Run the MultiNest retrieval
        pymultinest.run(
            LogLikelihood=self.PMN_lnL_func, 
            Prior=self.Param, 
            n_dims=self.Param.n_params, 
            outputfiles_basename=self.conf_output, 
            resume=True, 
            verbose=True, 
            const_efficiency_mode=self.conf.const_efficiency_mode, 
            sampling_efficiency=self.conf.sampling_efficiency, 
            n_live_points=self.conf.n_live_points, 
            evidence_tolerance=self.conf.evidence_tolerance, 
            dump_callback=self.PMN_callback_func, 
            n_iter_before_update=self.conf.n_iter_before_update, 
            )

    def synthetic_spectrum(self):
        
        # Update the parameters
        synthetic_params = np.array([
                0.8, # R_p
                #5.5, # log_g
                #5.0, # log_g
                5.25, # log_g
                0.65, # epsilon_limb

                #-3.3, # log_12CO
                #-3.6, # log_H2O
                #-6.2, # log_CH4
                #-6.3, # log_NH3
                #-2.0, # log_C_ratio
                -3.3, # log_12CO
                -3.6, # log_H2O
                -4.9, # log_CH4
                -6.0, # log_NH3
                -5.5, # log_13CO

                41.0, # vsini
                22.5, # rv

                #1300, # T_eff
                1400, # T_eff
        ])

        # Evaluate the model with best-fitting parameters
        for i, key_i in enumerate(self.Param.param_keys):
            # Update the Parameters instance
            self.Param.params[key_i] = synthetic_params[i]

        # Update the parameters
        self.Param.read_PT_params(cube=None)
        self.Param.read_uncertainty_params()
        self.Param.read_chemistry_params()
        self.Param.read_cloud_params()

        # Create the synthetic spectrum
        self.PMN_lnL_func(cube=None, ndim=None, nparams=None)

        # Save the PT profile
        np.savetxt(self.conf.prefix+'data/SONORA_temperature.dat', self.PT.temperature)
        np.savetxt(self.conf.prefix+'data/SONORA_RCB.dat', np.array([self.PT.RCB]))
        
        # Insert the NaNs from the observed spectrum
        self.m_spec.flux[~self.d_spec.mask_isfinite] = np.nan

        # Add noise to the synthetic spectrum
        self.m_spec.flux = np.random.normal(self.m_spec.flux, self.d_spec.err)

        # Replace the observed spectrum with the synthetic spectrum
        self.d_spec.flux = self.m_spec.flux.copy()

        # Plot the pre-processed spectrum
        figs.fig_spec_to_fit(self.d_spec, prefix=self.conf.prefix)

        # Save the synthetic spectrum
        np.save(self.conf.prefix+'data/d_spec_wave.npy', self.d_spec.wave)
        np.save(self.conf.prefix+'data/d_spec_flux.npy', self.d_spec.flux)
        np.save(self.conf.prefix+'data/d_spec_err.npy', self.d_spec.err)
        np.save(self.conf.prefix+'data/d_spec_transm.npy', self.d_spec.transm)

        # Save as pickle
        af.pickle_save(self.conf.prefix+'data/d_spec.pkl', self.d_spec)
        
    def copy_integrated_contribution_emission(self):
        for w_set in self.conf.config_data.keys():
            if hasattr(self.pRT_atm[w_set], 'int_contr_em'):
                print(f'Copying integrated contribution emission from pRT_atm to PT')
                self.PT.int_contr_em[w_set] = np.copy(self.pRT_atm[w_set].int_contr_em)
            if hasattr(self.m_spec[w_set], 'int_contr_em'):
                print(f'Copying integrated contribution emission from m_spec to PT')
                self.PT.int_contr_em[w_set] = np.copy(self.m_spec[w_set].int_contr_em)
            else:
                print(f'WARNING: No integrated contribution emission found in pRT_atm or m_spec')
        return self
    
    def list_memory_allocation(self, obj=None, min_size_mb=0.1):
        from pympler import asizeof
        
        if obj is None:
            obj = self
        print(f"Memory usage of object {obj}: {asizeof.asizeof(obj) / (1024**2):.2f} MB")
        memory_dict = {k:asizeof.asizeof(v) for k,v in obj.__dict__.items() if not k.startswith('__')}
        # sort by memory usage
        memory_dict = dict(sorted(memory_dict.items(), key=lambda item: item[1], reverse=True))
        for k, v in memory_dict.items():
            v_MB = v / (1024 ** 2)
            if v_MB < min_size_mb:
                continue
            print(f'{k}: {v_MB:.2f} MB')
            try:
                memory_dict_attr = {key:asizeof.asizeof(val) for key,val in getattr(obj, k).__dict__.items() if not key.startswith('__')}
                memory_dict_attr = dict(sorted(memory_dict_attr.items(), key=lambda item: item[1], reverse=True))
                for key, val in memory_dict_attr.items():
                    val_MB = val / (1024 ** 2)
                    if val_MB < min_size_mb:
                        continue
                    print(f'  {key}: {val_MB:.2f} MB')
            except:
                continue
            
        # highest allocation
        max_key = max(memory_dict, key=memory_dict.get)
        print(f'Highest allocation: {max_key} ({memory_dict[max_key] / (1024 ** 2):.2f} MB)')
        
        # break down usage of the highest allocation into sub-attributes
        print(f'Breakdown of {max_key}:')
    
        if hasattr(getattr(obj, max_key), '__dict__'):
            memory_dict = {k:asizeof.asizeof(v) for k,v in getattr(obj, max_key).__dict__.items() if not k.startswith('__')}
            memory_dict = dict(sorted(memory_dict.items(), key=lambda item: item[1], reverse=True))
            for k, v in memory_dict.items():
                v_MB = v / (1024 ** 2)
                if v_MB < min_size_mb:
                    continue
                print(f'  {k}: {v_MB:.2f} MB')
        
        cpu_GB = memory_dict[max_key] / (1024 ** 3)
        genoa_use = 192*cpu_GB
        print(f'Highest allocation: {max_key} * 192 ({genoa_use:.1f}) GB')
        if genoa_use > 336.0:
            print(f' WARNING: Genoa memory usage exceeds 336 GB')
            print(f' --> Maximum number of CPUs: {int(336.0/cpu_GB)}')
            
        return self
    
    def gradient_based_optimization(self, method='L-BFGS-B', bounds=None, options=None):
        from scipy.optimize import minimize
        
        # if bounds is None:
        #     bounds = self.Param.bounds
        
        # if options is None:
        #     options = {'disp': True}
            
        # initial guess
        x0 = 0.5 * np.ones(self.Param.n_params)
        
        # Update the parameters
        # self.Param.read_params()
        bounds = np.array(list(self.Param.param_priors.values()))
        # Run the optimization
        res = minimize(self.PMN_lnL_func, x0, method=method, bounds=bounds, options=options)
        # show the results
        print(res)
        return res