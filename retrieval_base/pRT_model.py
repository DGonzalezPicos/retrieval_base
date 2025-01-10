import numpy as np
import time
import pathlib

try:
    import line_profiler
    import memory_profiler
except:
    pass

try:
    from retrieval_base.slab_grid import SlabGrid
except:
    pass

from petitRADTRANS import Radtrans
import petitRADTRANS.nat_cst as nc

from .spectrum import Spectrum, ModelSpectrum

from retrieval_base.auxiliary_functions import get_path, apply_extinction, geom_thin_disk_emission, apply_keplerian_profile, select_species
from broadpy.utils import load_nirspec_resolution_profile
path = get_path()

class pRT_model:
    
    Av = 0.0 # default, no extinction
    disk_species = []

    calc_flux_fast = False
    
    def __init__(self, 
                 line_species, 
                 d_spec, 
                 mode='lbl', 
                 lbl_opacity_sampling=3, 
                 cloud_species=None, 
                 rayleigh_species=['H2', 'He'], 
                 continuum_opacities=['H2-H2', 'H2-He'], 
                 log_P_range=(-6,2), 
                 n_atm_layers=50, 
                 cloud_mode=None, 
                 chem_mode='free', 
                 rv_range=(-50,50), 
                 disk_species=[],
                 disk_kwargs={},
                 T_ex_range=None,
                N_mol_range=None,
                 T_cutoff=None,
                 P_cutoff=None,
                 species_wave={},
                 ):
        '''
        Create instance of the pRT_model class.

        Input
        -----
        line_species : list
            Names of line-lists to include.
        d_spec : DataSpectrum
            Instance of the DataSpectrum class.
        mode : str
            pRT mode to use, can be 'lbl' or 'c-k'.
        lbl_opacity_sampling : int
            Let pRT sample every n-th datapoint.
        cloud_species : list or None
            Chemical cloud species to include. 
        rayleigh_species : list
            Rayleigh-scattering species.
        continuum_opacities : list
            CIA-induced absorption species.
        log_P_range : tuple or list
            Logarithm of modelled pressure range.
        n_atm_layers : int
            Number of atmospheric layers to model.
        cloud_mode : None or str
            Cloud mode to use, can be 'MgSiO3', 'gray' or None.
        chem_mode : str
            Chemistry mode to use for clouds, can be 'free' or 'eqchem'.
        
        '''

        # Read in attributes of the observed spectrum
        self.d_wave          = d_spec.wave
        self.d_mask_isfinite = d_spec.mask_isfinite
        self.d_resolution    = d_spec.resolution
        # self.gratings = np.atleast_1d(set(list(d_spec.gratings)))
        self.gratings = np.unique(d_spec.gratings_list)
        # remove `_f100lp` from 'g140h_f100lp'
        self.gratings = [g.split('_')[0] for g in self.gratings]
        
        print(f' [pRT_model] Gratings: {self.gratings}')
        
        
        if len(self.gratings) > 0:
            self.load_nirspec_gratings()
            

        # self.apply_high_pass_filter = d_spec.high_pass_filtered
        self.apply_high_pass_filter = False
        self.w_set = d_spec.w_set

        self.line_species = line_species
        self.mode = mode
        self.lbl_opacity_sampling = lbl_opacity_sampling

        self.cloud_species     = cloud_species
        self.rayleigh_species  = rayleigh_species
        self.continuum_species = continuum_opacities
        
        self.T_cutoff = T_cutoff # temperature cutoff for custom line opacities
        self.P_cutoff = P_cutoff # pressure cutoff for custom line opacities
        self.species_wave = species_wave # dictionary containing the wavelength ranges for each species
        # Clouds
        # if self.cloud_species is None:
        #     self.do_scat_emis = False
        # else:
        #     self.do_scat_emis = True
        self.do_scat_emis = (self.cloud_species is not None)

        self.cloud_mode = cloud_mode
        self.chem_mode  = chem_mode

        self.rv_max = max(max(np.abs(list(rv_range))), 120.0) # 100 km/s for JWST is ~ 1 pixel

        # Define the atmospheric layers
        if log_P_range is None:
            log_P_range = (-6,2)
        if n_atm_layers is None:
            n_atm_layers = 50
        self.pressure = np.logspace(log_P_range[0], log_P_range[1], n_atm_layers)
        
        print(f' disk_species = {disk_species}')
        self.disk_species = disk_species if disk_species is not None else []
        self.disk_kwargs = disk_kwargs
        # print(f' [pRT_model] T_ex_range = {T_ex_range}')
        # print(f' [pRT_model] N_mol_range = {N_mol_range}')
        # print(f' [pRT_model] d_spec.gratings_list = {d_spec.gratings_list}')
        if (len(self.disk_species) > 0) and (T_ex_range != None):
            print(f' [pRT_model] Disk species: {disk_species}')
            # New approach (Oct 18.): interpolate on a (T_ex, N_mol) grid
            # for disk_species_i  
            self.slab = {}
            # self.slab_wave = {}
            # self.slab_range = {'T_ex': np.array(T_ex_range), 'N_mol': np.array(N_mol_range)}
            print(f' [pRT_model] T_ex_range = {T_ex_range}')
            print(f' [pRT_model] N_mol_range = {N_mol_range}')
            
            slab_wave = []
            for disk_species_i in self.disk_species:
                print(f' [pRT_model] Loading slab grid for {disk_species_i}...')
                self.slab[disk_species_i] = SlabGrid(species=disk_species_i, grating='g395h', path=pathlib.Path(path))
                self.slab[disk_species_i].get_grid(np.array(T_ex_range), np.array(N_mol_range), cache=True)
                self.slab[disk_species_i].load_interpolator()
                self.slab[disk_species_i].wavelength_to_nm()
                # self.slab_interpolator[disk_species_i] = slab.interpolator
                # self.slab_wave[disk_species_i] = slab.wave_grid * 1e3 # [um] --> [nm]
                print(f' [pRT_model] slab setup: Min wave = {np.min(self.slab[disk_species_i].wave_grid)}  Max wave = {np.max(self.slab[disk_species_i].wave_grid)}')
                
                
            
        # Make the pRT.Radtrans objects
        if mode == 'lbl':
            self.get_atmospheres(CB_active=False)
        elif mode == 'c-k':
            self.get_atmospheres_gratings(d_spec.gratings_list, CB_active=False)
            self.gratings = set(d_spec.gratings_list) # unique gratings
            
            
        
            
            
    def get_atmospheres(self, CB_active=False):

        # pRT model is somewhat wider than observed spectrum
        if CB_active:
            self.rv_max = 2000
        wave_pad = 1.1 * self.rv_max/(nc.c*1e-5) * np.nanmax(self.d_wave)

        self.wave_range_micron=np.concatenate(
            (np.nanmin(self.d_wave, axis=(1,2))[None,:]-wave_pad,
                np.nanmax(self.d_wave, axis=(1,2))[None,:]+wave_pad
                )).T
        self.wave_range_micron *= 1e-3
        print('[pRT_model.wave_range_micron] = ', self.wave_range_micron)


        self.atm = []            
        for wave_range_i in self.wave_range_micron:
            
            if len(self.species_wave) > 0:
                line_species_i = select_species(self.line_species,
                                                self.species_wave,
                                                wave_range_i[0]*1e3, # [um] -> [nm]
                                                wave_range_i[1]*1e3)
                # print(f' self.line_species = {self.line_species}')
                # print(f' self.species_wave = {self.species_wave}')
                # print(f' wave_range_i = {wave_range_i}')
                
                
                assert len(line_species_i) > 0, 'No line species in wavelength range'
                print(f' --> ({wave_range_i[0]}, {wave_range_i[1]}): {len(line_species_i)} line species')
            else:
                line_species_i = self.line_species
            # Make a pRT.Radtrans object
            atm_i = Radtrans(
                # line_species=self.line_species, 
                line_species=line_species_i,
                rayleigh_species=self.rayleigh_species, 
                continuum_opacities=self.continuum_species, 
                cloud_species=self.cloud_species, 
                wlen_bords_micron=wave_range_i, 
                mode=self.mode, 
                lbl_opacity_sampling=self.lbl_opacity_sampling, 
                do_scat_emis=self.do_scat_emis
                )

            # Set up the atmospheric layers
            atm_i.setup_opa_structure(self.pressure)
            self.atm.append(atm_i)
            
    def get_atmospheres_gratings(self, gratings, CB_active=False):
        
        assert len(gratings) > 0, 'Must be list of at least one grating'
        wave_pad = 2.0 * self.rv_max/(nc.c*1e-5) * np.nanmax(self.d_wave)
        self.wave_range_micron=np.concatenate(
                    (np.nanmin(self.d_wave, axis=(1,2))[None,:]-wave_pad,
                        np.nanmax(self.d_wave, axis=(1,2))[None,:]+wave_pad
                        )).T
        # print(self.wave_range_micron)
        self.wave_range_micron *= 1e-3

        self.atm = []   
        for g_i, wave_range_i in zip(gratings, self.wave_range_micron):
            line_species_g = [f'{ls}_{g_i}' for ls in self.line_species]
            print(f' Grating {g_i} -> wave range = {wave_range_i}')
            # Make a pRT.Radtrans object
            atm_i = Radtrans(
                line_species=line_species_g, 
                rayleigh_species=self.rayleigh_species, 
                continuum_opacities=self.continuum_species, 
                cloud_species=self.cloud_species, 
                wlen_bords_micron=wave_range_i, 
                # mode=self.mode, 
                mode='c-k', # NEW 2024-07-06: use c-k mode for grating
                lbl_opacity_sampling=None, 
                do_scat_emis=self.do_scat_emis
                )

            # Set up the atmospheric layers
            atm_i.setup_opa_structure(self.pressure)
            if self.T_cutoff is not None:
                self.P_cutoff = getattr(self, 'P_cutoff', (np.min(self.pressure), np.max(self.pressure)))
                atm_i = apply_PT_cutoff(atm_i, *self.T_cutoff, *self.P_cutoff, 
                                        ignore_species=['Na_Sam', 'K_static'], # NEW 2025-01-09: avoid Na and K for low pressure contribution
                                        )
                
            self.atm.append(atm_i)
         
    def __call__(self, 
                 mass_fractions, 
                 temperature, 
                 params, 
                 get_contr=False, 
                 get_full_spectrum=False, 
                 calc_flux_fast=False,
                 ):
        '''
        Create a new model spectrum with the given arguments.

        Input
        -----
        mass_fractions : dict
            Species' mass fractions in the pRT format.
        temperature : np.ndarray
            Array of temperatures at each atmospheric layer.
        params : dict
            Parameters of the current model.
        get_contr : bool
            If True, compute the emission contribution function. 

        Returns
        -------
        m_spec : ModelSpectrum class
            Instance of the ModelSpectrum class. 
        '''

        # Update certain attributes
        self.mass_fractions = mass_fractions.copy()
        # if hasattr(self, 'gratings'):
        #     for k,v in mass_fractions.items():
        #         if k.split('-')[0] in self.line_species:
        #             for g in self.gratings:
        #                 # print(f'{k}_{g}')
        #                 self.mass_fractions[f'{k}_{g}'] = v
            
        self.temperature    = temperature
        self.params = params

        if self.params.get('res') is not None:
            # self.d_resolution = np.atleast_1d(self.params['res'])
            res = self.params['res']
            if isinstance(res, (list, tuple)):
                # self.d_resolution = res
                # take list [a,b] and convert it to [a,a, b,b...]
                self.d_resolution = [r for r in res for _ in range(2)] # every JWST order has two filters
            if isinstance(res, (float, int)):
                self.d_resolution = [res for _ in range(self.d_wave.shape[0])]
        # if self.params.get(f'res_{self.w_set}') is not None:
        #     self.d_resolution = self.params[f'res_{self.w_set}']

        # Add clouds if requested
        self.add_clouds()
        
        # self.disk_emission = ("T_ex" in params.keys())
        self.disk_params = {k: params[k] for k in ['T_ex', 'N_mol', 'A_au', 'dV'] if k in params.keys()}
        
        self.geom_thin_disk_args = {k: params[k] for k in ['T_star', 'R_p', 'R_cav', 'R_out', 'i', 'd_pc', 'q'] if k in params.keys()}
        self.geom_thin_disk_emission = (len(self.geom_thin_disk_args) == 7)
        # print(f' [pRT_model] geom_thin_disk_emission = {self.geom_thin_disk_emission}')

        self.Av = params.get('Av', 0.0)
        # Generate a model spectrum
        m_spec = self.get_model_spectrum(
            get_contr=get_contr, 
            get_full_spectrum=get_full_spectrum,
            fast=calc_flux_fast,
            )

        return m_spec

    def add_clouds(self):
        '''
        Add clouds to the model atmosphere using the given parameters.
        '''

        self.give_absorption_opacity = None
        self.f_sed  = None
        self.K_zz   = None
        self.sigma_g = None

        if self.cloud_mode == 'MgSiO3':

            # Mask the pressure above the cloud deck
            mask_above_deck = (self.pressure <= self.params['P_base_MgSiO3'])

            # Add the MgSiO3 particles
            self.mass_fractions['MgSiO3(c)'] = np.zeros_like(self.pressure)
            self.mass_fractions['MgSiO3(c)'][mask_above_deck] = self.params['X_cloud_base_MgSiO3'] * \
                (self.pressure[mask_above_deck]/self.params['P_base_MgSiO3'])**self.params['f_sed']
            #self.params['K_zz'] = self.params['K_zz'] * np.ones_like(self.pressure)
            self.K_zz = self.params['K_zz'] * np.ones_like(self.pressure)
            self.sigma_g = self.params['sigma_g']

            self.f_sed = {'MgSiO3(c)': self.params['f_sed']}
        
        elif self.cloud_mode == 'gray':
            
            # Gray cloud opacity
            self.give_absorption_opacity = self.gray_cloud_opacity

    def gray_cloud_opacity(self, wave_micron, pressure):
        '''
        Function to be called by petitRADTRANS. 

        Input
        -----
        wave_micron: np.ndarray
            Wavelength in micron.
        pressure: np.ndarray
            Pressure in bar.

        Output
        ------
        opa_gray_cloud: np.ndarray
            Gray cloud opacity for each wavelength and pressure layer.
        '''

        # Create gray cloud opacity, i.e. independent of wavelength
        opa_gray_cloud = np.zeros((len(wave_micron), len(pressure)))

        # Constant below the cloud base
        #opa_gray_cloud[:,pressure >= params['P_base_gray']] = params['opa_base_gray']
        opa_gray_cloud[:,pressure > self.params['P_base_gray']] = 0

        # Opacity decreases with power-law above the base
        mask_above_deck = (pressure <= self.params['P_base_gray'])
        opa_gray_cloud[:,mask_above_deck] = self.params['opa_base_gray'] * \
            (pressure[mask_above_deck]/self.params['P_base_gray'])**self.params['f_sed_gray']

        if self.params.get('cloud_slope') is not None:
            opa_gray_cloud *= (wave_micron[:,None] / 1)**self.params['cloud_slope']

        return opa_gray_cloud

    # @line_profiler.profile
    # @memory_profiler.profile
    def get_model_spectrum(self, get_contr=False, get_full_spectrum=False, fast=False):
        '''
        Generate a model spectrum with the given parameters.

        Input
        -----
        get_contr : bool
            If True, computes the emission contribution 
            and cloud opacity. Updates the contr_em and 
            opa_cloud attributes.
        
        Returns
        -------
        m_spec : ModelSpectrum class
            Instance of the ModelSpectrum class
        '''

        # Loop over all orders
        wave = np.ones_like(self.d_wave) * np.nan
        flux = np.ones_like(self.d_wave) * np.nan
        
        if get_contr:
            self.int_contr_em  = np.zeros_like(self.pressure)
            self.int_opa_cloud = np.zeros_like(self.pressure)
            self.contr_em = np.zeros((*self.d_wave.shape, len(self.pressure)))
            self.int_contr_em_per_order = np.zeros((self.d_wave.shape[0], len(self.pressure)))

        self.CCF, self.m_ACF = [], []
        self.wave_pRT_grid, self.flux_pRT_grid = [], []

        self.m_slab = []
        for i, atm_i in enumerate(self.atm):
            
            # Compute the emission spectrum
            # if i == 0 and fast:
            #     atm_i.calc_flux_init(self.temperature, 
            #         self.mass_fractions, 
            #         gravity=10.0**self.params['log_g'], 
            #         mmw=self.mass_fractions['MMW'], 
            #         Kzz=self.K_zz, 
            #         fsed=self.f_sed, 
            #         sigma_lnorm=self.sigma_g,
            #         give_absorption_opacity=self.give_absorption_opacity, 
            #     )
            # else:
            #     # copy attributes from the previous order
            #     atm_i.copy_flux_init(self.atm[0])
            
            if fast:
                atm_i.calc_flux_fast(
                    self.temperature, 
                    self.mass_fractions, 
                    gravity=10.0**self.params['log_g'], 
                    mmw=self.mass_fractions['MMW'], 
                    Kzz=self.K_zz, 
                    fsed=self.f_sed, 
                    sigma_lnorm=self.sigma_g,
                    give_absorption_opacity=self.give_absorption_opacity, 
                    contribution=get_contr, 
                    )
            else:
                atm_i.calc_flux(
                    self.temperature, 
                    self.mass_fractions, 
                    gravity=10.0**self.params['log_g'], 
                    mmw=self.mass_fractions['MMW'], 
                    Kzz=self.K_zz, 
                    fsed=self.f_sed, 
                    sigma_lnorm=self.sigma_g,
                    give_absorption_opacity=self.give_absorption_opacity, 
                    contribution=get_contr, 
                    )
            # end_cf = time.time()
            # print(f'Order {i} took {end_cf-start_cf:.3f} s to compute the flux')
            wave_i = nc.c / atm_i.freq
            finite = np.isfinite(atm_i.flux)
            assert np.sum(finite) == len(finite), f'NaNs in flux ({np.sum(~finite)} non-finite values)'
            flux_i = np.where(np.isfinite(atm_i.flux), atm_i.flux, 0.0)        
            # [erg cm^{-2} s^{-1} Hz^{-1}] -> [erg cm^{-2} s^{-1} cm^{-1}]
            flux_i = atm_i.flux *  nc.c / (wave_i**2)

            # Convert [erg cm^{-2} s^{-1} cm^{-1}] -> [erg cm^{-2} s^{-1} nm^{-1}]
            # flux_i /= 1e7
            flux_i = flux_i * 1e-7

            # Convert [cm] -> [nm]
            wave_i *= 1e7
            # print(f'[pRT_model.get_model_spectrum] np.nanmean(np.diff(wave_i)) = {1e-3 * np.nanmean(np.diff(wave_i))} um')

            # Convert to observation by scaling with planetary radius
            flux_i *= (
                (self.params.get('R_p', 1.0)*nc.r_jup_mean) / \
                (1e3/self.params['parallax']*nc.pc)
                )**2
            
            if self.Av > 0.0:
                print(f'Applying extinction with Av = {self.Av}')
                flux_i = apply_extinction(flux_i, wave_i * 1e-3, self.Av) # wave in [um]
            

            # Create a ModelSpectrum instance
            m_spec_i = ModelSpectrum(
                wave=wave_i, flux=flux_i, 
                lbl_opacity_sampling=self.lbl_opacity_sampling
                )
            # remove attributes from atm_i: flux (keep freq)
            del atm_i.flux
            
            # Apply radial-velocity shift, rotational/instrumental broadening
            
            grating = self.params['gratings'][i]
            if not hasattr(self, 'fwhms'):
                self.gratings = set(list(self.params['gratings']))
                self.load_nirspec_gratings()
                
            self.fwhms_i = np.interp(wave_i, self.wave_fwhms[grating], self.fwhms[grating])
            # print(f'{i}: {grating} fwhm idx(0,mid,-1) = {fwhms[0]:.1f}, {fwhms[len(fwhms)//2]:.1f}, {fwhms[-1]:.1f}')
            # assert isinstance(fwhms, np.ndarray), f'fwhms has type {type(fwhms)}'
                
            # start_sbr = time.time()
            if self.mode =='lbl':
                
                # testing
                skip_shift_broaden_rebin = False
                if not skip_shift_broaden_rebin:
                    m_spec_i.shift_broaden_rebin(
                        rv=self.params['rv'], 
                        vsini=self.params['vsini'], 
                        epsilon_limb=self.params['epsilon_limb'], 
                        # out_res=self.d_resolution[i], # NEW 2024-05-26: resolution per order
                        # grating=self.params['gratings'][i], # NEW 2024-05-26: grating per order
                        in_res=m_spec_i.resolution, 
                        rebin=False, 
                        instr_broad_fast=False,
                        fwhms=self.fwhms_i,
                        )
            else:
                m_spec_i.rv_shift(rv=self.params['rv'], replace_wave=True)
                
            if get_full_spectrum:
                # Store the spectrum before the rebinning
                self.wave_pRT_grid.append(m_spec_i.wave)
                self.flux_pRT_grid.append(m_spec_i.flux)

            # Rebin onto the data's wavelength grid
            # m_spec_i.rebin(d_wave=self.d_wave[i,:], replace_wave_flux=True)
            # WARNING: add disk emission afte `shift_broaden_rebin` to avoid RV shift inconsistencies....???
            m_slab_i = 0.0
            if len(self.disk_species) > 0 and (self.params['gratings'][i]=='g395h'):
                for ds_i in self.disk_species:
                    # print(f'Adding disk emission for {ds_i}')
                    # grating = self.params['gratings'][i]
                    # # units are correct because input model already multiplied by A_au**2 / d_pc**2 with correct units
                    # factor = self.params[f'A_au_{ds_i}'] / np.pi / self.params['d_pc']**2
                    
                    # f_slab_i = self.slab[ds_i][grating][1,:] * factor # [erg s^-1 cm^-2 um^-1]
                    # w_slab_i = self.slab[ds_i][grating][0,:] * 1e3 # [um] -> [nm]
                    
                    # skip if all values of flux are below 1e-18
                                        
                    disk_params = {attr:self.params.get(f'{attr}_{ds_i}') for attr in ['T_ex', 'N_mol']}
                    disk_params['A_au'] = self.params.get(f'A_au_{ds_i}', self.params['A_au'])
                    disk_params['d_pc'] = self.params['d_pc']
                    rv_disk = self.params.get('rv_disk', 0.0)
                    
                    f_slab_i = self.slab[ds_i].interpolate(**disk_params)
                    if 'i_deg' in self.params.keys():
                        f_slab_i = apply_keplerian_profile(
                                                            self.slab[ds_i].wave_grid,
                                                            f_slab_i,
                                                            np.linspace(self.params['R_cav'], self.params['R_out'], self.disk_kwargs['nr']),
                                                            self.params.get("M_star_Mjup", 20.0),
                                                            inclination_deg=self.params['i_deg'],
                                                            ntheta=self.disk_kwargs['ntheta'],
                                                            nu=self.params.get('nu', 0.0),
                                                            vsys=self.params.get('rv', 0.0),
                                                                )
                    # fill with zeros values beyond the range of the slab model
                    m_flux_slab_i = np.interp(m_spec_i.wave, self.slab[ds_i].wave_grid * (1+(rv_disk/2.998e5)), f_slab_i, right=0.0, left=0.0)
                    assert np.sum(np.isnan(m_flux_slab_i)) == 0, '[pRT_model.get_model_spectrum] line 546: NaNs in m_flux_slab_i'
                    # print(f' [pRT_model] ds_i = {ds_i}  mean(f_slab_i) = {np.mean(f_slab_i)}')

                    m_slab_i += m_flux_slab_i # store for plotting purposes
                    m_spec_i.flux += m_flux_slab_i # add to model flux (already shifted and broadened)
                    
            
                    self.m_slab.append(m_slab_i) # store for plotting purposes
                    
            # print(f' Rebinning onto cenwave = {np.nanmedian(self.d_wave[i,]):.2f} nm from model cenwave = {np.nanmedian(m_spec_i.wave):.2f} nm')
            
            # print(f'[pRT_model] Rebinning...')
            # print(f'[pRT_model] Original wave: len({len(m_spec_i.wave)}, ({m_spec_i.wave.min():.2f}, {m_spec_i.wave.max():.2f}) nm')
            # print(f'[pRT_model] Data wave: len({len(self.d_wave[i,:])}, ({self.d_wave[i,:].min():.2f}, {self.d_wave[i,:].max():.2f}) nm')
            # nans_wave = np.sum(np.isnan(self.d_wave[i,:]))
            # assert nans_wave == 0, f'[pRT_model.get_model_spectrum] line 573: {nans_wave} NaNs in self.d_wave[i,:]'
            # m_spec_i.flux = np.interp(self.d_wave[i,:], m_spec_i.wave, m_spec_i.flux)
            # m_spec_i.wave = self.d_wave[i,:]
            m_spec_i.rebin_spectres(d_wave=self.d_wave[i,:], replace_wave_flux=True, numba=True)
            
            # end_sbr = time.time()   
            # print(f'Order {i} took {end_sbr-start_sbr:.3f} s to shift, broaden and rebin')

            wave[i,:,:] = m_spec_i.wave # nm
            flux[i,:,:] = m_spec_i.flux
            # print(f'{grating} mean(wave) = {m_spec_i.wave.mean()} nm, mean(flux) = {m_spec_i.flux.mean()} erg/s/cm2/nm')
            # print(f' std(flux) = {m_spec_i.flux.std()} erg/s/cm2/nm')
            
            if get_contr:
                # print(f'[pRT_model] Computing emission contribution for order {i}...')
                # Integrate the emission contribution function and cloud opacity
                self.get_integrated_contr_em_and_opa_cloud(
                    atm_i, m_wave_i=wave_i, 
                    d_wave_i=self.d_wave[i,:], 
                    d_mask_i=self.d_mask_isfinite[i], 
                    # d_mask_i=np.isfinite(m_spec_i.flux),
                    m_spec_i=m_spec_i, 
                    order=i
                    )
                    # print(f'ICE for order {i} = {self.int_contr_em}')

        # Create a new ModelSpectrum instance with all orders
        m_spec = ModelSpectrum(
            wave=wave, 
            flux=flux, 
            lbl_opacity_sampling=self.lbl_opacity_sampling, 
            multiple_orders=True, 
            high_pass_filtered=self.apply_high_pass_filter, 
            )
        if get_contr: # Store the integrated emission contribution
            m_spec.int_contr_em = np.copy(self.int_contr_em)
        #     # m_spec.int_contr_em_per_order = np.copy(self.int_contr_em_per_order)
        #     m_spec.contr_em = np.copy(self.contr_e    m)
            # m_spec.int_opa_cloud = self.int_opa_cloud   

        # Convert to arrays
        self.CCF, self.m_ACF = np.array(self.CCF), np.array(self.m_ACF)
        #self.wave_pRT_grid = np.array(self.wave_pRT_grid)
        #self.flux_pRT_grid = np.array(self.flux_pRT_grid)

        # Save memory, same attributes in DataSpectrum
        del m_spec.wave, m_spec.mask_isfinite

        return m_spec

    def get_integrated_contr_em_and_opa_cloud(self, 
                                              atm_i, 
                                              m_wave_i, 
                                              d_wave_i, 
                                              d_mask_i, 
                                              m_spec_i, 
                                              order
                                              ):
        
        # Get the emission contribution function
        contr_em_i = atm_i.contr_em
        n_layers = len(contr_em_i)
        # new_contr_em_i = []

        # Get the cloud opacity
        cloudy = False
        if self.cloud_mode == 'gray':
            opa_cloud_i = self.gray_cloud_opacity(m_wave_i*1e-3, self.pressure).T
            cloudy = True
        elif self.cloud_mode == 'MgSiO3':
            opa_cloud_i = atm_i.tau_cloud.T
            cloudy = True
        else:
            opa_cloud_i = np.zeros_like(contr_em_i)

        # for j, (contr_em_ij, opa_cloud_ij) in enumerate(zip(contr_em_i, opa_cloud_i)):
        for j in range(n_layers):
            contr_em_ij = contr_em_i[j]
            # Similar to the model flux
            contr_em_ij = ModelSpectrum(
                wave=m_wave_i, flux=contr_em_ij, 
                lbl_opacity_sampling=self.lbl_opacity_sampling
                )
            # Shift, broaden, rebin the contribution
            contr_em_ij.shift_broaden_rebin(
                d_wave=d_wave_i, 
                rv=self.params['rv'], 
                vsini=self.params['vsini'], 
                epsilon_limb=self.params['epsilon_limb'], 
                # out_res=self.d_resolution[order], 
                # grating=self.params['gratings'][order],
                # in_res=m_spec_i.resolution, 
                rebin=True, 
                fwhms=self.fwhms_i,
                # instr_broad_fast=True,
                )
            # Compute the spectrally-weighted emission contribution function
            # print(f' shape contr_em_ij.flux = {contr_em_ij.flux.shape}')
            self.contr_em[order,:,:,j] = np.copy(contr_em_ij.flux)
            # Integrate and weigh the emission contribution function
            self.int_contr_em_per_order[order,j] = \
                contr_em_ij.spectrally_weighted_integration(
                    wave=d_wave_i[d_mask_i].flatten(), 
                    flux=m_spec_i.flux[d_mask_i].flatten(), 
                    array=contr_em_ij.flux[d_mask_i].flatten(), 
                    )
            
            self.int_contr_em[j] += self.int_contr_em_per_order[order,j]

            # Similar to the model flux
            if cloudy:
                opa_cloud_ij = opa_cloud_i[j]
                opa_cloud_ij = ModelSpectrum(
                    wave=m_wave_i, flux=opa_cloud_ij, 
                    lbl_opacity_sampling=self.lbl_opacity_sampling
                    )
                # Shift, broaden, rebin the cloud opacity
                opa_cloud_ij.shift_broaden_rebin(
                    d_wave=d_wave_i, 
                    rv=self.params['rv'], 
                    vsini=self.params['vsini'], 
                    epsilon_limb=self.params['epsilon_limb'], 
                    out_res=self.d_resolution[order], 
                    in_res=m_spec_i.resolution, 
                    rebin=True, 
                    )
                # Integrate and weigh the cloud opacity
                self.int_opa_cloud[j] += \
                    opa_cloud_ij.spectrally_weighted_integration(
                        wave=d_wave_i[d_mask_i].flatten(), 
                        flux=m_spec_i.flux[d_mask_i].flatten(), 
                        array=opa_cloud_ij.flux[d_mask_i].flatten(), 
                        )
        return self
    
    def load_nirspec_gratings(self):
        self.wave_fwhms = {}
        self.fwhms = {}
        for g in self.gratings:
            # print(f' Loading resolution profile for grating {g}')
            self.wave_fwhms[g], resolution_g = load_nirspec_resolution_profile(grating=g)
            self.fwhms[g] = 2.998e5 / resolution_g
            
        return self