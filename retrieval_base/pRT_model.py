import numpy as np
import time

try:
    import line_profiler
    import memory_profiler
except:
    pass

from petitRADTRANS import Radtrans
import petitRADTRANS.nat_cst as nc

from .spectrum import Spectrum, ModelSpectrum

from retrieval_base.auxiliary_functions import get_path
path = get_path()

class pRT_model:

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
        # self.apply_high_pass_filter = d_spec.high_pass_filtered
        self.apply_high_pass_filter = False
        self.w_set = d_spec.w_set

        self.line_species = line_species
        self.mode = mode
        self.lbl_opacity_sampling = lbl_opacity_sampling

        self.cloud_species     = cloud_species
        self.rayleigh_species  = rayleigh_species
        self.continuum_species = continuum_opacities

        # Clouds
        if self.cloud_species is None:
            self.do_scat_emis = False
        else:
            self.do_scat_emis = True

        self.cloud_mode = cloud_mode
        self.chem_mode  = chem_mode

        self.rv_max = max(np.abs(list(rv_range)))

        # Define the atmospheric layers
        if log_P_range is None:
            log_P_range = (-6,2)
        if n_atm_layers is None:
            n_atm_layers = 50
        self.pressure = np.logspace(log_P_range[0], log_P_range[1], n_atm_layers)
        
        # print(f' disk_species = {disk_species}')
        self.disk_species = disk_species
        if len(self.disk_species) > 0:
            print(f' [pRT_model] Disk species: {disk_species}')
            # import iris as iris
            # from iris import setup
            # from iris import spectrum as sp
            from retrieval_base.slab_model import Disk
            
            # self.disk_species = disk_species
            self.disk = Disk(molecules=self.disk_species,
                # wave_range=(wmin, wmax),
                wave_range=(4.1,5.3), # WARNING: manually fixed to only cover the CO lines in G395H
                wave_step=None,
                grating=None,
                path_to_moldata=path+'data/hitran',
                )

        # Make the pRT.Radtrans objects
        if mode == 'lbl':
            self.get_atmospheres(CB_active=False)
        elif mode == 'c-k':
            self.get_atmospheres_gratings(d_spec.gratings_list, CB_active=False)
            self.gratings = set(d_spec.gratings_list) # unique gratings
            
            
        
            
            
    def get_atmospheres(self, CB_active=False):

        # pRT model is somewhat wider than observed spectrum
        if CB_active:
            self.rv_max = 1000
        wave_pad = 1.1 * self.rv_max/(nc.c*1e-5) * np.nanmax(self.d_wave)

        # self.wave_range_micron = np.concatenate(
        #     (self.d_wave.min(axis=(1,2))[None,:]-wave_pad, 
        #      self.d_wave.max(axis=(1,2))[None,:]+wave_pad
        #     )).T
        self.wave_range_micron=np.concatenate(
            (np.nanmin(self.d_wave, axis=(1,2))[None,:]-wave_pad,
                np.nanmax(self.d_wave, axis=(1,2))[None,:]+wave_pad
                )).T
        print(self.wave_range_micron)
        self.wave_range_micron *= 1e-3

        self.atm = []            
        for wave_range_i in self.wave_range_micron:
            
            # Make a pRT.Radtrans object
            atm_i = Radtrans(
                line_species=self.line_species, 
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
            self.atm.append(atm_i)
         
    def __call__(self, 
                 mass_fractions, 
                 temperature, 
                 params, 
                 get_contr=False, 
                 get_full_spectrum=False, 
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
        if hasattr(self, 'gratings'):
            for k,v in mass_fractions.items():
                if k.split('-')[0] in self.line_species:
                    for g in self.gratings:
                        # print(f'{k}_{g}')
                        self.mass_fractions[f'{k}_{g}'] = v
            
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

        # Generate a model spectrum
        m_spec = self.get_model_spectrum(
            get_contr=get_contr, 
            get_full_spectrum=get_full_spectrum
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
    def get_model_spectrum(self, get_contr=False, get_full_spectrum=False):
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

        for i, atm_i in enumerate(self.atm):
            
            # Compute the emission spectrum
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

            # Convert to observation by scaling with planetary radius
            flux_i *= (
                (self.params.get('R_p', 1.0)*nc.r_jup_mean) / \
                (1e3/self.params['parallax']*nc.pc)
                )**2
            
            # then broaden and resample together with the model spectrum
            # print(f' [pRT_model] wave_i[0] = {wave_i[0]}')
            if (len(self.disk_species)>0) and wave_i[0] > 4000.0: # only for G395H reddest filter
                # Compute the disk emission
                # Add the disk emission to the model spectrum
                # print(f' [pRT_model] Computing disk emission for order {i}...')
                self.disk.set_fine_wgrid(wave_i * 1e-3)
                # disk params must be a dictionary containing (at least): T_ex, N_mol, A_au, dV
                disk_keys = ['T_ex', 'N_mol', 'A_au', 'dV', 'd_pc']
                # assert all([k in self.disk_params.keys() for k in disk_keys]), \
                #     'Disk parameters must contain T_ex, N_mol, A_au, dV'
                    
                disk_dict = {k: self.params[k] for k in disk_keys}
                # print(f' disk_dict = {disk_dict}')
                
                # print(f' [pRT_model] self.disk.slab.distance = {self.disk.slab.distance}')
                # print(f' [pRT_model] self.disk.slab.A_au = {self.disk.slab.A_au}')
                # print(f' [pRT_model] self.disk.slab.T_ex = {self.disk.slab.T_ex}')
                # print(f' [pRT_model] self.disk.slab.N_mol = {self.disk.slab.N_mol}')
                # print(f' [pRT_model] self.disk.slab.dV = {self.disk.slab.dV}')
                flux_disk = self.disk(disk_dict,
                                    wave=None)
                # print(f' [pRT_model] flux_disk.shape = {flux_disk.shape}')
                # print(f' [pRT_model] mean(flux_disk) = {np.mean(flux_disk)}')
                flux_i += flux_disk

            # Create a ModelSpectrum instance
            m_spec_i = ModelSpectrum(
                wave=wave_i, flux=flux_i, 
                lbl_opacity_sampling=self.lbl_opacity_sampling
                )
            # remove attributes from atm_i: flux (keep freq)
            del atm_i.flux
            
            # Apply radial-velocity shift, rotational/instrumental broadening
            # start_sbr = time.time()
            if self.mode =='lbl':
                m_spec_i.shift_broaden_rebin(
                    rv=self.params['rv'], 
                    vsini=self.params['vsini'], 
                    epsilon_limb=self.params['epsilon_limb'], 
                    # out_res=self.d_resolution[i], # NEW 2024-05-26: resolution per order
                    grating=self.params['gratings'][i], # NEW 2024-05-26: grating per order
                    in_res=m_spec_i.resolution, 
                    rebin=False, 
                    instr_broad_fast=False,
                    )
            else:
                m_spec_i.rv_shift(rv=self.params['rv'], replace_wave=True)
                
            if get_full_spectrum:
                # Store the spectrum before the rebinning
                self.wave_pRT_grid.append(m_spec_i.wave)
                self.flux_pRT_grid.append(m_spec_i.flux)

            # Rebin onto the data's wavelength grid
            # m_spec_i.rebin(d_wave=self.d_wave[i,:], replace_wave_flux=True)
            m_spec_i.rebin_spectres(d_wave=self.d_wave[i,:], replace_wave_flux=True, numba=True)
            # end_sbr = time.time()   
            # print(f'Order {i} took {end_sbr-start_sbr:.3f} s to shift, broaden and rebin')
            
            if self.apply_high_pass_filter:
                # High-pass filter the model spectrum
                m_spec_i.high_pass_filter(
                    removal_mode='divide', 
                    filter_mode='gaussian', 
                    sigma=300, 
                    replace_flux_err=True
                    )

            wave[i,:,:] = m_spec_i.wave
            flux[i,:,:] = m_spec_i.flux
            
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
                grating=self.params['gratings'][order],
                in_res=m_spec_i.resolution, 
                rebin=True, 
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


class SlabModel(pRT_model):
    """ 1D model for disk emission at a given temperature and pressure """
    
    