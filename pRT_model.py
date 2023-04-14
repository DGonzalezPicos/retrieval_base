import numpy as np

from petitRADTRANS import Radtrans
import petitRADTRANS.nat_cst as nc

from spectrum import Spectrum, ModelSpectrum

class pRT_model:

    def __init__(self, 
                 line_species, 
                 wave_range_micron, 
                 mode='lbl', 
                 lbl_opacity_sampling=3, 
                 cloud_species=None, 
                 rayleigh_species=['H2', 'He'], 
                 continuum_opacities=['H2-H2', 'H2-He'], 
                 log_P_range=(-6,2), 
                 n_atm_layers=50, 
                 ):

        self.line_species = line_species
        self.wave_range_micron = wave_range_micron
        self.lbl_opacity_sampling = lbl_opacity_sampling

        self.cloud_species     = cloud_species
        self.rayleigh_species  = rayleigh_species
        self.continuum_species = continuum_opacities

        if self.cloud_species is None:
            do_scat_emis = False
        else:
            do_scat_emis = True

        # Define the atmospheric layers
        self.pressure = np.logspace(log_P_range[0], log_P_range[1], n_atm_layers)

        self.atm = []
        for wave_range_i in self.wave_range_micron:
            
            # Make a pRT.Radtrans object
            atm_i = Radtrans(line_species=self.line_species, 
                             rayleigh_species=self.rayleigh_species, 
                             continuum_opacities=self.continuum_species, 
                             cloud_species=self.cloud_species, 
                             wlen_bords_micron=wave_range_i, 
                             mode=mode, 
                             lbl_opacity_sampling=self.lbl_opacity_sampling, 
                             do_scat_emis=do_scat_emis
                             )

            # Set up the atmospheric layers
            atm_i.setup_opa_structure(self.pressure)
            self.atm.append(atm_i)

    def __call__(self, 
                 mass_fractions, 
                 temperature, 
                 params, 
                 d_wave, 
                 d_wave_bins=None, 
                 d_resolution=1e5, 
                 apply_high_pass_filter=False, 
                 get_contr=False
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
        d_wave : np.ndarray
            Wavelength grid of observed spectrum.
        d_wave_bins : np.ndarray
            Wavelength separation between neighboring pixels.
        d_resolution : float
            Spectral resolution of observed spectrum. 

        Returns
        -------
        m_spec : ModelSpectrum class
            Instance of the ModelSpectrum class. 
        '''

        # Update certain attributes
        self.mass_fractions = mass_fractions
        self.temperature    = temperature
        self.params = params

        # Add clouds if requested
        self.add_clouds()

        # Generate a model spectrum
        m_spec = self.get_model_spectrum(get_contr=get_contr)
        # Apply radial-velocity shift, rotational/instrumental 
        # broadening and rebin onto the data's wavelength grid
        m_spec.shift_broaden_rebin(new_wave=d_wave, 
                                   new_wave_bins=d_wave_bins, 
                                   rv=self.params['rv'], 
                                   vsini=self.params['vsini'], 
                                   epsilon_limb=self.params['epsilon_limb'], 
                                   out_res=d_resolution, 
                                   in_res=m_spec.resolution, 
                                   )
        # Convert to observation by scaling with planetary radius
        m_spec.flux *= (
            (self.params['R_p']*nc.r_jup_mean) / \
            (1e3/self.params['parallax']*nc.pc)
            )**2

        if apply_high_pass_filter:
            # High-pass filter the model spectrum
            m_spec.high_pass_filter(
                removal_mode='divide', 
                filter_mode='gaussian', 
                sigma=300, 
                replace_flux_err=True
                )

        return m_spec


    def add_clouds(self):
        '''
        Add clouds to the model atmosphere using the given parameters.
        '''

        if self.params['log_X_cloud_base_MgSiO3'] is not None:
            
            mask_above_deck = (self.pressure < self.params['P_base_MgSiO3'])

            # Add the MgSiO3 particles
            self.mass_fractions['MgSiO3(c)'] = np.zeros_like(self.pressure)
            self.mass_fractions['MgSiO3(c)'][mask_above_deck] = self.params['X_cloud_base_MgSiO3'] * \
                (self.pressure[mask_above_deck]/self.params['P_base_MgSiO3'])**self.params['f_sed']
            self.params['K_zz'] = self.params['K_zz'] * np.ones_like(self.pressure)

            self.f_seds = {'MgSiO3(c)': self.params['f_sed']}
        
        else:
            self.f_seds = None
        

        if self.params['log_opa_base_gray'] is not None:
            # Gray cloud opacity
            self.give_absorption_opacity = self.gray_cloud_opacity

        else:
            self.give_absorption_opacity = None

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
        mask_above_deck = (pressure < self.params['P_base_gray'])
        opa_gray_cloud[:,mask_above_deck] = self.params['opa_base_gray'] * \
            (pressure[mask_above_deck]/self.params['P_base_gray'])**self.params['f_sed_gray']

        return opa_gray_cloud

    def get_model_spectrum(self, get_contr=False):
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
        wave, flux = [], []
        self.contr_em, self.opa_cloud = [], []
        for i, atm_i in enumerate(self.atm):

            # Compute the emission spectrum
            atm_i.calc_flux(self.temperature, 
                            self.mass_fractions, 
                            gravity=10**self.params['log_g'], 
                            mmw=self.mass_fractions['MMW'], 
                            Kzz=self.params['K_zz'], 
                            fsed=self.f_seds, 
                            sigma_lnorm=self.params['sigma_g'],
                            give_absorption_opacity=self.give_absorption_opacity, 
                            contribution=get_contr, 
                            )
            wave_i = nc.c / atm_i.freq
            flux_i = atm_i.flux

            wave.append(wave_i)
            flux.append(flux_i)
            
            if get_contr:
                # Save the emission contribution function
                self.contr_em.append(atm_i.contr_em)

                # Save the (gray) cloud opacity in cm^2 g^-1
                if self.params['log_opa_base_gray'] is not None:
                    self.opa_cloud.append(self.gray_cloud_opacity(wave_i*1e4, self.pressure))
                else:
                    self.opa_cloud.append(atm_i.tau_cloud)

        # Collapse the arrays
        wave = np.concatenate(wave, axis=0)
        flux = np.concatenate(flux, axis=0)

        # Convert [erg cm^{-2} s^{-1} Hz^{-1}] -> [erg cm^{-2} s^{-1} cm^{-1}]
        flux *= nc.c / (wave**2)

        # Convert [erg cm^{-2} s^{-1} cm^{-1}] -> [erg cm^{-2} s^{-1} nm^{-1}]
        flux /= 1e7

        # Convert [cm] -> [nm]
        wave *= 1e7

        # Return a ModelSpectrum instance
        return ModelSpectrum(wave, flux)