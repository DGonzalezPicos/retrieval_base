import numpy as np
from scipy.stats import invgamma, norm

from .chemistry import Chemistry

class Parameters:

    # Dictionary of all possible parameters and their default values
    all_params = {

        # Uncertainty scaling
        'a': 0, 'l': 1, 
        # 'a_f': 0, 'l_f': 1, 
        'beta': 1,  

        # Cloud properties (cloud base from condensation)
        'f_sed': None, 
        'log_K_zz': None, 
        'sigma_g': None, 
    }

    def __init__(
            self, 
            free_params, 
            constant_params, 
            PT_mode='free', 
            PT_adiabatic=True,
            n_T_knots=None, 
            enforce_PT_corr=False, 
            chem_mode='free', 
            cloud_mode='gray', 
            cov_mode=None, 
            wlen_settings={
                'J1226': [9,3], 'K2166': [7,3], 
                }
            ):

        # Separate the prior range from the mathtext label
        self.param_priors, self.param_mathtext = {}, {}
        for key_i, (prior_i, mathtext_i) in free_params.items():
            self.param_priors[key_i]   = prior_i
            self.param_mathtext[key_i] = mathtext_i

        self.param_keys = np.array(list(self.param_priors.keys()))
        self.n_params = len(self.param_keys)
        
        # self.initial_guess = 

        # Create dictionary with constant parameter-values
        self.params = self.all_params.copy()
        self.params.update(constant_params)
        

        for key_i in list(self.params.keys()):
            if key_i.startswith('log_'):
                self.params = self.log_to_linear(self.params, key_i)

        # Check the used PT profile
        self.PT_mode = PT_mode
        assert(self.PT_mode in ['free', 'free_gradient', 'grid', 'Molliere', 'RCE'])
        self.PT_adiabatic = PT_adiabatic
        
        self.n_T_knots = n_T_knots
        self.enforce_PT_corr = enforce_PT_corr

        # Check the used chemistry type
        self.chem_mode = chem_mode
        assert(self.chem_mode in ['eqchem', 'free', 'fastchem', 'SONORAchem'])
        
        # Check the used cloud type
        self.cloud_mode = cloud_mode
        if cloud_mode == 'grey':
            self.cloud_mode = 'gray'
        assert(self.cloud_mode in ['gray', 'MgSiO3', None])

        # Check the used covariance definition
        self.cov_mode = cov_mode
        assert(self.cov_mode in ['GP', None])

        self.wlen_settings = wlen_settings
        assert isinstance(self.wlen_settings, dict), 'wlen_settings must be a dictionary'
        # check that the keys are in ['K2166', 'NIRSpec']
        assert all([key in ['K2166', 'NIRSpec'] for key in self.wlen_settings.keys()]), 'wlen_settings keys must be in [K2166, NIRSpec]'
        
        # check distance / parallax parameters
        for p in ['parallax', 'parallax_mas']:
            if p in self.params.keys():
                self.params['d_pc'] = 1 / (self.params[p] * 1e-3)
            
            
        
    def __str__(self):
        out = '** Parameters **\n'
        # add line of dashes
        out += '-'*len(out) + '\n'
        for key_i in self.__dict__.keys():
            if key_i.startswith('__'):
                continue
            out += f'{key_i}: {self.__dict__[key_i]}\n'
        return out
    def __repr__(self):
        return self.__str__()

    def __call__(self, cube, ndim=None, nparams=None):
        '''
        Update values in the params dictionary.

        Input
        -----
        cube : np.ndarray
            Array of values between 0-1 for the free parameters.
        ndim : int or None
            Number of dimensions. (Equal to number of free parameters)
        nparams : int or None
            Number of free parameters.
        '''
        # new_cube = np.array(cube) # copy the cube to avoid changing the original
        # Convert to numpy array if necessary
        if (ndim is None) and (nparams is None):
            self.cube_copy = cube
        else:
            self.cube_copy = np.array(cube[:ndim])

        # Loop over all parameters
        for i, key_i in enumerate(self.param_keys):

            # Sample within the boundaries
            low, high = self.param_priors[key_i]
            
            cond = (self.PT_mode == 'RCE') and (key_i.startswith('dlnT_dlnP_')) and (key_i != 'dlnT_dlnP_RCE')
            cond = cond and (self.PT_adiabatic) # default is True
            
            if cond:
                high = min(self.params['dlnT_dlnP_RCE'], high)
                low =  min(self.params['dlnT_dlnP_RCE'], low)
                
            if key_i == 'R_out':
                low = max(self.params['R_cav'], low)
                
            # print(f' [Parameters.__call__]: key_i = {key_i}, low = {low}, high = {high}')
            cube[i] = low + (high-low)*cube[i]
            # new_cube[i] = low + (high-low)*cube[i]

            self.params[key_i] = cube[i]

            if key_i.startswith('log_'):
                # print(f' [Parameters.__call__]: log_to_linear for {key_i}')
                self.params = self.log_to_linear(self.params, key_i)
                                
            

        # Read the parameters for the model's segments
        # cube = self.read_PT_params(cube)
        # self.read_uncertainty_params()
        # self.read_chemistry_params()
        # self.read_cloud_params()
        # self.read_resolution_params() # new 2024-05-27: read resolution parameters of each grating
        # self.read_disk_params()
        self.read_params()

        if (ndim is None) and (nparams is None):
            return cube
        else:
            return
    
    def read_PT_params(self, cube=None):

        if self.params.get('temperature') is not None:
            return cube

        if self.PT_mode == 'grid':
            return cube

        if (self.PT_mode == 'Molliere') and (cube is not None):

            # Update the parameters in the MultiNest cube 
            # for the Molliere et al. (2020) PT parameterization

            # T_0 is connection temperature at P=0.1 bar
            T_0 = (3/4 * self.params['T_int']**4 * (0.1 + 2/3))**(1/4)

            # Define the prior based on the other knots
            low, high = self.param_priors['T_1']
            idx = np.argwhere(self.param_keys=='T_1').flatten()[0]
            self.params['T_1'] = T_0 * (high - (high-low)*self.cube_copy[idx])
            cube[idx] = self.params['T_1']

            low, high = self.param_priors['T_2']
            idx = np.argwhere(self.param_keys=='T_2').flatten()[0]
            self.params['T_2'] = self.params['T_1'] * (high - (high-low)*self.cube_copy[idx])
            cube[idx] = self.params['T_2']

            low, high = self.param_priors['T_3']
            idx = np.argwhere(self.param_keys=='T_3').flatten()[0]
            self.params['T_3'] = self.params['T_2'] * (high - (high-low)*self.cube[idx])
            cube[idx] = self.params['T_3']

        if self.PT_mode in ['free', 'free_gradient', 'Molliere']:

            # Fill the pressure knots
            self.params['log_P_knots'] = np.array(self.params['log_P_knots'])
            for i in range(self.n_T_knots-1):
                
                if f'd_log_P_{i}{i+1}' in list(self.params.keys()):
                    # Add the difference in log P to the previous knot
                    self.params['log_P_knots'][-(i+1)-1] = \
                        self.params['log_P_knots'][::-1][i] - \
                        self.params[f'd_log_P_{i}{i+1}']
            
            self.params['P_knots']     = 10**self.params['log_P_knots']
            self.params['ln_P_knots']  = np.log(self.params['P_knots'])

        if self.PT_mode in ['free', 'Molliere']:
            
            # Combine the upper temperature knots into an array
            self.params['T_knots'] = []
            for i in range(self.n_T_knots-1):

                T_i = self.params[f'T_{i+1}']

                if (cube is not None) and self.enforce_PT_corr:
                    # Temperature knot is product of previous knots
                    T_i = np.prod([self.params[f'T_{j}'] for j in range(i+2)])
                    
                    idx = np.argwhere(self.param_keys==f'T_{i+1}').flatten()[0]
                    cube[idx] = T_i

                self.params['T_knots'].append(T_i)

            self.params['T_knots'] = np.array(self.params['T_knots'])[::-1]
            
            if 'invgamma_gamma' in self.param_keys:
                self.params['gamma'] = self.params['invgamma_gamma']

        if (self.PT_mode == 'free_gradient'):

            # Combine the upper temperature knots into an array            
            self.params['T_knots'] = [self.params['T_0'], ]
            self.params['dlnT_dlnP_knots'] = np.array([
                self.params[f'dlnT_dlnP_{i}'] for i in range(self.n_T_knots)
                ])
            for i in range(self.n_T_knots-1):

                ln_P_i1 = self.params['ln_P_knots'][::-1][i+1]
                ln_P_i  = self.params['ln_P_knots'][::-1][i]

                T_i1 = np.exp(
                    np.log(self.params['T_knots'][-1]) + \
                    (ln_P_i1 - ln_P_i) * self.params[f'dlnT_dlnP_{i}']
                    )
                self.params['T_knots'].append(T_i1)

            self.params['T_knots'] = np.array(self.params['T_knots'])[::-1]
            self.params['dlnT_dlnP_knots'] = self.params['dlnT_dlnP_knots'][::-1]
            
        if (self.PT_mode == 'RCE'):
            
            self.PT_adiabatic = self.params.get('PT_adiabatic', True)
            dlnT_dlnP_keys = [key for key in self.param_keys if key.startswith('dlnT_dlnP_') and key != 'dlnT_dlnP_RCE']
            # print(f' [Parameters.read_PT_params]: dlnT_dlnP_keys = {dlnT_dlnP_keys}')
            self.params['dlnT_dlnP_knots'] = np.array([self.params[key] for key in dlnT_dlnP_keys])
            # pop index containing key 'dlnT_dlnP_RCE' and insert back in the middle
            self.params['dlnT_dlnP_knots'] = np.insert(self.params['dlnT_dlnP_knots'], len(dlnT_dlnP_keys)//2, self.params['dlnT_dlnP_RCE'])
            
        return cube

    def read_uncertainty_params(self):
        
        # cov_keys = ['beta', 'a', 'l', 'a_f', 'l_f']
        cov_keys = ['a', 'l']
        for w_set, (n_orders, n_dets) in self.wlen_settings.items():
           
            
            assert ('l' in self.params.keys()) or (f'l_{w_set}' in self.params.keys()), ' [Parameters.read_uncertainty_params]: l parameter not found in the parameter keys'
            # self.params[f'l_{w_set}'] = np.ones((n_orders, n_dets)) * self.params['l']
            if f'l_{w_set}' in self.params.keys():
                self.params[f'l_{w_set}'] = self.params[f'l_{w_set}'] * np.ones((n_orders, n_dets))
            else:
                self.params[f'l_{w_set}'] = np.ones((n_orders, n_dets)) * self.params['l']
                                                                                        
            if w_set == 'NIRSpec':
                a_list = [k for k in list(self.params.keys()) if k.startswith('a_') and 'NIRSpec' not in k]
                beta_list = [k for k in list(self.params.keys()) if k.startswith('beta_') and 'NIRSpec' not in k]
                # print(f' [Parameters.read_uncertainty_params]: a_list = {a_list} for {w_set}')
                # print(f' [Parameters.read_uncertainty_params]: beta_list = {beta_list} for {w_set}')
                if len(a_list) == 0:
                    self.params[f'a_{w_set}'] = np.zeros((n_orders, n_dets))
                if len(a_list) == 1:
                    self.params[f'a_{w_set}'] = self.params[a_list[0]] * np.ones((n_orders, n_dets))
                if len(a_list) > 1: 
                    # set a different parameter for each grating (both filters of same grating have same a and beta)
                    self.params[f'a_{w_set}'] = np.array([self.params[k] for k in a_list for _ in range(len(a_list))]).reshape((n_orders, n_dets))
                # print(f' [Parameters.read_uncertainty_params]: a_{w_set} = {self.params[f"a_{w_set}"]}')
                
                if len(beta_list) == 0:
                    self.params[f'beta_{w_set}'] = np.ones((n_orders, n_dets))
                if len(beta_list) == 1:
                    self.params[f'beta_{w_set}'] = self.params[beta_list[0]] * np.ones((n_orders, n_dets))
                if len(beta_list) > 1:
                    # set a different parameter for each grating (both filters of same grating have same a and beta)
                    self.params[f'beta_{w_set}'] = np.array([self.params[k] for k in beta_list for _ in range(len(beta_list))]).reshape((n_orders, n_dets))
            else:
                
                if len(a_list) == 0:
                    self.params[f'a_{w_set}'] = np.zeros((n_orders, n_dets))
                if len(a_list) == 1:
                    self.params[f'a_{w_set}'] = self.params[a_list[0]] * np.ones((n_orders, n_dets))
                if len(a_list) > 1:
                    # warning, not implemented
                    self.params[f'a_{w_set}'] = self.params[a_list[0]] * np.ones((n_orders, n_dets))
                    print(f' [Parameters.read_uncertainty_params]: Warning, several GP amplitudes not implemented for {w_set}')

            # print(f' [Parameters.read_uncertainty_params]: a_{w_set} = {self.params[f"a_{w_set}"]}')
            # print(f' [Parameters.read_uncertainty_params]: l_{w_set} = {self.params[f"l_{w_set}"]}')
            #         self.params['a'] = self.params[a_list[0]]
                
            #     self.params[f'a_{w_set}'] = self.params[f'{key_i}']
                
            # for key_i in cov_keys:
            #     key_i_list = [k for k in self.param_keys if k.startswith(f'{key_i}_')]
                
                
            #     # Make specific for each wlen setting
            #     if f'{key_i}_{w_set}' not in self.param_keys:
            #         self.params[f'{key_i}_{w_set}'] = self.params[f'{key_i}']

            #     # Reshape to values for each order and detector
            #     self.params[f'{key_i}_{w_set}'] = \
            #         np.ones((n_orders, n_dets)) * self.params[f'{key_i}_{w_set}']
                
            #     # Loop over the orders
            #     for i in range(n_orders):

            #         # Replace the constant with a free parameter
            #         if f'{key_i}_{i+1}' in self.param_keys:
            #             self.params[f'{key_i}_{w_set}'][i,:] = \
            #                 self.params[f'{key_i}_{i+1}']
                        
            #         if f'{key_i}_{w_set}_{i+1}' in self.param_keys:
            #             self.params[f'{key_i}_{w_set}'][i,:] = \
            #                 self.params[f'{key_i}_{w_set}_{i+1}']


    def read_chemistry_params(self):

        if self.chem_mode in ['eqchem', 'fastchem', 'SONORAchem']:
            # Use chemical equilibrium
            self.VMR_species = None

        # elif self.chem_mode == 'free': # deprecated?
        #     # print(f'[Parameters.read_chemistry_params]: Using free chemistry')
        #     # print(f'[Parameters.read_chemistry_params]: self.Param.param_keys = {self.param_keys}')
        #     # Use free chemistry
        #     self.params['C/O'], self.params['Fe/H'] = None, None

        #     # Loop over all possible species
        #     print(f' Chemistry.species_info.keys() = {Chemistry.species_info.keys()}')
        #     self.VMR_species = {}
        #     for species_i in Chemistry.species_info.keys():
        #     # for species_i in Chemistry.species_info.name.tolist():
            
        #         # print(f'[Parameters.read_chemistry_params]: {species_i}')
        #         # If multiple VMRs are given
        #         # for j in range(3):
        #         #     if f'log_{species_i}_{j}' in self.param_keys:
        #         #         self.VMR_species[f'{species_i}_{j}'] = self.params[f'{species_i}_{j}']

        #         if f'log_{species_i}' in list(self.params.keys()):
        #             self.VMR_species[f'{species_i}'] = self.params[f'{species_i}']
        #             continue

        #         if species_i == '13CO' and ('log_13C/12C_ratio' in self.param_keys):
        #             # Use isotope ratio to retrieve the VMR
        #             self.VMR_species[species_i] = self.params['13C/12C_ratio'] * self.params['12CO']

        #         if species_i == '13CH4' and ('log_13C/12C_ratio' in self.param_keys):
        #             # Use isotope ratio to retrieve the VMR
        #             self.VMR_species[species_i] = self.params['13C/12C_ratio'] * self.params['CH4']

        #         if species_i == 'C18O' and ('log_18O/16O_ratio' in self.param_keys):
        #             self.VMR_species[species_i] = self.params['18O/16O_ratio'] * self.params['12CO']
        #         if species_i == 'C17O' and ('log_17O/16O_ratio' in self.param_keys):
        #             self.VMR_species[species_i] = self.params['17O/16O_ratio'] * self.params['12CO']

        #         if species_i == 'H2O_181' and ('log_18O/16O_ratio' in self.param_keys):
        #             self.VMR_species[species_i] = self.params['18O/16O_ratio'] * self.params['H2O']
        #         if species_i == 'H2O_171' and ('log_17O/16O_ratio' in self.param_keys):
        #             self.VMR_species[species_i] = self.params['17O/16O_ratio'] * self.params['H2O']

    def read_resolution_params(self):
         # check for resolution parameters and place them in a list `res`
        res_keys = [key for key in self.params.keys() if key.startswith('res_')]
        if len(res_keys) > 0:
            self.params['res'] = [self.params[key] for key in res_keys]
            
        self.params['gratings'] = self.params.get('gratings', ['g235h', 'g235h', 'g395h', 'g395h'])
            
            
    @classmethod
    def log_to_linear(cls, param_dict, key_log, key_lin=None, verbose=False):

        if not isinstance(key_log, (list, tuple, np.ndarray)):
            key_log = [key_log]
            
        if not isinstance(key_lin, (list, tuple, np.ndarray)):
            key_lin = [key_lin]
        
        for key_log_i, key_lin_i in zip(key_log, key_lin):

            if key_lin_i is None:
                key_lin_i = key_log_i.replace('log_', '')

            # Value of the logarithmic parameter
            val_log = param_dict[key_log_i]

            if isinstance(val_log, (float, int, np.ndarray)):
                # Convert from log to linear if float or integer
                param_dict[key_lin_i] = 10**val_log
            elif isinstance(val_log, list):
                param_dict[key_lin_i] = 10**np.array(val_log)

            elif val_log is None:
                # Set linear parameter to None as well
                param_dict[key_lin_i] = None

        return param_dict
    
    def lower_prior_sample(self):
        '''Get the sample from the lower prior boundaries.'''
        cube = np.zeros(self.n_params)
        for i, key_i in enumerate(self.param_keys):
            low, high = self.param_priors[key_i]
            cube[i] = low
        return cube
    
    def upper_prior_sample(self):
        '''Get the sample from the upper prior boundaries.'''
        cube = np.zeros(self.n_params)
        for i, key_i in enumerate(self.param_keys):
            low, high = self.param_priors[key_i]
            cube[i] = high
        return cube
    
    # @property
    # def random_sample(self):
    #     '''Get a random sample from the prior.'''
    #     cube = np.random.rand(self.n_params)
    #     return self.__call__(cube)
        
        
    def read_disk_params(self):
        ''' Read the disk parameters '''
        
        if 'T_ex' in self.params.keys():
            disk_default = {
                'T_ex': [600.0],
                'N_mol': [1e17],
                'A_au': [1.0],
                'dV': [1.0],
            }
            all_keys = list(self.params.keys())
            for k in disk_default.keys():
                
                v_list = [self.params[key] for key in all_keys if key.startswith(f'{k}_')]
                if len(v_list) == 0:
                    v_list = disk_default[k]

                self.params[k] = np.array([np.array(v_list)])
                # print(f' [Parameters.read_disk_params]: k = {k}, self.params[k] = {self.params[k]}')
        
            assert 'd_pc' in self.params.keys(), ' [Parameters.read_disk_params]: d_pc not found in the parameter keys'
            
        if 'R_cav' in self.param_keys:
            self.params['R_cav'] = self.params['R_cav']
            self.params['R_out'] = self.params.get('R_out', self.params['R_cav'] * 100.0)
            # self.params['T_star']
            assert 'T_star' in self.params.keys(), ' [Parameters.read_disk_params]: T_star not found in the parameter keys'
            self.params['i'] = np.radians(self.params.get('i_deg', 45.0))
            
            assert 'd_pc' in self.params.keys(), ' [Parameters.read_disk_params]: d_pc not found in the parameter keys'
            self.params['q'] = self.params.get('q', 0.75)
            
        if 'log_A_au_12CO' in self.param_keys:
            # self.params['A_au_12CO'] = 10**self.params['log_A_au_12CO']
            # self.params['A_au_13CO'] = 10**self.params['log_A_au_13CO']
            # self.params['A_au_H2O'] = 10**self.params['log_A_au_H2O']
            
            assert 'd_pc' in self.params.keys(), ' [Parameters.read_disk_params]: d_pc not found in the parameter keys'
        # if 'R_d' in self.param_keys:
        #     rjup_cm = 7.1492e9
        #     au_cm = 1.496e13
        #     self.params['A_au'] = np.pi * (self.params['R_d'] * (rjup_cm/au_cm))**2
        
            
        
            
    def read_params(self):
        '''Parse the parameters from the parameter dictionary.'''
        self.read_PT_params()
        self.read_uncertainty_params()
        self.read_chemistry_params()
        # self.read_cloud_params()
        self.read_resolution_params() # new 2024-05-27: read resolution parameters of each grating
        self.read_disk_params()
        return self