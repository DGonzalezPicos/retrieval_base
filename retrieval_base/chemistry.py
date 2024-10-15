import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline, RegularGridInterpolator
import petitRADTRANS.nat_cst as nc

from retrieval_base.auxiliary_functions import quantiles, get_path
import os
import h5py
path = get_path()

def get_Chemistry_class(line_species, pressure, mode, **kwargs):

    if mode == 'free':
        return FreeChemistry(line_species, pressure, **kwargs)
    if mode == 'eqchem':
        return EqChemistry(line_species, pressure, **kwargs)
    if mode == 'fastchem':
        return FastChemistry(line_species, pressure, **kwargs)
    if mode == 'SONORAchem':
        return SONORAChemistry(line_species, pressure, **kwargs)
    if mode == 'SPHINX':
        return SPHINXChemistry(line_species, pressure, **kwargs)
    
    

class Chemistry:

    # Dictionary with info per molecular/atomic species
    species_info_default_file = f'{path}data/species_info.csv'    
    
    # Neglect certain species to find respective contribution
    neglect_species = {
        '12CO': False, 
        '13CO': False, 
        'C18O': False, 
        'C17O': False, 
        
        'H2O': False, 
        'H2O_181': False, 
        'HDO': False, 

        'CH4': False, 
        '13CH4': False, 
        
        'NH3': False, 
        'HCN': False, 
        'H2S': False, 
        'FeH': False, 
        'CrH': False, 
        'NaH': False, 

        'TiO': False, 
        'VO': False, 
        'AlO': False, 
        'CO2': False, 

        'HF': False, 
        'HCl': False, 

        #'H2': False, 
        'HD': False, 

        'K': False, 
        'Na': False, 
        'Ti': False, 
        'Fe': False, 
        'Ca': False, 
        'Al': False, 
        'Mg': False, 
        #'He': False, 
        'CN': False,
        # ions
        'CaII': False,
        'FeII': False,
        }

    def __init__(self, line_species, pressure):

        
        self.line_species = line_species

        self.pressure     = pressure
        self.n_atm_layers = len(self.pressure)

        # Set to None initially, changed during evaluation
        self.mass_fractions_envelopes = None
        self.mass_fractions_posterior = None
        self.unquenched_mass_fractions_posterior = None
        self.unquenched_mass_fractions_envelopes = None


    def set_species_info(self, line_species_dict=None, file=None):
    
        if file is not None:
            self.species_info = pd.read_csv(file)
            self.species_info['label'] = self.species_info['mathtext_name']
            
        assert hasattr(self, 'species_info'), 'species_info not yet loaded'
        if line_species_dict is not None:
            line_species_dict_default = dict(zip(self.species_info['name'].tolist(), self.species_info['pRT_name'].tolist()))
            line_species_dict_new = line_species_dict_default.copy()
            line_species_dict_new.update(line_species_dict)
            
            # update pRT_name with line_species_dict
            self.species_info['pRT_name'] = self.species_info['name'].map(line_species_dict_new)
        
        
        self.pRT_name_dict = {v['pRT_name']: v['name'] for i, v in self.species_info.iterrows()}
        self.pRT_name_dict_r = {v['name']: v['pRT_name'] for i, v in self.species_info.iterrows()}
        
        return self.species_info
    
    def remove_species(self):

        # Remove the contribution of the specified species
        for species_i, remove in self.neglect_species.items():
            
            if not remove:
                continue

            # Read the name of the pRT line species
            line_species_i = self.read_species_info(species_i, 'pRT_name')

            # Set abundance to 0 to evaluate species' contribution
            if line_species_i in self.line_species:
                self.mass_fractions[line_species_i] *= 0

    # @classmethod
    # def read_species_info(cls, species, info_key):
        
    #     if info_key == 'pRT_name':
    #         return cls.species_info[species][0]
    #     if info_key == 'pyfc_name':
    #         return cls.species_info[species][1]
        
    #     if info_key == 'mass':
    #         return cls.species_info[species][2]
        
    #     if info_key == 'COH':
    #         return cls.species_info[species][3]
    #     if info_key == 'C':
    #         return cls.species_info[species][3][0]
    #     if info_key == 'O':
    #         return cls.species_info[species][3][1]
    #     if info_key == 'H':
    #         return cls.species_info[species][3][2]

    #     if info_key == 'c' or info_key == 'color':
    #         return cls.species_plot_info[species][0]
    #     if info_key == 'label':
    #         return cls.species_plot_info[species][1]
    # @classmethod
    def read_species_info(self, species, info_key):
        assert species in self.species_info['name'].values, f'species = {species} not in species_info'
        assert info_key in self.species_info.columns, f'info_key = {info_key} not in species_info.columns'
            
        return self.species_info.loc[self.species_info['name'] == species, info_key].values[0]
        
    def get_VMRs_posterior(self, save_to=None):
        
        assert hasattr(self, 'mass_fractions_posterior')
        self.VMRs_posterior = {}
        self.VMRs_envelopes = {}
        # info = self.species_info
        MMW = self.mass_fractions_posterior['MMW'].mean() if hasattr(self, 'mass_fractions_posterior') else self.mass_fractions['MMW']
        print(f'[Chemistry.get_VMRs_posterior] Calculating VMRs posterior and envelopes for {self.line_species}')
        for line_species_i in self.line_species:
            # key_i = [key_i for key_i in info.keys() if info[key_i][0]==line_species_i][0]
            key_i = self.pRT_name_dict.get(line_species_i, None)
            # key_i = 
            # check key_i is not empty
            # print(f' {line_species_i} ({key_i}) -> {info[key_i][2]}')
            if len(key_i) == 0:
                continue
            # mu = info[key_i][2] # atomic mass
            mu = self.read_species_info(key_i, 'mass')
            # print(f' mu_{key_i} = {mu}')
            # free-chemistry = constant VMR
            # WARNING: equilibrium chemistry should use the mean value or something else
            vmr_i = self.mass_fractions_posterior[line_species_i] * (MMW/ mu)
            self.VMRs_posterior[key_i] = vmr_i[:,0]
            # print(f' vmr_i.shape = {vmr_i.shape}')
            self.VMRs_envelopes[key_i] = quantiles(vmr_i, q=[0.16, 0.5, 0.84], axis=0)
            
        if "13CO" in list(self.VMRs_posterior.keys()) and "12CO" in list(self.VMRs_posterior.keys()):
            self.VMRs_posterior["12_13CO"] = self.VMRs_posterior["12CO"] / self.VMRs_posterior["13CO"]
        if "C18O" in list(self.VMRs_posterior.keys()) and "12CO" in list(self.VMRs_posterior.keys()):
            # check it is detected i.e. uncertainty of C18O is smaller than 1.0
            q16, q50, q84 = quantiles(self.VMRs_posterior["C18O"], q=[0.16, 0.5, 0.84])
            if abs(q84 - q16) < 1.0 and abs(q50 - q16) < 1.0 and abs(q84 - q50) < 1.0:
                self.VMRs_posterior["C16_18O"] = self.VMRs_posterior["12CO"] / self.VMRs_posterior["C18O"]
        
        if "H2O_181" in list(self.VMRs_posterior.keys()) and "H2O" in list(self.VMRs_posterior.keys()):
            self.VMRs_posterior["H2_16_18O"] = self.VMRs_posterior["H2O"] / self.VMRs_posterior["H2O_181"]
            
        if hasattr(self, 'CO_posterior'):
            self.VMRs_posterior["C/O"] = self.CO_posterior
        if hasattr(self, 'FeH_posterior'):
            self.VMRs_posterior["Fe/H"] = self.FeH_posterior
        del self.mass_fractions_posterior
        
        if save_to is not None:
            # save the VMRs to a file
            file_posterior = save_to + 'posterior.npy'
            file_envelopes = save_to + 'envelopes.npy'
            file_labels    = save_to + 'labels.npy'
            np.save(file_posterior, np.array(list(self.VMRs_posterior.values())))
            np.save(file_envelopes, np.array(list(self.VMRs_envelopes.values())))
            np.save(file_labels, np.array(list(self.VMRs_posterior.keys())))
            print(f'[Chemistry.get_VMRs_posterior] Saved VMRs posterior and envelopes to {file_posterior}, {file_envelopes}, {file_labels}')
        return self

class FreeChemistry(Chemistry):

    def __init__(self, line_species, pressure, spline_order=0, **kwargs):

        # Give arguments to the parent class
        super().__init__(line_species, pressure)

        self.spline_order = spline_order

    def __call__(self, VMRs, params):

        self.VMRs = VMRs

        # Total VMR without H2, starting with He
        VMR_He = 0.15
        VMR_wo_H2 = 0 + VMR_He

        # Create a dictionary for all used species
        self.mass_fractions = {}

        C, O, H = 0, 0, 0

        for species_i in self.species_info.keys():
            line_species_i = self.read_species_info(species_i, 'pRT_name')
            mass_i = self.read_species_info(species_i, 'mass')
            COH_i  = self.read_species_info(species_i, 'COH')

            if species_i in ['H2', 'He']:
                continue

            if line_species_i in self.line_species:
                
                if self.VMRs.get(species_i) is not None:
                    # Single value given: constant, vertical profile
                    VMR_i = self.VMRs[species_i] * np.ones(self.n_atm_layers)
                    
                    iso_profiles = {'13CO': '12CO',
                                    'C18O': '12CO',
                                    'C17O': '12CO',
                                    'H2O_181': 'H2O'}
                    # iso_profiles = {}
                    if len(iso_profiles) > 0:
                        for k, v in iso_profiles.items():
                            if species_i == k and self.VMRs.get(v) is not None:
                    
                                VMR_i = (self.VMRs[v] / self.VMRs[v].max()) * self.VMRs[k]
                                # check that is is smaller than 1.0
                                assert (VMR_i <= 1.0).all(), f'VMR_{v} = {VMR_i} > 1.0'         
                       

                if self.VMRs.get(f'{species_i}_0') is not None:
                    # Multiple values given, use spline interpolation

                    # Define the spline knots in pressure-space
                    if params.get(f'log_P_{species_i}') is not None:
                        log_P_knots = np.array([
                            np.log10(self.pressure).min(), params[f'log_P_{species_i}'], 
                            np.log10(self.pressure).max()
                            ])
                    else:
                        log_P_knots = np.linspace(
                            np.log10(self.pressure).min(), 
                            np.log10(self.pressure).max(), num=3
                            )
                    
                    # Define the abundances at the knots
                    # VMR_knots = np.array([self.VMRs[f'{species_i}_{j}'] for j in range(3)])[::-1]
                    # passing keys from bottom to top (0, 1, 2) and inverting here to match the pressure (top to bottom)
                    # VMR_knots = np.array([self.VMRs.get(f'{species_i}_{j}', None) for j in range(3)])[::-1]
                    VMR_knots = [self.VMRs.get(f'{species_i}_{j}', None) for j in range(3)]
                    if VMR_knots[-1] is None: # corresponding to log_VMR_2 in the free_params dictionary
                        VMR_knots[-1] = VMR_knots[1] # constant VMR at the top
                    
                    # linear interpolation
                    VMR_i = 10.0**(np.interp(np.log10(self.pressure), log_P_knots, np.log10(np.array(VMR_knots)[::-1])))
                     

                self.VMRs[species_i] = VMR_i

                # Convert VMR to mass fraction using molecular mass number
                self.mass_fractions[line_species_i] = mass_i * VMR_i
                VMR_wo_H2 += VMR_i

                # Record C, O, and H bearing species for C/O and metallicity
                C += COH_i[0] * VMR_i
                O += COH_i[1] * VMR_i
                H += COH_i[2] * VMR_i

        # Add the H2 and He abundances
        self.mass_fractions['He'] = self.read_species_info('He', 'mass') * VMR_He
        # if 'H2' in self.VMRs.keys():
        #     self.mass_fractions['H2'] = self.read_species_info('H2', 'mass') * 
            
        self.mass_fractions['H2'] = self.read_species_info('H2', 'mass') * (1 - VMR_wo_H2)
        # assert self.mass_fractions['H2'].all() < 1, 'H2 mass fraction > 1'
        # assert self.mass_fractions['He'].all() < 1, 'He mass fraction > 1'
    
        # self.mass_fractions['H-'] = 6e-9 # solar
        self.mass_fractions['H-'] = self.VMRs.get('H-', 6e-9)
        # self.mass_fractions['e-'] = 1e-10# solar
        self.mass_fractions['e-'] = 1e-10 * (self.mass_fractions['H-'] / 6e-9)
        
    
        # Add to the H-bearing species
        H += self.read_species_info('H2', 'H') * (1 - VMR_wo_H2)
        self.mass_fractions['H'] = H

        if VMR_wo_H2.any() > 1: #or (self.mass_fractions['H2'] > 1).any():
            # Other species are too abundant
            self.mass_fractions = -np.inf
            print(f' VMR_wo_H2 = {VMR_wo_H2} > 1 --> mass_fractions = -np.inf')
            return self.mass_fractions

        # Compute the mean molecular weight from all species
        MMW = 0
        for mass_i in self.mass_fractions.values():
            MMW += mass_i
        MMW *= np.ones(self.n_atm_layers)

        # Turn the molecular masses into mass fractions
        for line_species_i in self.mass_fractions.keys():
            self.mass_fractions[line_species_i] /= MMW

        # pRT requires MMW in mass fractions dictionary
        self.mass_fractions['MMW'] = MMW

        # Compute the C/O ratio and metallicity
        self.CO = C/O

        log_CH_solar = 8.43 - 12 # Asplund et al. (2009)
        # log_CH_solar = 8.46 - 12 # Asplund et al. (2021)
        self.FeH = np.log10(C/H) - log_CH_solar
        self.CH  = self.FeH

        self.CO = np.mean(self.CO)
        self.FeH = np.mean(self.FeH)
        self.CH = np.mean(self.CH)

        # Remove certain species
        self.remove_species()

        return self.mass_fractions
    
    def get_VMRs_envelopes(self):
        
        assert hasattr(self, 'mass_fractions_envelopes'), 'Mass fractions not yet evaluated.'
        
        self.VMRs_envelopes = {}
        # MMW = self.mass_fractions_envelopes['MMW'][3].mean()
        # for line_species_i, key in zip(self.line_species, self.VMRs.keys()):
        for key in self.mass_fractions_envelopes.keys():
            
            # get atomic mass
            # print(f'key = {key}')
            # mu_i = self.read_species_info(key, 'mass')
            # print(f'mu_{key} = {mu_i}')
            # mu_i = 1.
            # self.VMRs_envelopes[key] = self.mass_fractions_envelopes[line_species_i] * (MMW / mu_i)
            self.VMRs_envelopes[key] = quantiles(self.VMRs_posterior[key])
        # return self.VMRs_envelopes
        return self
    
    
class EqChemistry(Chemistry):

    def __init__(self, line_species, pressure, quench_setup={}, **kwargs):

        # Give arguments to the parent class
        super().__init__(line_species, pressure)

        # Retrieve the mass ratios of the isotopologues
        self.mass_ratio_13CO_12CO = self.read_species_info('13CO', 'mass') / \
                                    self.read_species_info('12CO', 'mass')
        self.mass_ratio_C18O_12CO = self.read_species_info('C18O', 'mass') / \
                                    self.read_species_info('12CO', 'mass')
        self.mass_ratio_C17O_12CO = self.read_species_info('C17O', 'mass') / \
                                    self.read_species_info('12CO', 'mass')
        
        # Load the interpolation function
        import petitRADTRANS.poor_mans_nonequ_chem as pm
        self.pm_interpol_abundances = pm.interpol_abundances

        # Species to quench per quench pressure
        self.quench_setup = quench_setup
        
    def get_pRT_mass_fractions(self, params):

        # Retrieve the mass fractions from the chem-eq table
        pm_mass_fractions = self.pm_interpol_abundances(
            self.CO*np.ones(self.n_atm_layers), 
            self.FeH*np.ones(self.n_atm_layers), 
            self.temperature, 
            self.pressure
            )
        
        # Fill in the dictionary with the right keys
        self.mass_fractions = {
            'MMW': pm_mass_fractions['MMW']
            }
        
        for line_species_i in self.line_species:

            if line_species_i in ['CO_main_iso', 'CO_high']:
                # 12CO mass fraction
                self.mass_fractions[line_species_i] = \
                    (1 - self.C13_12_ratio * self.mass_ratio_13CO_12CO - \
                     self.O18_16_ratio * self.mass_ratio_C18O_12CO - \
                     self.O17_16_ratio * self.mass_ratio_C17O_12CO
                    ) * pm_mass_fractions['CO']
                continue
                
            if line_species_i in ['CO_36', 'CO_36_high']:
                # 13CO mass fraction
                self.mass_fractions[line_species_i] = \
                    self.C13_12_ratio * self.mass_ratio_13CO_12CO * \
                    pm_mass_fractions['CO']
                continue
            
            if line_species_i in ['CO_28', 'CO_28_high']:
                # C18O mass fraction
                self.mass_fractions[line_species_i] = \
                    self.O18_16_ratio * self.mass_ratio_C18O_12CO * \
                    pm_mass_fractions['CO']
                continue
            
            if line_species_i in ['CO_27', 'CO_27_high']:
                # C17O mass fraction
                self.mass_fractions[line_species_i] = \
                    self.O17_16_ratio * self.mass_ratio_C17O_12CO * \
                    pm_mass_fractions['CO']
                continue

            # All other species    
            species_i = line_species_i.split('_')[0]
            self.mass_fractions[line_species_i] = pm_mass_fractions.get(species_i)
        
        # Add the H2 and He abundances
        self.mass_fractions['H2'] = pm_mass_fractions['H2']
        self.mass_fractions['He'] = pm_mass_fractions['He']
        self.mass_fractions['H'] = pm_mass_fractions['H']
        # Add H- and e-
        self.mass_fractions['H-'] = pm_mass_fractions['H-']
        self.mass_fractions['e-'] = pm_mass_fractions['e-']
        # For solar C/O and Fe/H:
        # self.mass_fractions['H-'] = 6e-9 # solar
        # self.mass_fractions['e-'] = 1e-10 # solar
        
        # print(f'mass_fractions[e-] = {np.mean(self.mass_fractions["e-"]):.2e}')
        # print(f'mass_fractions[H-] = {np.mean(self.mass_fractions["H-"]):.2e}')

        # Convert the free-chemistry VMRs to mass fractions
        for species_i in self.species_info.keys():
            line_species_i = self.read_species_info(species_i, 'pRT_name')
            mass_i = self.read_species_info(species_i, 'mass')

            if line_species_i not in self.line_species:
                continue
            if self.mass_fractions.get(line_species_i) is not None:
                continue

            # Confirm that free-chemistry VMR is defined
            assert(params.get(f'log_{species_i}') is not None)

            VMR_i = 10**params.get(f'log_{species_i}')
            self.mass_fractions[line_species_i] = VMR_i * mass_i / self.mass_fractions['MMW']
            print(f'{line_species_i} = {np.mean(VMR_i):.2e}')

    def quench_chemistry(self, quench_key='P_quench'):

        # Layers to be replaced by a constant abundance
        mask_quenched = (self.pressure < self.P_quench[quench_key])

        for species_i in self.quench_setup[quench_key]:

            if self.species_info.get(species_i) is None:
                continue

            line_species_i = self.read_species_info(species_i, 'pRT_name')
            if not line_species_i in self.line_species:
                continue

            # Store the unquenched abundance profiles
            mass_fraction_i = self.mass_fractions[line_species_i]
            #self.unquenched_mass_fractions[line_species_i] = np.copy(mass_fraction_i)
            
            # Own implementation of quenching, using interpolation
            mass_fraction_i[mask_quenched] = np.interp(
                np.log10(self.P_quench[quench_key]), 
                xp=np.log10(self.pressure), fp=mass_fraction_i
                )
            self.mass_fractions[line_species_i] = mass_fraction_i

    def __call__(self, params, temperature):

        # Update the parameters
        self.CO  = params.get('C/O')
        self.FeH = params.get('Fe/H')

        self.C13_12_ratio = params.get('C13_12_ratio')
        self.O18_16_ratio = params.get('O18_16_ratio')
        self.O17_16_ratio = params.get('O17_16_ratio')

        if self.C13_12_ratio is None:
            self.C13_12_ratio = 0
        if self.O18_16_ratio is None:
            self.O18_16_ratio = 0
        if self.O17_16_ratio is None:
            self.O17_16_ratio = 0

        self.temperature = temperature

        # Retrieve the mass fractions
        self.get_pRT_mass_fractions(params)

        self.unquenched_mass_fractions = self.mass_fractions.copy()
        self.P_quench = {}
        for quench_key, species_to_quench in self.quench_setup.items():

            if params.get(quench_key) is None:
                continue

            # Add to all quenching points
            self.P_quench[quench_key] = params.get(quench_key)

            # Quench this chemical network
            self.quench_chemistry(quench_key)

        # Remove certain species
        self.remove_species()

        return self.mass_fractions
    
class SPHINXChemistry(Chemistry):
    
    
    isotopologues_dict = {'12CO': ['13CO', 'C18O', 'C17O'], 
                    'H2O': ['H2O_181', 'H2O_171']}
    # reverse dictionary so every value is a key
    isotopologues_dict_rev = {value: key for key, values in isotopologues_dict.items() for value in values}
    isotopologues = list(isotopologues_dict_rev.keys())
    
    def __init__(self, line_species, pressure, **kwargs):

        # Give arguments to the parent class
        super().__init__(line_species, pressure)
        
        assert kwargs.get('vmr_interpolator') is not None, 'No VMR interpolator given'
        self.vmr_interpolator = kwargs.get('vmr_interpolator')
        
        self.sphinx_species = kwargs.get('species')
        # replace keys
        self.replace_keys = {
                            "CO": "12CO",
                            #  'H2H2': 'H2',
                            # 'H2He': 'He',
                            # 'HMFF': 'H-',
                                }
        # TODO: review H- and e- mixing ratios... take them from HMFF and H2H2?
        
        if "line_species_dict" in kwargs.keys():
            # update the pRT_name_dict
            self.pRT_name_dict_r.update(kwargs.get('line_species_dict'))
            self.pRT_name_dict = {v: k for k, v in self.pRT_name_dict_r.items()}
            # print(f' pRT_name_dict = {self.pRT_name_dict}')
        
    @property
    def CO(self): # alias for C/O ratio
        return self.C_O
    
    @property
    def FeH(self): # alias for Fe/H ratio
        return self.Z

    def __call__(self, params):

        # Update the parameters
        grid_attrs = ['Teff', 'logg', 'Z', 'C_O']
        
        assert all([params.get(attr) is not None for attr in grid_attrs]), 'Missing grid attributes'
        [setattr(self, attr, params.get(attr)) for attr in grid_attrs]
        
        
        self.VMRs = {}
        for s in self.sphinx_species:
            self.VMRs[s] = self.vmr_interpolator[s]([self.Teff, self.logg, self.Z, self.C_O])[0]
            
            # print(f' VMR_{s} = {self.VMRs[s]}')
            # assert len(self.VMRs[s]) == 40, f' VMR_{s} has wrong length: {len(self.VMRs[s])}'
        # VMRs_values = self.vmr_interpolator([self.Teff, self.logg, self.Z, self.C_O]) # shape (n_layers, n_species)
        # self.VMRs = dict(zip(self.sphinx_species, VMRs_values.T))
        self.VMRs = {self.replace_keys.get(k, k):v for k,v in self.VMRs.items()}
        
        # Total VMR without H2, starting with He
        VMR_He = 0.15
        VMR_wo_H2 = 0 + VMR_He

        # Create a dictionary for all used species
        self.mass_fractions = {}

        C, O, H = 0, 0, 0

        # for species_i in self.species_info.keys():
        for line_species_i in self.line_species:
            # line_species_i = self.read_species_info(species_i, 'pRT_name')
            # print(f' pRT_name_dict = {self.pRT_name_dict}')
            species_i = self.pRT_name_dict.get(line_species_i, None)
            # check alpha enhancement
            alpha_i = params.get(f'alpha_{species_i}', 0.0)
            # print(f' species_i = {species_i}, line_species_i = {line_species_i}, alpha_i = {alpha_i}')
            if species_i is None:
                continue
            
            mass_i = self.read_species_info(species_i, 'mass')
            COH_i  = [self.read_species_info(species_i, 'C'), 
                      self.read_species_info(species_i, 'O'), 
                      self.read_species_info(species_i, 'H')]

            if species_i in ['H2', 'He']:
                continue

            # Convert VMR to mass fraction using molecular mass number
            # print(f' {line_species_i} ({species_i})')
            # print(f' self.isotopologues = {self.isotopologues}')
            if species_i in self.isotopologues:
                main = self.isotopologues_dict_rev[species_i]
                alpha_main = params.get(f'alpha_{main}', 0.0)
                ratio = params.get(f'{main}/{species_i}') # in VMR
                assert ratio is not None, f'No ratio {main}/{species_i} given'
                self.VMRs[species_i] = (self.VMRs[main] * 10.**alpha_main) / ratio
                # self.mass_fractions[line_species_i] = mass_i * self.VMRs[species_i]
            # elif species_i in params.keys():
            elif species_i in params.keys():
                # self.mass_fractions[line_species_i] = mass_i * params[species_i] * np.ones(self.n_atm_layers)
                self.VMRs[species_i] = params[species_i] * np.ones(self.n_atm_layers)
            # else:   
            #     self.mass_fractions[line_species_i] = mass_i * self.VMRs[species_i] # VMRs is already an array
            elif species_i not in self.VMRs.keys():
                print(f' WARNING: {species_i} not in VMRs, setting to 0')
                self.VMRs[species_i] = 0.0 * np.ones(self.n_atm_layers)
                
            self.mass_fractions[line_species_i] = mass_i * (self.VMRs[species_i] * 10.**alpha_i) # VMRs is already an array
            VMR_wo_H2 += self.VMRs[species_i]


        # Add the H2 and He abundances
        self.mass_fractions['He'] = self.read_species_info('He', 'mass') * VMR_He * np.ones(self.n_atm_layers)
        self.mass_fractions['H2'] = self.read_species_info('H2', 'mass') * (1 - VMR_wo_H2)
    
        # self.mass_fractions['H-'] = 6e-9 # solar
        self.mass_fractions['H-'] = self.VMRs.get('H-', 6e-9) * np.ones(self.n_atm_layers)
        # self.mass_fractions['e-'] = 1e-10# solar
        self.mass_fractions['e-'] = 1e-10 * (self.mass_fractions['H-'] / 6e-9)
        
    
        # Add to the H-bearing species
        H += self.read_species_info('H2', 'H') * (1 - VMR_wo_H2)
        self.mass_fractions['H'] = H

        if VMR_wo_H2.any() > 1: #or (self.mass_fractions['H2'] > 1).any():
            # Other species are too abundant
            self.mass_fractions = -np.inf
            print(f' VMR_wo_H2 = {VMR_wo_H2} > 1 --> mass_fractions = -np.inf')
            return self.mass_fractions

        # Compute the mean molecular weight from all species
        # assert 'CO_high_Sam' in self.mass_fractions.keys(), 'CO_high_Sam not found'
        MMW = np.sum([mass_i for mass_i in self.mass_fractions.values()], axis=0)
        # print(f' MMW = {MMW}')

        # Turn the molecular masses into mass fractions
        for line_species_i in self.mass_fractions.keys():
            self.mass_fractions[line_species_i] /= MMW

        # pRT requires MMW in mass fractions dictionary
        self.mass_fractions['MMW'] = MMW
        return self.mass_fractions
    
class FastChemistry(Chemistry):
    isotopologues_dict = {'12CO': ['13CO', 'C18O', 'C17O'], 
                    'H2O': ['H2O_181', 'H2O_171']}
    # reverse dictionary so every value is a key
    isotopologues_dict_rev = {value: key for key, values in isotopologues_dict.items() for value in values}
    isotopologues = list(isotopologues_dict_rev.keys())
    
    
    def __init__(self, line_species, pressure, **kwargs):

        # Give arguments to the parent class
        self.species_info = self.set_species_info(file=self.species_info_default_file)
        if isinstance(line_species, dict):
            self.species_info = self.set_species_info(line_species_dict=line_species)

            line_species = list(line_species.values())
            
            
        super().__init__(line_species, pressure)
        self.species = [self.pRT_name_dict.get(line_species_i, None) for line_species_i in self.line_species]
        
        assert kwargs.get('fastchem_grid_file') is not None, 'No fastchem grid file given'
        fc_grid_file = kwargs.get('fastchem_grid_file')
        # check if file exists
        assert os.path.exists(fc_grid_file), f'File {fc_grid_file} not found'
        self.load_grid(fc_grid_file)
    
    def load_grid(self, file):
        with h5py.File(file, 'r') as f:
            data = f['data'][:]
            self.t_grid = f.attrs['temperature']
            self.p_grid = f.attrs['pressure']
            # Retrieve the metadata (column headers and description)
            columns = [col.decode() if isinstance(col, bytes) else col for col in f.attrs['columns']]  # handle bytes or str
            description = f.attrs['description'] if isinstance(f.attrs['description'], str) else f.attrs['description'].decode()
            species = f.attrs['species'].tolist()
            # labels  = [str(l) for l in f.attrs['labels']]
            labels = f.attrs['labels'].tolist()
        
        self.data = np.moveaxis(data, 0, 1) # shape (n_pressure, n_temperature, n_species) -> (n_temperature, n_pressure, n_species)
        self.fc_species_dict = dict(zip(species, labels))
        
        # create interpolator for each species
        self.interpolator = {}
        for i, (species_i, label_i) in enumerate(self.fc_species_dict.items()):
            if (species_i in self.species) or (species_i in ['H2', 'He', 'e-']):
                
                if label_i in columns:
                    idx = columns.index(label_i)
                    print(f' Loading interpolator for {label_i} ({species_i}) from column {idx}')
                    self.interpolator[species_i] = RegularGridInterpolator((self.t_grid, self.p_grid), self.data[:,:,idx], bounds_error=False, fill_value=None)
                # else:
                    # print(f' WARNING: {label_i} not in columns')
            # else:
            #     print(f' WARNING: {species_i} not in line_species')
        
        return self
    
    
    
    
    def __call__(self, params, temperature):
        
        assert len(temperature) == len(self.pressure), f' Len(temperature) = {len(temperature)} != len(pressure) = {len(self.pressure)}'
        self.VMRs = {k: self.interpolator[k]((temperature, self.pressure)) for k in self.interpolator.keys()}
        self.mass_fractions = {}
        
        for line_species_i, species_i in zip(self.line_species, self.species):
            
            alpha_i = params.get(f'alpha_{species_i}', 0.0)
            
            if species_i is None:
                continue
            
            mass_i = self.read_species_info(species_i, 'mass')
            # print(f' species_i = {species_i}, line_species_i = {line_species_i}, alpha_i = {alpha_i}')
            if species_i in self.isotopologues:
                main = self.isotopologues_dict_rev[species_i]
                alpha_main = params.get(f'alpha_{main}', 0.0)
                ratio = params.get(f'{main}/{species_i}') # in VMR
                assert ratio is not None, f'No ratio {main}/{species_i} given'
                self.VMRs[species_i] = (self.VMRs[main] * 10.**alpha_main) / ratio
                
            elif species_i in params.keys():
                self.VMRs[species_i] = params[species_i] * np.ones(self.n_atm_layers)
                
            elif species_i not in self.VMRs.keys():
                # print(f' WARNING: {species_i} not in VMRs, setting to 0')
                self.VMRs[species_i] = 0.0 * np.ones(self.n_atm_layers)
                
            self.mass_fractions[line_species_i] = mass_i * (self.VMRs[species_i] * 10.**alpha_i) # VMRs is already an array
                
        self.mass_fractions['He'] = self.read_species_info('He', 'mass') * self.VMRs['He']
        self.mass_fractions['H2'] = self.read_species_info('H2', 'mass') * self.VMRs['H2']
        self.mass_fractions['H']  = self.VMRs['H2']
        
        # mass of electron in amu
        mass_e = 5.48579909070e-4
        self.mass_fractions['e-'] = mass_e * self.VMRs['e-']
        self.mass_fractions['H-'] = self.VMRs['e-'] # amu of H is 1.0
        # self.mass_fractions['H-'] = 6e-9 * (self.VMRs['e-'] / 6e-9) # scale with respect to solar using e-
        # self.mass_fractions['e-'] = self.VMRs['e-']
        MMW = np.sum([mass_i for mass_i in self.mass_fractions.values()], axis=0)
        # hot fix for OH linelist 
        # print(f' [FastChemistry] self.mass_fractions.keys() = {self.mass_fractions.keys()}')
        # if 'OH_MYTHOS_main_iso' in self.mass_fractions.keys():
            # self.mass_fractions['OH_MoLLIST_main_iso'] = self.mass_fractions['OH_MYTHOS_main_iso']
        
        # Turn the molecular masses into mass fractions
        for line_species_i in self.mass_fractions.keys():
            self.mass_fractions[line_species_i] /= MMW
        # pRT requires MMW in mass fractions dictionary
        self.mass_fractions['MMW'] = MMW
        return self.mass_fractions
    
    @property
    def CO(self): # alias for C/O ratio
        if hasattr(self, 'C_O'):
            return self.C_O
        else:
            return np.nan
    
    @property
    def FeH(self): # alias for Fe/H ratio
        # return self.Z
        if hasattr(self, 'Z'):
            return self.Z
        else:
            return np.nan
    
if __name__ == '__main__':
    
    # chem = Chemistry()
    
    # amu = {k:np.round(v[-2],3) for k,v in Chemistry.species_info.items()}
    # amu = {v[0]:np.round(v[-2],3) for k,v in Chemistry.species_info.items()}
    # # save as .txt file with two columns
    # with open('data/lbl_masses.txt', 'w') as f:
    #     for key in amu.keys():
    #         f.write(f"{key:28} {amu[key]:6.3f}")
    #         if key != list(amu.keys())[-1]:
    #             f.write('\n')
    # print(f' data/lbl_masses.txt saved!')
    
    kwargs = dict(fastchem_grid_file='/home/dario/phd/fastchem/output/output_grid.h5')
    
    pressure = np.logspace(-5, 2, 40)
    temperature = np.linspace(1000.0, 3000.0, len(pressure))
    opacity_params = {
    'log_12CO': ([(-12,-2), r'$\log\ \mathrm{^{12}CO}$'], 'CO_high_Sam'),
    'log_13CO': ([(-12,-2), r'$\log\ \mathrm{^{13}CO}$'], 'CO_36_high_Sam'),
    'log_C18O': ([(-14,-2), r'$\log\ \mathrm{C^{18}O}$'], 'CO_28_high_Sam'),
        
    'log_H2O': ([(-12,-2), r'$\log\ \mathrm{H_2O}$'], 'H2O_pokazatel_main_iso'),
    'log_H2O_181': ([(-14,-2), r'$\log\ \mathrm{H_2^{18}O}$'], 'H2O_181_HotWat78'),
    
    'log_HF': ([(-14,-2), r'$\log\ \mathrm{HF}$'], 'HF_high'),
    'log_Na': ([(-14,-2), r'$\log\ \mathrm{Na}$'], 'Na_allard_high'),
    'log_Ca': ([(-14,-2), r'$\log\ \mathrm{Ca}$'], 'Ca_high'), 
    'log_Ti': ([(-14,-2), r'$\log\ \mathrm{Ti}$'], 'Ti_high'), 
    'log_Mg': ([(-14,-2), r'$\log\ \mathrm{Mg}$'], 'Mg_high'),
    'log_Fe': ([(-14,-2), r'$\log\ \mathrm{Fe}$'], 'Fe_high'),
    'log_Sc': ([(-14,-2), r'$\log\ \mathrm{Sc}$'], 'Sc_high'),
    'log_OH': ([(-14,-2), r'$\log\ \mathrm{OH}$'], 'OH_MMMYTHOS_main_iso'),
    'log_CN': ([(-14,-2), r'$\log\ \mathrm{CN}$'], 'CN_high'),
    }
    
    keys = [k.split('log_')[-1] for k in opacity_params.keys()]
    values = [v[1] for v in opacity_params.values()]
    line_species_dict = dict(zip(keys, values))
    # chem = FastChemistry(['CO_high_Sam', 'H2O_pokazatel_main_iso', 'OH_MYTHOS_main_iso', 'Sc_high'], pressure, **kwargs)

    # # update pRT_name with line_species_dict
    # line_species_dict_default = dict(zip(chem.species_info['name'].tolist(), chem.species_info['pRT_name'].tolist()))
    # line_species_dict_new = line_species_dict_default.copy()
    # line_species_dict_new.update(line_species_dict)
    
    # # update pRT_name with line_species_dict
    # chem.species_info['pRT_name'] = chem.species_info['name'].map(line_species_dict_new)
    chem = FastChemistry(line_species={'12CO':'CO_high_Sam', 'OH':'OH_main_iso'}, pressure=pressure, **kwargs)
    
    
    
    params = {'alpha_12CO':-1.0,
              'alpha_H2O':-0.2,
              'Sc': 1e-7,
    }
    mf = chem(params, temperature)
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,2, figsize=(12,6), sharey=True, tight_layout=True)
    ax[0].plot(temperature, pressure, 'k-')
    ax[0].set(yscale='log', xscale='linear', ylim=(pressure.max(), pressure.min()), xlabel='Temperature [K]', ylabel='Pressure [bar]')
    
    for i, (k,v) in enumerate(chem.VMRs.items()):
        ax[1].plot(v, pressure, label=k)
        
    ax[1].set(yscale='log', xscale='log', ylim=(pressure.max(), pressure.min()), xlabel='Mass fraction', ylabel='Pressure [bar]')
    ax[1].legend()
    plt.show()