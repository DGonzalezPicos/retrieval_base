import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline
import petitRADTRANS.nat_cst as nc
import pathlib
import os
import h5py
from scipy.interpolate import RegularGridInterpolator

from retrieval_base.auxiliary_functions import quantiles, get_path
path = get_path()

def get_Chemistry_class(line_species, pressure, mode, **kwargs):

    if mode == 'free' or mode == 'freechem':
        return FreeChemistry(line_species, pressure, **kwargs)
    if mode == 'eqchem':
        return EqChemistry(line_species, pressure, **kwargs)
    if mode == 'fastchem':
        return FastChemistry(line_species, pressure, **kwargs)
    if mode == 'SONORAchem':
        return SONORAChemistry(line_species, pressure, **kwargs)

class Chemistry:

    # Dictionary with info per molecular/atomic species
    # Dictionary with info per molecular/atomic species
    # species_info = pd.read_csv(f'{path}data/species_info.csv')
    # # create alias column 'mathtext_name' to 'label'
    # species_info['label'] = species_info['mathtext_name']
    # pRT_name_dict = {v['pRT_name']: v['name'] for i, v in species_info.iterrows()}
    
    # Dictionary with info per molecular/atomic species
    species_info_default_file = f'{path}data/species_info.csv'
    # species_info = pd.read_csv(species_info_default_file)
    
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
        
            file_posterior = pathlib.Path(save_to) / 'VMR_posteriors.npy'
            file_envelopes = pathlib.Path(save_to) / 'VMR_envelopes.npy'
            file_labels    = pathlib.Path(save_to) / 'VMR_labels.npy'
            np.save(file_posterior, np.array(list(self.VMRs_posterior.values())))
            np.save(file_envelopes, np.array(list(self.VMRs_envelopes.values())))
            np.save(file_labels, np.array(list(self.VMRs_posterior.keys())))
            print(f'[Chemistry.get_VMRs_posterior] Saved VMRs posterior and envelopes to:\n {file_posterior}\n {file_envelopes}\n {file_labels}')
        return self
    
    
    def two_point_profile(self, log_K_1, log_K_2, log_K_P):
        """
        Create a two-point profile for a given species
        log_K_1: log(K) at the bottom of the atmosphere
        log_K_2: log(K) at the top of the atmosphere
        log_K_P: log(K) at the pressure where the profile changes
        log_p: log(pressure)
        """
        log_p = np.log10(self.pressure)
        assert log_p[0] < log_p[-1], 'Pressure must be in ascending order'
        assert log_K_P < log_p.max(), f'log_K_P = {log_K_P} must be smaller than the bottom of the atmosphere ({log_p.max()})'
        assert log_K_P > log_p.min(), f'log_K_P = {log_K_P} must be larger than the top of the atmosphere ({log_p.min()})'
        top = log_p <= log_K_P
        log_K_slope = (log_K_1 - log_K_2) / (log_K_P - log_p[0])
        log_K = log_K_1 * np.ones(len(log_p))
        log_K[top] = log_K_1 + log_K_slope * (log_p[top] - log_K_P)
        K = 10**log_K
        return K
    
    def get_carbon_oxygen_ratio(self):
        """
        Get the C/O ratio from the mass fractions
        """
        
        assert hasattr(self, 'mass_fractions'), 'Mass fractions not yet evaluated'
        C, O, H = 0, 0, 0
        MMW = self.mass_fractions['MMW']
        for line_species_i in self.line_species:
            species_i = self.pRT_name_dict.get(line_species_i, None)
            mass_i = self.read_species_info(species_i, 'mass')
            C_i, O_i, H_i = self.read_species_info(species_i, 'C'), self.read_species_info(species_i, 'O'), self.read_species_info(species_i, 'H')
            VMR_i = self.mass_fractions[line_species_i] * (MMW / mass_i)
            C += C_i * VMR_i
            O += O_i * VMR_i
            H += H_i * VMR_i
            
        H += 2. * self.mass_fractions['H2'] * (MMW / self.read_species_info('H2', 'mass'))
        # H += 1. * self.mass_fractions['H'] * (MMW / self.read_species_info('H', 'mass'))
        
            
        self.ratios = {'C/O': np.nanmean(C / O), 
                    #    'C/H': np.mean(C / H), 
                       '[C/H]': np.nanmean(np.log10(C/H)) - (8.46 - 12) # Asplund et al. (2021)
                       }
        return self.ratios


class FreeChemistry(Chemistry):

    def __init__(self, line_species, pressure, spline_order=0, **kwargs):

        # Give arguments to the parent class
        # super().__init__(line_species, pressure)

        # self.spline_order = spline_order
        
        # Give arguments to the parent class
        self.species_info = self.set_species_info(file=self.species_info_default_file)
        if isinstance(line_species, dict):
            # print(f' Updating species_info with {line_species}')
            self.species_info = self.set_species_info(line_species_dict=line_species)

            line_species = list(line_species.values())
            
            
        super().__init__(line_species, pressure)
        self.species = [self.pRT_name_dict.get(line_species_i, None) for line_species_i in self.line_species]

    def __call__(self, params):

        # self.VMRs = VMRs

        # Total VMR without H2, starting with He
        VMR_He = 0.15
        VMR_wo_H2 = 0 + VMR_He

        # Create a dictionary for all used species
        self.VMRs = {}
        self.mass_fractions = {}

        C, O, H = 0, 0, 0

        # for species_i in self.species_info.keys():
        for line_species_i, species_i in zip(self.line_species, self.species):
            # print(f'line_species_i = {line_species_i}, species_i = {species_i}')
            
                
            if species_i in ['H2', 'He']:
                continue
            
            # if line_species_i not in self.line_species:
            #     continue
            
            mass_i = self.read_species_info(species_i, 'mass')
            # COH_i  = self.read_species_info(species_i, 'COH')
            COH_i  = [self.read_species_info(species_i, 'C'), 
                      self.read_species_info(species_i, 'O'), 
                      self.read_species_info(species_i, 'H')]

            # if line_species_i in self.line_species:
            # print(f' self.VMRs.keys() = {self.VMRs.keys()}')
                
            # if self.VMRs.get(species_i) is not None:
            # if params.get(species_i, None) is None:
            #     print(f' params[{species_i}] = {params[species_i]}')
            #     continue
            
            # Single value given: constant, vertical profile
            if params.get(f'log_{species_i}_P', None) is not None:
                assert params.get(f'log_{species_i}_1', None) is not None, f'log_{species_i}_1 not given'
                assert params.get(f'log_{species_i}_2', None) is not None, f'log_{species_i}_2 not given'
                assert params.get(f'log_{species_i}_P', None) is not None, f'log_{species_i}_P not given'
                # print(f' Creating two-point profile for {species_i}')
                VMR_i = self.two_point_profile(params[f'log_{species_i}_1'], 
                                               params[f'log_{species_i}_2'], 
                                               params[f'log_{species_i}_P'])
                
                assert len(VMR_i) == self.n_atm_layers, f'len(VMR_i) = {len(VMR_i)} != len(pressure) = {self.n_atm_layers}'
            else:
                # print(f' Using constant profile for {species_i}')
                VMR_i = params[species_i] * np.ones(self.n_atm_layers)

            # self.VMRs[species_i] = VMR_i

            # Convert VMR to mass fraction using molecular mass number
            self.mass_fractions[line_species_i] = mass_i * VMR_i
            VMR_wo_H2 += VMR_i

            # Record C, O, and H bearing species for C/O and metallicity
            C += COH_i[0] * VMR_i
            O += COH_i[1] * VMR_i
            H += COH_i[2] * VMR_i

        # Add the H2 and He abundances
        self.mass_fractions['He'] = self.read_species_info('He', 'mass') * VMR_He * np.ones(self.n_atm_layers)
        # if 'H2' in self.VMRs.keys():
        #     self.mass_fractions['H2'] = self.read_species_info('H2', 'mass') * 
            
        self.mass_fractions['H2'] = self.read_species_info('H2', 'mass') * (1 - VMR_wo_H2) * np.ones(self.n_atm_layers)
        if self.read_species_info('H2', 'pRT_name') in self.line_species:
            self.mass_fractions[self.read_species_info('H2', 'pRT_name')] = self.mass_fractions['H2']
        
    
        # Add to the H-bearing species
        H += self.read_species_info('H2', 'H') * (1 - VMR_wo_H2)
        # self.mass_fractions['H'] = H * self.mass_fractions['H2'] / self.read_species_info('H2', 'mass')
        
        if 'Hminus' in params.keys():
            # print(f' [FastChemistry] params["Hminus"] = {params["Hminus"]}')
            self.mass_fractions['H-'] = params['Hminus'] * np.ones(self.n_atm_layers)
        else:
            self.mass_fractions['H-'] = 6e-9 * np.ones(self.n_atm_layers)# FIXME: fix to solar value?
            
            
        self.mass_fractions['e-'] = 5.48579909070e-4 * self.mass_fractions['H-']
        # print('[FreeChemistry.__call__] VMR_wo_H2 =', VMR_wo_H2)
        if VMR_wo_H2.any() > 1:
            # Other species are too abundant
            self.mass_fractions = -np.inf
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
        # self.CO = C/O

        # log_CH_solar = 8.43 - 12 # Asplund et al. (2009)
        # # log_CH_solar = 8.46 - 12 # Asplund et al. (2021)
        # self.FeH = np.log10(C/H) - log_CH_solar
        # self.CH  = self.FeH

        # self.CO = np.mean(self.CO)
        # self.FeH = np.mean(self.FeH)
        # self.CH = np.mean(self.CH)

        # Remove certain species
        self.remove_species()
        
        self.ratios = {'C/O': np.nanmean(C / O), 
                    #    'C/H': np.mean(C / H), 
                       '[C/H]': np.nanmean(np.log10(C/H)) - (8.46 - 12) # Asplund et al. (2021)
                       }

        return self.mass_fractions
    
    def get_VMRs_envelopes(self):
        
        # assert hasattr(self, 'mass_fractions_envelopes'), 'Mass fractions not yet evaluated.'
        assert hasattr(self, 'VMRs_posterior'), 'Mass fractions not yet evaluated.'
        q = [0.5-0.997/2, 0.5-0.95/2, 0.5-0.68/2, 0.5, 
             0.5+0.68/2, 0.5+0.95/2, 0.5+0.997/2
             ]   
        self.VMRs_envelopes = {key:quantiles(self.VMRs_posterior[key], q) for key in self.VMRs_posterior.keys()}
        return self
    
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
        
        # print('[FastChemistry] self.species = ', self.species)
        # create interpolator for each species
        self.interpolator = {}
        for i, (species_i, label_i) in enumerate(self.fc_species_dict.items()):
            # print(f' Looking for {label_i} ({species_i})')
            if (species_i in self.species) or (species_i in ['H2', 'He', 'e-']):
                
                if label_i in columns:
                    idx = columns.index(label_i)
                    # print(f' Loading interpolator for {label_i} ({species_i}) from column {idx}')
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
        
        VMR_He = 0.15
        VMR_wo_H2 = VMR_He * np.ones(self.n_atm_layers)
        C, O, H = 0, 0, 0

        for line_species_i, species_i in zip(self.line_species, self.species):
            
            alpha_i = params.get(f'alpha_{species_i}', 0.0)
            
            if species_i is None:
                continue
            
            mass_i = self.read_species_info(species_i, 'mass')
            COH_i  = [self.read_species_info(species_i, 'C'), 
                      self.read_species_info(species_i, 'O'), 
                      self.read_species_info(species_i, 'H')]
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
                
            VMR_i = np.clip((self.VMRs[species_i] * 10.**alpha_i),a_max=1e-1, a_min=1e-20) # VMRs is already an array
            self.mass_fractions[line_species_i] = VMR_i * mass_i
            VMR_wo_H2 += VMR_i
            
            C += COH_i[0] * VMR_i
            O += COH_i[1] * VMR_i
            H += COH_i[2] * VMR_i
            
            
        # self.mass_fractions['He'] = self.read_species_info('He', 'mass') * self.VMRs['He']
        self.mass_fractions['He'] = VMR_He * self.read_species_info('He', 'mass') * np.ones(self.n_atm_layers)
        # self.mass_fractions['H2'] = self.read_species_info('H2', 'mass') * self.VMRs['H2']
        VMR_wo_H2 = np.clip(VMR_wo_H2, a_max=0.99, a_min=0.001)
        # print(f' VMR_wo_H2 = {VMR_wo_H2}')
        self.mass_fractions['H2'] = self.read_species_info('H2', 'mass') * (1 - VMR_wo_H2) # already an array
        self.mass_fractions['H'] = self.VMRs.get('H', 1e-20 * np.ones(self.n_atm_layers))
        
        # Add number density (=vmr) of hydrogen atoms from H2 and H
        H += 2.0 * (1 - VMR_wo_H2)
        H += self.mass_fractions['H']
        
        # H- and electron density
        self.mass_fractions['e-'] = 5.485799e-4 * self.VMRs.get('e-', 1e-20 * np.ones(self.n_atm_layers))
        self.mass_fractions['H-'] = params.get('Hminus', 1e-20) * np.ones(self.n_atm_layers)
        
        
        # # mass of electron in amu
        # mass_e = 5.48579909070e-4
        # self.mass_fractions['e-'] = mass_e * self.VMRs['e-']
        # if 'Hminus' in params.keys():
        #     # print(f' [FastChemistry] params["Hminus"] = {params["Hminus"]}')
        #     self.mass_fractions['H-'] = params['Hminus'] * np.ones(self.n_atm_layers)
        # else:
        #     self.mass_fractions['H-'] = 6e-9 * np.ones(self.n_atm_layers)# FIXME: fix to solar value?
        # self.mass_fractions['H-'] = 6e-9 * (self.VMRs['e-'] / 6e-9) # scale with respect to solar using e-
        # self.mass_fractions['e-'] = self.VMRs['e-']
        # MMW = np.sum([mass_i for mass_i in self.mass_fractions.values()], axis=0)
        MMW = np.sum(np.array(list(self.mass_fractions.values())), axis=0)
        assert len(MMW) == self.n_atm_layers, f'MMW has wrong shape {len(MMW)} != {self.n_atm_layers}'
        # MMW = 0
        # for mass_i in self.mass_fractions.values():
        #     MMW += mass_i
        # MMW *= np.ones(self.n_atm_layers)
        # print(f' mmw = {MMW}')
        # assert all(MMW > 0), f' MMW = {MMW} has negative values'
        # hot fix for OH linelist 
        # print(f' [FastChemistry] self.mass_fractions.keys() = {self.mass_fractions.keys()}')
        # if 'OH_MYTHOS_main_iso' in self.mass_fractions.keys():
            # self.mass_fractions['OH_MoLLIST_main_iso'] = self.mass_fractions['OH_MYTHOS_main_iso']
        
        # Turn the molecular masses into mass fractions
        for line_species_i in self.mass_fractions.keys():
            self.mass_fractions[line_species_i] /= MMW
        # pRT requires MMW in mass fractions dictionary
        self.mass_fractions['MMW'] = MMW
        
        self.ratios = {'C/O': np.nanmean(C / O), 
                    #    'C/H': np.mean(C / H), 
                       '[C/H]': np.nanmean(np.log10(C/H)) - (8.46 - 12) # Asplund et al. (2021)
                       }
        # MMW_mean = np.mean( 
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
    
    # create_amu_file = False
    # if create_amu_file:
    #     amu = {v[0]:np.round(v[-2],3) for k,v in Chemistry.species_info.items()}
    #     # save as .txt file with two columns
    #     with open('data/lbl_masses.txt', 'w') as f:
    #         for key in amu.keys():
    #             f.write(f"{key:28} {amu[key]:6.3f}")
    #             if key != list(amu.keys())[-1]:
    #                 f.write('\n')
    #     print(f' data/lbl_masses.txt saved!')
        
    # chem = FreeChemistry(line_species=['12CO', 'H2O'],
    #                      pressure=np.logspace(-6, 2, 10),
    # )
    
    # test FastChemistry
    import matplotlib.pyplot as plt
    import json
    # p = np.logspace(-5, 2, 100)
    # t = np.linspace(1200, 3000, len(p))
    path = pathlib.Path(get_path(return_pathlib=True))

    target = 'TWA27A'
    w_set='NIRSpec'
    run = 'lbl20_G1_3'
    run_bf = 'lbl15_G2G3_8'

    # PT = af.pickle_load(path / target / 'retrieval_outputs' / run_bf / 'test_data' / 'bestfit_PT.pkl')
    bestfit = json.load(open(path / target / f'retrieval_outputs/{run_bf}/test_data/bestfit.json'))
    t = np.array(bestfit['temperature'])
    p = np.array(bestfit['pressure'])
    # bestfit['params'].update({'gratings' : ['g140h'] *4})
        
    line_species_dict = {'12CO': 'CO_high_Sam', 'H2O': 'H2O_pokazatel_main_iso'}
    
    fc_grid = '/home/dario/phd/retrieval_base/data/fastchem_grid_twx.h5'
    chem = FastChemistry(line_species=list(line_species_dict.values()),
                            pressure=p,
                            fastchem_grid_file=fc_grid
        )
    
    
    linelists = {'12CO': 'CO_high_Sam', 'H2O': 'H2O_pokazatel_main_iso'}
    # random float between 0 and 1
    randf = np.random.rand()
    fc_params = {f'alpha_{key}': -3.0 + (np.random.random() * 6) for key in linelists.keys()}
    mf = chem(fc_params, t)
    
    freechem = FreeChemistry(line_species=list(line_species_dict.values()),
                            pressure=p,
                            )
    # free_params = {key:10.0**(-8.0 + (np.random.random() * 4)) for key in linelists.keys()}
    # free_params = {key:np.mean(mf[val]) for key, val in line_species_dict.items()}
    free_params = {} # define volume mixing ratios for the species directly from the mean of the fastchem grid
    mmw = chem.mass_fractions['MMW'].mean()
    print(f' FastChem MMW = {mmw:.2f}')
    for key, val in line_species_dict.items():
        mass_i = chem.read_species_info(key, 'mass')
        free_params[key] = np.mean(mf[val]) * (mmw/mass_i)
        
        
    mf_free = freechem(free_params)
    print(f' FreeChem MMW = {mf_free["MMW"].mean():.2f}')
    # plt.plot(mf['12CO'], p)
    
                         
    fig, ax = plt.subplots(1,1, figsize=(8,6))
    for key, value in mf.items():
        if key in list(line_species_dict.values()):
            ax.plot(value, p, label=key)
            
            ax.plot(mf_free[key], p, label=key, ls='--')
            print(f'{key}: fastchem: {np.mean(value):.2e} freechem: {np.mean(mf_free[key]):.2e}')
            
            
        
    # plot abundance of H, H2, He
    species = ['H', 'H2', 'He', 'H-', 'e-']
    for key in species:
        ax.plot(mf[key], p, label=key)
        ax.plot(mf_free[key], p, label=key, ls='--')
        print(f'{key}: fastchem: {np.mean(mf[key]):.2e} freechem: {np.mean(mf_free[key]):.2e}')
            
            
    ax.set(xscale='log', yscale='log', ylim=(np.max(p), np.min(p)))
    ax.set_xlim(1e-9, 1e-2)
    ax.legend()
    plt.show()