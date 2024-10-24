import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline
import petitRADTRANS.nat_cst as nc
import pathlib

from retrieval_base.auxiliary_functions import quantiles, get_path
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


class FreeChemistry(Chemistry):

    def __init__(self, line_species, pressure, spline_order=0, **kwargs):

        # Give arguments to the parent class
        # super().__init__(line_species, pressure)

        # self.spline_order = spline_order
        
        # Give arguments to the parent class
        self.species_info = self.set_species_info(file=self.species_info_default_file)
        if isinstance(line_species, dict):
            print(f' Updating species_info with {line_species}')
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
            if params.get(species_i, None) is None:
                print(f' params[{species_i}] = {params[species_i]}')
                continue
            
            # Single value given: constant, vertical profile
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
        
    
        # self.mass_fractions['H-'] = 6e-9 # solar
        self.mass_fractions['H-'] = params.get('H-', 6e-9) * np.ones(self.n_atm_layers)
        # self.mass_fractions['e-'] = 1e-10# solar
        self.mass_fractions['e-'] = 1e-10 * (self.mass_fractions['H-'] / 6e-9) * np.ones(self.n_atm_layers)
        
    
        # Add to the H-bearing species
        H += self.read_species_info('H2', 'H') * (1 - VMR_wo_H2)
        self.mass_fractions['H'] = H * self.mass_fractions['H2'] / self.read_species_info('H2', 'mass')

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
        
        # assert hasattr(self, 'mass_fractions_envelopes'), 'Mass fractions not yet evaluated.'
        assert hasattr(self, 'VMRs_posterior'), 'Mass fractions not yet evaluated.'
        q = [0.5-0.997/2, 0.5-0.95/2, 0.5-0.68/2, 0.5, 
             0.5+0.68/2, 0.5+0.95/2, 0.5+0.997/2
             ]   
        self.VMRs_envelopes = {key:quantiles(self.VMRs_posterior[key], q) for key in self.VMRs_posterior.keys()}
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
    
    
if __name__ == '__main__':
    
    # chem = Chemistry()
    
    # amu = {k:np.round(v[-2],3) for k,v in Chemistry.species_info.items()}
    
    create_amu_file = False
    if create_amu_file:
        amu = {v[0]:np.round(v[-2],3) for k,v in Chemistry.species_info.items()}
        # save as .txt file with two columns
        with open('data/lbl_masses.txt', 'w') as f:
            for key in amu.keys():
                f.write(f"{key:28} {amu[key]:6.3f}")
                if key != list(amu.keys())[-1]:
                    f.write('\n')
        print(f' data/lbl_masses.txt saved!')
        
    chem = FreeChemistry(line_species=['12CO', 'H2O'],
                         pressure=np.logspace(-6, 2, 10),
    )