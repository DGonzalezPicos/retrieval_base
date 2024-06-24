import numpy as np
from scipy.interpolate import make_interp_spline
import petitRADTRANS.nat_cst as nc

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
    # (pRT_name, pyfc_name, mass, number of (C,O,H) atoms
    species_info = {
        # '12CO':    ('CO_main_iso',             'C1O1',     12.011 + 15.999,            (1,1,0)), 
       '12CO':    ('CO_high',                 'C1O1',     12.011 + 15.999,            (1,1,0)), 
        # '13CO':    ('CO_36',                   None,       13.003355 + 15.999,         (1,1,0)), 
       '13CO':    ('CO_36_high',              None,       13.003355 + 15.999,         (1,1,0)), 
        'C18O':    ('CO_28',                   None,       12.011 + 17.9991610,        (1,1,0)), 
        'C17O':    ('CO_27',                   None,       12.011 + 16.999131,         (1,1,0)), 
  
        'H2O':     ('H2O_pokazatel_main_iso',  'H2O1',     2*1.00784 + 15.999,         (0,1,2)), 
       'H2O_181': ('H2O_181_HotWat78',          None,       2*1.00784 + 17.9991610,     (0,1,2)), 
       #'HDO':     ('HDO_voronin',             None,       1.00784 + 2.014 + 15.999,   (0,1,2)), 
  
        'CH4':     ('CH4_hargreaves_main_iso', 'C1H4',     12.011 + 4*1.00784,         (1,0,4)), 
       '13CH4':   ('CH4_31111_hargreaves',    None,       13.003355 + 4*1.00784,      (1,0,4)), 
  
        'NH3':     ('NH3_coles_main_iso',      'H3N1',     14.0067 + 3*1.00784,        (0,0,3)), 
        'HCN':     ('HCN_main_iso',            'C1H1N1_1', 1.00784 + 12.011 + 14.0067, (1,0,1)), 
        'CN' :     ('CN_main_iso',             'C1N1',     12.011 + 14.0067,           (1,0,0)),
        'H2S':     ('H2S_ExoMol_main_iso',     'H2S1',     2*1.00784 + 32.065,         (0,0,2)), 
        'FeH':     ('FeH_main_iso',            'Fe1H1',    55.845 + 1.00784,           (0,0,1)), 
        'CrH':     ('CrH_main_iso',            'Cr1H1',    51.9961 + 1.00784,          (0,0,1)), 
        'NaH':     ('NaH_main_iso',            'H1Na1',    22.989769 + 1.00784,        (0,0,1)), 
        'CaH':     ('CaH_XAB_main_iso',        'Ca1H1',    40.078 + 1.00784,            (0,0,1)),
        'TiH':     ('TiH_main_iso',            'Ti1H1',    47.867 + 1.00784,           (0,0,1)),
        'AlH':     ('AlH_main_iso',            'Al1H1',    26.981539 + 1.00784,        (0,0,1)),
        'Mg_H':    ('MgH_main_iso',            'Mg1H1',    24.305 + 1.00784,           (0,0,1)),

        'TiO':     ('TiO_48_Exomol_McKemmish', 'O1Ti1',    47.867 + 15.999,            (0,1,0)), 
        # 'VO':      ('VO_ExoMol_McKemmish',     'O1V1',     50.9415 + 15.999,           (0,1,0)), 
        'VO':     ('VO_HyVO_main_iso',     'O1V1',     50.9415 + 15.999,           (0,1,0)),
        'AlO':     ('AlO_main_iso',            'Al1O1',    26.981539 + 15.999,         (0,1,0)), 
        'CO2':     ('CO2_main_iso',            'C1O2',     12.011 + 2*15.999,          (1,2,0)),
    
        'HF':      ('HF_main_iso',             'F1H1',     1.00784 + 18.998403,        (0,0,1)), 
        'HCl':     ('HCl_main_iso',            'Cl1H1',    1.00784 + 35.453,           (0,0,1)), 
        'C2H2':  ('C2H2_main_iso',           'C2H2',     2*12.011 + 2*1.00784,       (2,0,2)),
        
        'H2':      ('H2_main_iso',                      'H2',       2*1.00784,                  (0,0,2)), 
       #'HD':      ('H2_12',                   None,       1.00784 + 2.014,            (0,0,2)), 
       'OH':      ('OH_main_iso',             'H1O1',     1.00784 + 15.999,           (0,1,1)),

        'K':       ('K',                       'K',        39.0983,                    (0,0,0)), 
        'Na':      ('Na_allard',               'Na',       22.989769,                  (0,0,0)), 
        'Ti':      ('Ti',                      'Ti',       47.867,                     (0,0,0)), 
        'Fe':      ('Fe',                      'Fe',       55.845,                     (0,0,0)), 
        'Ca':      ('Ca',                      'Ca',       40.078,                     (0,0,0)), 
        'Al':      ('Al',                      'Al',       26.981539,                  (0,0,0)), 
        'Mg':      ('Mg',                      'Mg',       24.305,                     (0,0,0)), 
        'Mn':     ('Mn',                      'Mn',       54.938044,                  (0,0,0)),
        'Si':      ('Si',                      'Si',       28.085,                     (0,0,0)),
        'He':      ('He',                      'He',       4.002602,                   (0,0,0)), 
        # ions
        'CaII':   ('Ca+',                    'CaII',     40.078,                     (0,0,0)),
        'FeII':   ('Fe+',                    'FeII',     55.845,                     (0,0,0)),
        }

    species_plot_info = {
        # '12CO': ('green', r'$^{12}$CO'), 
        '12CO': ('#71bf6e', r'$^{12}$CO'),
        # '13CO': ('chocolate', r'$^{13}$CO'), 
        '13CO': ('#e30b5d', r'$^{13}$CO'), 
        'C18O': ('C6', r'C$^{18}$O'), 
        'C17O': ('C7', r'C$^{17}$O'), 

        # 'H2O': ('deepskyblue', r'H$_2$O'), 
        'H2O': ('#3881bb', r'H$_2$O'), 
        'H2O_181': ('C7', r'H$_2^{18}$O'), 
        'HDO': ('b', r'HDO'), 

        'CH4': ('#ff71ce', r'CH$_4$'), 
        '13CH4': ('magenta', r'$^{13}$CH$_4$'), 
        
        'NH3': ('#b967ff', r'NH$_3$'), 
        'HCN': ('orange', r'HCN'), 
        'H2S': ('C11', r'H$_2$S'), 
        'FeH': ('C12', r'FeH'), 
        'CrH': ('C15', r'CrH'), 
        'NaH': ('C16', r'NaH'), 

        'TiO': ('C13', r'TiO'), 
        'VO': ('C15', r'VO'), 
        'AlO': ('C14', r'AlO'), 
        'CO2': ('C9', r'CO$_2$'),

        # 'HF': ('C14', r'HF'), 
         'HF': ('#00ced1', r'HF'), 
        'HCl': ('C15', r'HCl'), 
        
        #'H2': ('C16', r'H$_2$'), 
        'HD': ('C17', r'HD'), 

        'K': ('C18', r'K'), 
        # 'Na': ('purple', r'Na'), 
        'Na': ('#daa520', r'Na'), 
        # 'Ti': ('olive', r'Ti'), 
        'Ti': ('#ff8e56', r'Ti'), 
        'Fe': ('C21', r'Fe'), 
        # 'Ca': ('red', r'Ca'), 
        'Ca': ('#984ea3', r'Ca'), 
        'Al': ('C23', r'Al'), 
        'Mg': ('#845224', r'Mg'), 
        'Si': ('C25', r'Si'),
        #'He': ('C22', r'He'), 
        'CN': ('magenta', r'CN'),
        'C2H2': ('#50FF45', r'C$_2$H$_2$'),
        # '13CH4': ('magenta', r'$^{13}$CH$_4$'),
        # ions
        'CaII': ('C26', r'CaII'),
        'FeII': ('C27', r'FeII'),
        }

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

    @classmethod
    def read_species_info(cls, species, info_key):
        
        if info_key == 'pRT_name':
            return cls.species_info[species][0]
        if info_key == 'pyfc_name':
            return cls.species_info[species][1]
        
        if info_key == 'mass':
            return cls.species_info[species][2]
        
        if info_key == 'COH':
            return cls.species_info[species][3]
        if info_key == 'C':
            return cls.species_info[species][3][0]
        if info_key == 'O':
            return cls.species_info[species][3][1]
        if info_key == 'H':
            return cls.species_info[species][3][2]

        if info_key == 'c' or info_key == 'color':
            return cls.species_plot_info[species][0]
        if info_key == 'label':
            return cls.species_plot_info[species][1]
        
    def get_VMRs_posterior(self):
        
        assert hasattr(self, 'mass_fractions_posterior')
        self.VMRs_posterior = {}
        info = self.species_info
        MMW = self.mass_fractions_posterior['MMW'].mean() if hasattr(self, 'mass_fractions_posterior') else self.mass_fractions['MMW']

        for line_species_i in self.line_species:
            key_i = [key_i for key_i in info.keys() if info[key_i][0]==line_species_i][0]
            # check key_i is not empty
            # print(f' {line_species_i} ({key_i}) -> {info[key_i][2]}')
            if len(key_i) == 0:
                continue
            mu = info[key_i][2] # atomic mass
            # free-chemistry = constant VMR
            # WARNING: equilibrium chemistry should use the mean value or something else
            self.VMRs_posterior[key_i] = self.mass_fractions_posterior[line_species_i][:,0] * (MMW/ mu)
            
        if "13CO" in list(self.VMRs_posterior.keys()) and "12CO" in list(self.VMRs_posterior.keys()):
            self.VMRs_posterior["12_13CO"] = self.VMRs_posterior["12CO"] / self.VMRs_posterior["13CO"]
        
        if "H2O_181" in list(self.VMRs_posterior.keys()) and "H2O" in list(self.VMRs_posterior.keys()):
            self.VMRs_posterior["H2_16_18O"] = self.VMRs_posterior["H2O"] / self.VMRs_posterior["H2O_181"]
            
        if hasattr(self, 'CO_posterior'):
            self.VMRs_posterior["C/O"] = self.CO_posterior
        if hasattr(self, 'FeH_posterior'):
            self.VMRs_posterior["Fe/H"] = self.FeH_posterior
        del self.mass_fractions_posterior
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
                    VMR_knots = np.array([self.VMRs[f'{species_i}_{j}'] for j in range(3)])[::-1]
                    
                    # Use a k-th order spline to vary the abundance profile
                    spl = make_interp_spline(log_P_knots, np.log10(VMR_knots), k=self.spline_order)
                    VMR_i = 10**spl(np.log10(self.pressure))

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
        
    
        # self.mass_fractions['H-'] = 6e-9 # solar
        self.mass_fractions['H-'] = self.VMRs.get('H-', 6e-9)
        # self.mass_fractions['e-'] = 1e-10# solar
        self.mass_fractions['e-'] = 1e-10 * (self.mass_fractions['H-'] / 6e-9)
        
    
        # Add to the H-bearing species
        H += self.read_species_info('H2', 'H') * (1 - VMR_wo_H2)
        self.mass_fractions['H'] = H

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
    
    def VMRs_envelopes(self):
        
        assert hasattr(self, 'mass_fractions_envelopes'), 'Mass fractions not yet evaluated.'
        
        self.VMRs_envelopes = {}
        MMW = self.mass_fractions_envelopes['MMW'][3].mean()
        for line_species_i, key in zip(self.line_species, self.VMRs.keys()):
            
            # get atomic mass
            mu_i = self.read_species_info(key, 'mass')
            # print(f'mu_{key} = {mu_i}')
            # mu_i = 1.
            self.VMRs_envelopes[key] = self.mass_fractions_envelopes[line_species_i] * (MMW / mu_i)
        return self.VMRs_envelopes
    
    
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