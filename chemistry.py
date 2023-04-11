import numpy as np

import petitRADTRANS.poor_mans_nonequ_chem as pm

class Chemistry:

    # Dictionary with info per molecular/atomic species
    # (line_species name, mass, number of (C,O,H) atoms
    species_info = {'12CO': ('CO_main_iso', 12.011 + 15.999, (1,1,0)), 
                    'H2O': ('H2O_main_iso', 2*1.00784 + 15.999, (0,1,2)), 
                    'CH4': ('CH4_hargreaves_main_iso', 12.011 + 4*1.00784, (1,0,4)), 
                    '13CO': ('CO_36', 13.003355 + 15.999, (0,0,0)), 
                    'C18O': ('CO_28', 12.011 + 17.9991610, (0,0,0)), 
                    'H2O_181': ('H2O_181', 2*1.00784 + 17.9991610, (0,0,0)), 
                    'NH3': ('NH3_main_iso', 14.0067 + 3*1.00784, (0,0,3)), 
                    'CO2': ('CO2_main_iso', 12.011 + 2*15.999, (1,2,0)),
                    'HCN': ('HCN_main_iso', 1.00784 + 12.011 + 14.0067, (1,0,1)), 
                    'He': ('He', 4.002602, (0,0,0)), 
                    'H2': ('H2', 2*1.00784, (0,0,2)), 
                    }

    # Neglect certain species to find respective contribution
    neglect_species = {'12CO': False, 
                       'H2O': False, 
                       'CH4': False, 
                       '13CO': False, 
                       'C18O': False, 
                       'H2O_181': False, 
                       'NH3': False, 
                       'CO2': False,
                       'HCN': False, 
                       #'He': False, 
                       #'H2': False, 
                       }

    def __init__(self, line_species, pressure, temperature):

        self.line_species = line_species

        self.pressure     = pressure
        self.temperature  = temperature
        self.n_atm_layers = len(self.pressure)

    def remove_species(self, species):

        # Remove the contribution of the specified species
        for species_i in species:
            
            # Read the name of the pRT line species
            line_species_i = self.read_species_info(species_i, 'pRT_name')

            # Set mass fraction to negligible values
            # TODO: does 0 work?
            if line_species_i in line_species:
                self.mass_fractions[line_species_i] = 0

    @classmethod
    def read_species_info(cls, species, info_key):
        
        if info_key == 'pRT_name':
            return cls.species_info[species][0]
        elif info_key == 'mass':
            return cls.species_info[species][1]
        elif info_key == 'C':
            return cls.species_info[species][2][0]
        elif info_key == 'O':
            return cls.species_info[species][2][1]
        elif info_key == 'H':
            return cls.species_info[species][2][2]


class FreeChemistry(Chemistry):

    def __init__(self, line_species, pressure, temperature, VMRs):

        # Give arguments to the parent class
        super().__init__(line_species, pressure, temperature)

        self.VMRs = VMRs

    def get_mass_fractions(self):

         # Total VMR without H2, starting with He
        VMR_wo_H2 = 0.15

        # Create a dictionary for all used species
        self.mass_fractions = {}

        C, O, H = 0, 0, 0
        for species_i, (line_species_i, mass_i, COH_i) in Chemistry.species_info.items():

            if species_i in ['H2', 'He']:
                continue

            if line_species_i in self.line_species:
                # Convert VMR to mass fraction using molecular mass number
                self.mass_fractions[line_species_i] = mass_i * self.VMRs[species_i]
                VMR_wo_H2 += self.VMRs[species_i]

                # Record C, O, and H bearing species for C/O and metallicity
                C += COH_i[0] * self.VMRs[species_i]
                O += COH_i[1] * self.VMRs[species_i]
                H += COH_i[2] * self.VMRs[species_i]

        # Add the H2 and He abundances
        self.mass_fractions['He'] = self.read_species_info('He', 'mass') * VMR_wo_H2
        self.mass_fractions['H2'] = self.read_species_info('H2', 'mass') * (1 - VMR_wo_H2)

        # Add to the H-bearing species
        H += self.read_species_info('H2', 'H') * (1 - VMR_wo_H2)

        if self.mass_fractions['H2'] < 0:
            # Other species are too abundant
            self.mass_fractions = -np.inf
            return

        # Compute the mean molecular weight from all species
        self.mass_fractions['MMW'] = 0
        for mass_i in self.mass_fractions.values():
            self.mass_fractions['MMW'] += mass_i
        self.mass_fractions['MMW'] *= np.ones(self.n_atm_layers)

        # Turn the molecular masses into mass fractions
        for line_species_i in self.mass_fractions.keys():
            self.mass_fractions[line_species_i] /= self.mass_fractions['MMW']

        # Compute the C/O ratio and metallicity
        self.CO = C/O

        log_CH_solar = 8.43 - 12 # Asplund et al. (2009)
        self.FeH = np.log10(C/H) - log_CH_solar
        self.CH  = self.FeH


class EqChemistry(Chemistry):

    def __init__(self, line_species, pressure, temperature, CO, FeH, C_ratio=None, O_ratio=None, P_quench=None):

        # Give arguments to the parent class
        super().__init__(line_species, pressure, temperature)

        self.CO  = CO
        self.FeH = FeH

        self.C_ratio = C_ratio
        self.mass_ratio_13CO_12CO = self.read_species_info('13CO', 'mass') / \
                                    self.read_species_info('12CO', 'mass')

        self.O_ratio = O_ratio
        self.mass_ratio_C18O_12CO = self.read_species_info('C18O', 'mass') / \
                                    self.read_species_info('12CO', 'mass')
        self.mass_ratio_H2_18O_H2O = self.read_species_info('H2O_181', 'mass') / \
                                     self.read_species_info('H2O', 'mass')


        self.P_quench = P_quench

    def quench_carbon_chemistry(self, pm_mass_fractions):

        # Own implementation of quenching, using interpolation

        # Layers to be replaced by a constant abundance
        mask_quenched = (self.pressure < self.P_quench)

        #for species_i in ['CO', 'H2O', 'CH4', 'CO2', 'HCN']:
        for species_i in ['CO', 'CH4', 'CO2', 'HCN']:
            mass_fraction_i = pm_mass_fractions[species_i]
            mass_fraction_i[mask_quenched] = np.interp(self.P_quench, 
                                                       xp=self.pressure, 
                                                       fp=mass_fraction_i
                                                       )
            pm_mass_fractions[species_i] = mass_fraction_i

        return pm_mass_fractions

    def get_mass_fractions(self):

        # Retrieve the mass fractions from the chem-eq table
        pm_mass_fractions = pm.interpol_abundances(self.CO*np.ones(self.n_atm_layers), 
                                                   self.FeH*np.ones(self.n_atm_layers), 
                                                   self.temperature, 
                                                   self.pressure
                                                   )
        
        self.mass_fractions = {'MMW': pm_mass_fractions['MMW']}

        if self.P_quench is not None:
            pm_mass_fractions = self.quench_carbon_chemistry(pm_mass_fractions)

        for line_species_i in line_species:
            if species_i == 'CO_main_iso':

                # 12CO mass fraction
                self.mass_fractions[species_i] = (1 - self.C_ratio * self.mass_ratio_13CO_12CO - \
                                                  self.O_ratio * self.mass_ratio_C18O_12CO
                                                  ) * pm_mass_fractions['CO']
            elif species_i == 'CO_36':
                # 13CO mass fraction
                self.mass_fractions['CO_36'] = self.C_ratio * self.mass_ratio_13CO_12CO * pm_mass_fractions['CO']
            elif species_i == 'CO_28':
                # C18O mass fraction
                self.mass_fractions['CO_28'] = self.O_ratio * self.mass_ratio_C18O_12CO * pm_mass_fractions['CO']
            else:
                self.mass_fractions[species_i] = pm_mass_fractions[species_i.split('_')[0]]

        # Add the H2 and He abundances
        self.mass_fractions['H2'] = pm_mass_fractions['H2']
        self.mass_fractions['He'] = pm_mass_fractions['He']