import numpy as np
import os

file_params = 'config_freechem.py'

####################################################################################
# Files and physical parameters
####################################################################################

run = 'sphinx1'
prefix = f'./retrieval_outputs/{run}/test_'
copy_pRT_from = None

config_data = {
    'spirou': {
        # 'w_set': 'G395H_F290LP', 'wave_range': (4100, 5300), 
        'w_set': 'spirou',
        # 'wave_range': (1650, 3200), # g235h-f170lp
        
        # 'orders': (30,31,32,45,46,47,48), # J+K band
        # 'orders': (46,47,48), # K band
        'orders': (0,1,2), # file only contains 3 orders
        # 'wave_range': (2295, 2440), # 2 orders
        # 'wave_range': (1630, 3250), 
        
        'lbl_opacity_sampling' : 5,
        'slit': 'spirou',
        # 'lbl_opacity_sampling' : None,
        'sigma_clip': 5,
        'sigma_clip_width': 11, 
        'Nedge': 50, # DGP (2024-07-16): update from 30 --> 50
        'tell_threshold': 0.55,
        'tell_n_grow': 10,
        'emission_line_threshold': 1.3,
        
        'log_P_range': (-5,2),
        'n_atm_layers': 40, # WARNING: 40 for SPHINX
        
        'file_target':'data/spec_orders_46_47_48_mad.npy' # sep 28, new data with proper normalization and MAD error estimate
        }, 
    }


####################################################################################
# Model parameters
####################################################################################
opacity_params = {
    'log_12CO': ([(-12,-2), r'$\log\ \mathrm{^{12}CO}$'], 'CO_high_Sam'),
    'log_13CO': ([(-12,-2), r'$\log\ \mathrm{^{13}CO}$'], 'CO_36_high_Sam'),
    'log_C18O': ([(-14,-2), r'$\log\ \mathrm{C^{18}O}$'], 'CO_28_high_Sam'),
    # 'log_C17O': ([(-14,-2), r'$\log\ \mathrm{C^{17}O}$'], 'CO_27'),
    
    'log_H2O': ([(-12,-2), r'$\log\ \mathrm{H_2O}$'], 'H2O_pokazatel_main_iso'),
    'log_H2O_181': ([(-14,-2), r'$\log\ \mathrm{H_2^{18}O}$'], 'H2O_181_HotWat78'),
    # 'log_HDO': ([(-14,-2), r'$\log\ \mathrm{HDO}$'], 'HDO_voronin'),
    'log_HF': ([(-14,-2), r'$\log\ \mathrm{HF}$'], 'HF_high'), # DGP (2024-07-16): accidentally removed 
    # 'log_HCl': ([(-14,-2), r'$\log\ \mathrm{HCl}$'], 'HCl_main_iso'), # DGP (2024-07-16): try this one
    
    'log_Na': ([(-14,-2), r'$\log\ \mathrm{Na}$'], 'Na_allard_high'),
    # 'log_K': ([(-14,-2), r'$\log\ \mathrm{K}$'], 'K_high'),
    'log_Ca': ([(-14,-2), r'$\log\ \mathrm{Ca}$'], 'Ca_high'), 
    'log_Ti': ([(-14,-2), r'$\log\ \mathrm{Ti}$'], 'Ti_high'), 
    'log_Mg': ([(-14,-2), r'$\log\ \mathrm{Mg}$'], 'Mg_high'),
    # 'log_Mn': ([(-14,-2), r'$\log\ \mathrm{Mn}$'], 'Mn'),
    'log_Fe': ([(-14,-2), r'$\log\ \mathrm{Fe}$'], 'Fe_high'),
    # 'log_Ni': ([(-14,-2), r'$\log\ \mathrm{Ni}$'], 'Ni_high'),
    # 'log_Si': ([(-14,-2), r'$\log\ \mathrm{Si}$'], 'Si_high'),
    # 'log_Mn': ([(-14,-2), r'$\log\ \mathrm{Mn}$'], 'Mn_high'),
    'log_Sc': ([(-14,-2), r'$\log\ \mathrm{Sc}$'], 'Sc_high'),
    # 'log_Co': ([(-14,-2), r'$\log\ \mathrm{Co}$'], 'Co_high'), # has 1 line at 2319 nm and that's it
    
    # 'log_Al': ([(-14,-2), r'$\log\ \mathrm{Al}$'], 'Al'),
    
    # 'log_FeH': ([(-14,-2), r'$\log\ \mathrm{FeH}$'], 'FeH_main_iso'),
    # 'log_CrH': ([(-14,-2), r'$\log\ \mathrm{CrH}$'], 'CrH_main_iso'),
    # 'log_TiH': ([(-14,-2), r'$\log\ \mathrm{TiH}$'], 'TiH_main_iso'),

    'log_OH': ([(-14,-2), r'$\log\ \mathrm{OH}$'], 'OH_MYTHOS_main_iso'),
    'log_CN': ([(-14,-2), r'$\log\ \mathrm{CN}$'], 'CN_high'),
    # 'log_13CN': ([(-14,-2), r'$\log\ \mathrm{^{13}CN}$'], 'CN_34_high'),
    # 'log_H2': ([(-12,-0.1), r'$\log\ \mathrm{H_2}$'], 'H2_main_iso'),
    
    # 'log_VO': ([(-14,-2), r'$\log\ \mathrm{VO}$'], 'VO_HyVO_main_iso'), # DGP (2024-07-16): 3.4 um bump?
    # 'log_TiO': ([(-14,-2), r'$\log\ \mathrm{TiO}$'], 'TiO_48_Exomol_McKemmish'),
    # 'log_SiO': ([(-14,-2), r'$\log\ \mathrm{SiO}$'], 'SiO_SiOUVenIR_main_iso'),
    # 'log_AlO': ([(-14,-2), r'$\log\ \mathrm{AlO}$'], 'AlO_main_iso'),
    # 'log_H2S': ([(-14,-2), r'$\log\ \mathrm{H_2S}$'], 'H2S_Sid_main_iso'),
}
print(f' --> {len(opacity_params)} opacity parameters')
# Define the priors of the parameters
free_params = {

    # Uncertainty scaling
    # 'log_a_G': [(-2,0.6), r'$\log\ a$'], 
    # 'log_a_G235': [(-2,0.6), r'$\log\ a_{G235}$'],
    # 'log_a_G395': [(-2,0.6), r'$\log\ a_{G395}$'],
    # 'log_l': [(-2,0.3), r'$\log\ l$'], 
    # 'beta_G' : [(1., 20.), r'$\beta$'], # (NEW 2024-06-11): manage underestimated errors without inflating the GP kernel

    # SPHINX
    'Teff': [(2100.0, 3000.0), r'$T_\mathrm{eff}$'],
    'log_g': [(4.5,5.5), r'$\log\ g$'],
    'Z': [(-0.25, 0.25), 'Z'],
    # 'C_O': [(0.3, 0.9), 'C/O'],
    'alpha_12CO': [(-4., 2.), r'$\alpha(^{12}$CO)'],
    'alpha_H2O': [(-4., 2.), r'$\alpha$(H2O)'],
    # 'alpha_H2S': [(-4., 2.), r'$\alpha(H2S)$'],
    'alpha_Na': [(-4., 2.), r'$\alpha(Na)$'],
    'alpha_Ca': [(-4., 2.), r'$\alpha(Ca)$'],
    'alpha_Ti': [(-4., 2.), r'$\alpha(Ti)$'],
    'alpha_Mg': [(-4., 2.), r'$\alpha(Mg)$'],
    'alpha_Fe': [(-4., 2.), r'$\alpha(Fe)$'],
    'alpha_OH': [(-4., 2.), r'$\alpha(OH)$'],
    # 'alpha_K': [(-4., 2.), r'$\alpha(K)$'],
    # 'alpha_Si': [(-4., 2.), r'$\alpha(Si)$'], 

    # Velocities
    'vsini': [(0.5,11.0), r'$v\ \sin\ i$'], 
    'rv': [(-40., 40.), r'$v_\mathrm{rad}$'],
    
    # 'resolution': [(60e3, 80e3), r'$R$'], # 
}
# free_params.update({k:v[0] for k,v in opacity_params.items()})
SPHINX_species = ['H2O', '12CO', 'CO2', 'CH4', 'NH3', 'H2S', 'PH3', 
                  'HCN', 'C2H2', 'TiO', 'VO', 'SiO', 'FeH', 'CaH',
                  'MgH', 'CrH', 'AlH', 'TiH', 'Na', 'K', 'Fe', 'Mg',
                  'Ca', 'C', 'Si', 'Ti', 'O', 'FeII', 'MgII', 'TiII', 'CaII', 'CII', 
                  'N2', 'AlO', 'SH', 'OH', 'NO', 'SO2']

isotopologues_dict = {'13CO': ['log_12CO/13CO', [(1., 3.), r'$\log\ \mathrm{^{12}CO/^{13}CO}$']],
                      'C18O': ['log_12CO/C18O', [(1.5, 4.), r'$\log\ \mathrm{C^{16}O/C^{18}O}$']],
                        'C17O': ['log_12CO/C17O', [(1.5, 4.), r'$\log\ \mathrm{C^{16}O/C^{17}O}$']],
                        'H2O_181': ['log_H2O/H2O_181', [(1.5, 4.), r'$\log\ \mathrm{H_2^{16}O/H_2^{18}O}$']],
}


                      
for log_k, v in opacity_params.items():
    k = log_k[4:]
    if k in SPHINX_species:
        pass
    elif k in isotopologues_dict.keys():
        # add isotope ratio as free parameter
        free_params[isotopologues_dict[k][0]] = isotopologues_dict[k][1]
    else:
        free_params[log_k] = v[0]
        

print(f' --> {free_params} free parameters')

# Constants to use if prior is not given
# distance in pc to parallax
parallax_mas = 16.88 # Gaia DR3
d_pc = 1e3 / parallax_mas # ~ 59.17 pc

# dlnT_dlnP = [free_params[key] for key in free_params.keys() if 'dlnT_dlnP' in key]
# log_P_knots = [-5, -3, -2.0, -1.25, -0.5, 0.25, 1.0, 2.0] # 8 knots
# N_PT_knots = len(log_P_knots) # PT knots = 8 (NEW 2024-05-08)
# assert len(dlnT_dlnP) == N_PT_knots, 'Number of knots does not match number of dlnT_dlnP parameters'
PT_interp_mode = 'linear'
PT_mode = 'SPHINX'

N_knots = 25 # spline knots (continuum fitting)

constant_params = {
    # General properties
    # 'R_p' : 1.0, 
    # 'parallax': parallax_mas, 
    'epsilon_limb': 0.20, 
    'C_O': 0.59,
    'resolution': 69e3, # R=69,000, equivalent to 4.35 km/s
    # 'log_g': 4.72, # +- 0.12 (M15)
    # 'vsini':1.,

    # PT profile
    # 'log_P_knots': [-6., -3., -1., 1., 2.], 
    # 'log_P_knots': log_P_knots,
    'N_knots': N_knots, # avoid using spline to fit the continuum
}

####################################################################################
#
####################################################################################
scale_flux = True
scale_flux_eps = 0.00 # no scaling, set to 0.05 for a 5% deviation even with scale_flux=False
scale_err  = True
apply_high_pass_filter = False

# cloud_mode = 'gray'
cloud_mode = None
cloud_species = None

# mask_lines = {'Ni': (2298.2, 2299.4)}
mask_lines = {} # FIXME: manage the Ni line and other missing opacity sources...
mask_lines = {'telluric_red': (2493.0, 2500.0)}

####################################################################################
# Chemistry parameters
####################################################################################

#chem_mode  = 'free'
chem_mode  = 'SPHINX'
if chem_mode == 'SPHINX':
    assert PT_mode == 'SPHINX', 'SPHINX mode requires SPHINX PT mode'
    assert config_data['spirou']['n_atm_layers'] == 40, 'SPHINX mode requires 40 atm layers'
sphinx_grid_cache = True
    
# Rayleigh scattering and continuum opacities
rayleigh_species=['H2','He']
continuum_opacities=['H2-H2', 'H2-He', 'H-']
line_species =list(set([v[1] for _,v in opacity_params.items()]))
line_species_dict = {k[4:]: v[1] for k,v in opacity_params.items()}
print(f' --> line_species_dict = {line_species_dict}')

chem_kwargs = dict(species=[
            # 'H2H2',
            # 'H2He',
            # 'HMFF',
            'H2O', 
              'CO', 
            #   'TiO', 'VO',
            #   'SiO', 
            #   'FeH', 
            #   'CaH', 'MgH', 
            # 'H2S',
              'Na', 
            #   'K', 
              'Fe', 
              'Mg',
              'Ca',
              'Si', 
              'Ti',
            #   'AlO',
            #   'SH',
              'OH'],
            line_species_dict=line_species_dict,
)



# add H2 as line species, not a free parameter
# abundance of H2 calculated to sum(VMR) = 1
# line_species.append('H2_main_iso') # TODO: this?

# species_to_plot_VMR , species_to_plot_CCF = [], []
# species_to_plot_VMR = [k.split('_')[1] for k in opacity_params.keys() if 'log_' in k]
species_to_plot_VMR = ['H2O', 'OH', '12CO', '13CO', 'C18O', 'Na', 'Ca', 'Ti', 'Mg', 'Fe']
species_to_plot_CCF = []

####################################################################################
# Covariance parameters
####################################################################################

cov_mode = None
if 'log_a' in free_params.keys():
    cov_mode = 'GP'
    
cov_kwargs = dict(
    trunc_dist   = 1, # set to 3 for accuracy, 2 for speed
    scale_GP_amp = True, 
    # max_separation = 20, 

    # Prepare the wavelength separation and
    # average squared error arrays and keep 
    # in memory
    prepare_for_covariance = True
)
if free_params.get('log_l') is not None:
    cov_kwargs['max_separation'] =  cov_kwargs['trunc_dist']
    cov_kwargs['max_separation'] *= 10**free_params['log_l'][0][1]
    
####################################################################################
# PT parameters
####################################################################################


PT_kwargs = dict(
    conv_adiabat = True, 
    
    ln_L_penalty_order = 3, 
    PT_interp_mode = PT_interp_mode, 

    enforce_PT_corr = False, 
    # n_T_knots = N_PT_knots,
    sonora=dict(teff=2400, log_g=4.0),
    
)

####################################################################################
# Multinest parameters
####################################################################################
testing = False
const_efficiency_mode = True
sampling_efficiency = 0.05 if not testing else 0.10
evidence_tolerance = 0.5 if not testing else 1.0
n_live_points = 200
n_iter_before_update = n_live_points * 2
# n_iter_before_update = 1
# generate a .txt version of this file

if __name__ == '__main__':
    from retrieval_base.config import Config
    import pathlib
    
    conf = Config(path=pathlib.Path(__file__).parent.absolute(), target=None, run=run)
    conf.save_json(file_params, globals())