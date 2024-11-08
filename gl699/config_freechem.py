import numpy as np
import os
from retrieval_base.auxiliary_functions import get_path
file_params = 'config_freechem.py'

####################################################################################
# Files and physical parameters
####################################################################################

run = 'fc5'
prefix = f'./retrieval_outputs/{run}/test_'
copy_pRT_from = None

config_data = {
    'spirou': {
        'w_set': 'spirou',
        'orders': (0,1,2), # file only contains 3 orders

        'lbl_opacity_sampling' : 3, # (2024-11-08): 5 --> 3 for final results
        'slit': 'spirou',
        'sigma_clip': 5,
        'sigma_clip_width': 11, 
        'Nedge': 30, # DGP (2024-10-07): 50 --> 30 for SPIRou
        'tell_threshold': 0.55,
        'tell_n_grow': 10,
        'emission_line_threshold': 1.3,
        
        'log_P_range': (-5,2),
        'n_atm_layers': 60, # DGP (2024-10-10): 40 --> 60 for SPIRou
        'T_cutoff': (1800.0, 5000.0),
        'P_cutoff': (1e-3, 1e2),
        
        'file_target':'data/spec_orders_46_47_48_mad.npy' # sep 28, new data with proper normalization and MAD error estimate
        }, 
    }

# priors for the radial velocity parameter, adjust to the expected RV of the target to avoid weird results
rv_min = -130.1
rv_max= -90.1

####################################################################################
# Model parameters
####################################################################################
opacity_params = {
    'log_12CO': ([(-14,-2), r'$\log\ \mathrm{^{12}CO}$'], 'CO_high_Sam'),
    'log_13CO': ([(-14,-2), r'$\log\ \mathrm{^{13}CO}$'], 'CO_36_high_Sam'),
    'log_C18O': ([(-14,-2), r'$\log\ \mathrm{C^{18}O}$'], 'CO_28_high_Sam'),
    'log_C17O': ([(-14,-2), r'$\log\ \mathrm{C^{17}O}$'], 'CO_27_high_Sam'),
        
    'log_H2O': ([(-14,-2), r'$\log\ \mathrm{H_2O}$'], 'H2O_pokazatel_main_iso'),
    'log_H2O_181': ([(-14,-2), r'$\log\ \mathrm{H_2^{18}O}$'], 'H2O_181_HotWat78'),
    
    'log_HF': ([(-14,-2), r'$\log\ \mathrm{HF}$'], 'HF_high'),
    'log_Na': ([(-14,-2), r'$\log\ \mathrm{Na}$'], 'Na_allard_high'),
    'log_Ca': ([(-14,-2), r'$\log\ \mathrm{Ca}$'], 'Ca_high'), 
    'log_Ti': ([(-14,-2), r'$\log\ \mathrm{Ti}$'], 'Ti_high'), 
    'log_Mg': ([(-14,-2), r'$\log\ \mathrm{Mg}$'], 'Mg_high'),
    'log_Fe': ([(-14,-2), r'$\log\ \mathrm{Fe}$'], 'Fe_high'),
    'log_Sc': ([(-14,-2), r'$\log\ \mathrm{Sc}$'], 'Sc_high'),
    'log_OH': ([(-14,-2), r'$\log\ \mathrm{OH}$'], 'OH_MYTHOS_main_iso'),
    'log_CN': ([(-14,-2), r'$\log\ \mathrm{CN}$'], 'CN_high'),
}
ignore_opacity_params = []
if len(ignore_opacity_params) > 0:
    opacity_params = {k:v for k,v in opacity_params.items() if k not in ignore_opacity_params}
print(f' --> {len(opacity_params)} opacity parameters')
# Define the priors of the parameters
free_params = {
    'log_g': [(4.5,5.5), r'$\log\ g$'],

    'alpha_12CO': [(-4., 2.), r'$\alpha(^{12}$CO)'],
    'alpha_H2O': [(-4., 2.), r'$\alpha$(H2O)'],
    'alpha_Na': [(-4., 2.), r'$\alpha(Na)$'],
    'alpha_Ca': [(-4., 2.), r'$\alpha(Ca)$'],
    'alpha_Ti': [(-4., 2.), r'$\alpha(Ti)$'],
    'alpha_Mg': [(-4., 2.), r'$\alpha(Mg)$'],
    'alpha_Fe': [(-4., 2.), r'$\alpha(Fe)$'],
    'alpha_OH': [(-4., 2.), r'$\alpha(OH)$'],
    'alpha_HF': [(-4., 2.), r'$\alpha(HF)$'],
    'alpha_CN': [(-4., 2.), r'$\alpha(CN)$'],
    
    

    # Velocities
    'vsini': [(0.5, 11.0), r'$v\ \sin\ i$'], 
    'rv': [(float(rv_min), float(rv_max)), r'$v_\mathrm{rad}$'],
        
    'T_0': [(4e3,16e3), r'$T_0$'], 
    'log_P_RCE': [(-2.0,1.2), r'$\log\ P_\mathrm{RCE}$'],
    'dlog_P_1' : [(0.2, 1.6), r'$\Delta\log\ P_1$'], 
    'dlog_P_3' : [(0.2, 1.6), r'$\Delta\log\ P_3$'],
    'dlnT_dlnP_0':   [(0.04, 0.44), r'$\nabla_{T,0}$'],
    'dlnT_dlnP_1':   [(0.04, 0.44), r'$\nabla_{T,1}$'],
    'dlnT_dlnP_RCE': [(0.04, 0.44), r'$\nabla_{T,RCE}$'],
    'dlnT_dlnP_2':   [(0.04, 0.44), r'$\nabla_{T,2}$'],
    'dlnT_dlnP_3':   [(0.00, 0.32), r'$\nabla_{T,3}$'],
    'dlnT_dlnP_4':   [(0.00, 0.32), r'$\nabla_{T,4}$'],
    'dlnT_dlnP_5':   [(0.00, 0.32), r'$\nabla_{T,5}$'], # new points
}
fc_species_dict = species_to_formula = {
    'H2': 'H2',
    'He': 'He',
    'e-': 'e-',
    'H2O': 'H2O1',
    '12CO': 'C1O1',
    'Na': 'Na',
    'K': 'K',
    'Fe': 'Fe',
    'Mg': 'Mg',
    'Ca': 'Ca',
    'Si': 'Si',
    'Ti': 'Ti',
    # 'O': 'O1',
    'OH': 'H1O1',
    'CN': 'C1N1',
    'HF': 'F1H1',
}
fc_species = list(fc_species_dict.keys()) # available species in chemistry table

isotopologues_dict = {
                        '13CO': ['log_12CO/13CO', [(1., 3.), r'$\log\ \mathrm{^{12}CO/^{13}CO}$']],
                        'C18O': ['log_12CO/C18O', [(1.5, 4.), r'$\log\ \mathrm{C^{16}O/C^{18}O}$']],
                        'C17O': ['log_12CO/C17O', [(1.5, 4.), r'$\log\ \mathrm{C^{16}O/C^{17}O}$']],
                        'H2O_181': ['log_H2O/H2O_181', [(1.5, 4.), r'$\log\ \mathrm{H_2^{16}O/H_2^{18}O}$']],
}


                      
for log_k, v in opacity_params.items():
    k = log_k[4:]
    if k in fc_species:
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

# assert len(dlnT_dlnP) == N_PT_knots, 'Number of knots does not match number of dlnT_dlnP parameters'
PT_interp_mode = 'linear'
PT_mode = 'RCE'

N_knots = 25 # spline knots (continuum fitting)

# pressure = np.concatenate([np.arange(-5.0, -3.0, 0.50), np.arange(-3.0, 1.0, 0.05), np.arange(1.0, 2.00+0.25, 0.25)]) # size 89
# pressure = list(10.0**np.concatenate([np.arange(-5.0, -3.0, 0.50), np.arange(-3.0, 1.0, 0.25/4), np.arange(1.0, 2.00+0.25, 0.25)])) # size 73


constant_params = {
    # General properties
    'epsilon_limb': 0.20, 
    # 'C_O': 0.59,
    'resolution': 69e3, # R=69,000, equivalent to 4.35 km/s
    # 'log_g': 4.72, # +- 0.12 (M15)

    'N_knots': N_knots, # avoid using spline to fit the continuum
}

####################################################################################
#
####################################################################################
scale_flux = True
scale_flux_eps = 0.00 # no scaling, set to 0.05 for a 5% deviation even with scale_flux=False
scale_err  = True
apply_high_pass_filter = False

cloud_mode = None
cloud_species = None

mask_lines = {'telluric_red': (2493.0, 2500.0)}

####################################################################################
# Chemistry parameters
####################################################################################

#chem_mode  = 'free'
# chem_mode  = 'SPHINX'
chem_mode = 'fastchem'

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

chem_kwargs = dict(
            fastchem_grid_file = '../data/fastchem_grid.h5',
            line_species_dict=line_species_dict,
)



species_to_plot_VMR = ['H2O', 'OH', '12CO', '13CO', 'C18O', 'Na', 'Ca', 'Ti', 'Mg', 'Fe', 'HF', 'CN']
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
n_live_points = 400 if not testing else 200
n_iter_before_update = n_live_points * 2
# n_iter_before_update = 1
# generate a .txt version of this file

if __name__ == '__main__':
    from retrieval_base.config import Config
    import pathlib
    
    conf = Config(path=pathlib.Path(__file__).parent.absolute(), target=None, run=run)
    conf.save_json(file_params, globals())