import numpy as np
import os

file_params = 'config_jwst.py'

####################################################################################
# Files and physical parameters
####################################################################################

prefix = 'jwst_may_25'
prefix = f'./retrieval_outputs/{prefix}/test_'

config_data = {
    'G395H_F290LP': {
        'w_set': 'G395H_F290LP', 'wave_range': (4100, 5300), 
        # 'w_set': 'K2166', 'wave_range': (1900, 2500), 

        'file_target': './jwst/TWA28_g395h-f290lp.fits', 
        
        'lbl_opacity_sampling' : 20,
        'sigma_clip': 3,
        'sigma_clip_width': 50, 
    
        'log_P_range': (-5,2),
        'n_atm_layers': 30, 
        }, 
    }


####################################################################################
# Model parameters
####################################################################################

# Define the priors of the parameters
free_params = {

    # Uncertainty scaling
    'log_a': [(-2,0.6), r'$\log\ a$'], 
    'log_l': [(-1,0.2), r'$\log\ l$'], 

    # General properties
    # R = 0.29 [R_sun]
    # convert to jupiter radii
    # R = 0.29 * 9.73116 = 2.82 [R_jup]
    # 'R_p': [(1.0, 5.0), r'$R_\mathrm{p}$'], 
    'log_g': [(2.0,5.0), r'$\log\ g$'], 
    # 'epsilon_limb': [(0.1,0.98), r'$\epsilon_\mathrm{limb}$'], 

    # Velocities
    # 'vsini': [(2,30), r'$v\ \sin\ i$'], 
    'rv': [(-40,40), r'$v_\mathrm{rad}$'], 
    
    # Chemistry
    'log_12CO': [(-12,-2), r'$\log\ \mathrm{^{12}CO}$'], 
    'log_13CO': [(-12,-2), r'$\log\ \mathrm{^{13}CO}$'], 
    'log_C18O': [(-12,-2), r'$\log\ \mathrm{C^{18}O}$'], 
    'log_C17O': [(-12,-2), r'$\log\ \mathrm{C^{17}O}$'],
    
    'log_H2O': [(-12,-2), r'$\log\ \mathrm{H_2O}$'], 
    'log_H2O_181': [(-12,-2), r'$\log\ \mathrm{H_2^{18}O}$'],
    'log_CO2': [(-12,-2), r'$\log\ \mathrm{CO_2}$'],

   # PT profile
    'dlnT_dlnP_0': [(0.06,0.32), r'$\nabla_{T,0}$'], # 100 bar
    'dlnT_dlnP_1': [(0.06,0.22), r'$\nabla_{T,1}$'],  # 10 bar
    'dlnT_dlnP_2': [(0.06,0.32), r'$\nabla_{T,2}$'],  # 1 bar
    'dlnT_dlnP_3': [(0.06,0.32), r'$\nabla_{T,3}$'],  # 0.1 bar
    'dlnT_dlnP_4': [(0.06,0.32), r'$\nabla_{T,4}$'],  # 10 mbar
    'dlnT_dlnP_5': [(0.02,0.24), r'$\nabla_{T,5}$'],  # 10 mbar
    'dlnT_dlnP_6': [(0.00,0.22), r'$\nabla_{T,6}$'],  # 10 mbar
    'dlnT_dlnP_7': [(0.00,0.22), r'$\nabla_{T,7}$'],  # 10 mbar
    # 'dlnT_dlnP_8': [(0.04,0.22), r'$\nabla_{T,8}$'],  # 10 mbar
    # 'dlnT_dlnP_9': [(0.00,0.22), r'$\nabla_{T,9}$'],  # 10 mbar
    # 'dlnT_dlnP_10': [(0.00,0.34), r'$\nabla_{T,10}$'],  # 10 mbar
    # 'dlnT_dlnP_11': [(0.00,0.34), r'$\nabla_{T,11}$'],  # 10 mbar

    'dlog_P':[(-0.8,0.8), r'$\Delta\log\ P$'],
    'T_0': [(3000,9000), r'$T_0$'], 
    # 'f_slope': [(-0.1, 0.1), r'$f_\mathrm{slope}$'],
    'res': [(1500, 5000), r'$\mathrm{R}$'], # instrumental spectral resolution
}
# Constants to use if prior is not given
# distance in pc to parallax
d_pc = 59.2 # pc
parallax = 1/d_pc
parallax_mas = parallax * 1000

dlnT_dlnP = [free_params[key] for key in free_params.keys() if 'dlnT_dlnP' in key]
log_P_knots = [-5, -3, -2.0, -1.25, -0.5, 0.25, 1.0, 2.0] # 8 knots
N_knots = len(log_P_knots) # PT knots = 8 (NEW 2024-05-08)
assert len(dlnT_dlnP) == N_knots, 'Number of knots does not match number of dlnT_dlnP parameters'
PT_interp_mode = 'linear'

constant_params = {
    # General properties
    'R_p' : 2.8, 
    'parallax': parallax_mas, 
    'epsilon_limb': 0.65, 
    # 'log_g': 4.0,
    'vsini':1.,

    # PT profile
    # 'log_P_knots': [-6., -3., -1., 1., 2.], 
    'log_P_knots': log_P_knots,
    'N_knots': 1, # avoid using spline to fit the continuum
}

####################################################################################
#
####################################################################################
# N_knots = 5
scale_flux = False
scale_err  = True
apply_high_pass_filter = False

# cloud_mode = 'gray'
cloud_mode = None
cloud_species = None

mask_lines = {} 

####################################################################################
# Chemistry parameters
####################################################################################

#chem_mode  = 'free'
chem_mode  = 'free'

chem_kwargs = dict()

# Rayleigh scattering and continuum opacities
rayleigh_species=['H2','He']
continuum_opacities=['H2-H2', 'H2-He', 'H-']

line_species = [
    'CO_high', 
    'CO_36_high', 
    'CO_28', 
    'CO_27', 

    'H2O_pokazatel_main_iso', 
    'H2O_181_HotWat78',
    'CO2_main_iso',
    ]
species_to_plot_VMR = [
    '12CO', '13CO', 'H2O',
    'CO2'
    ]
species_to_plot_CCF = [
    '12CO', '13CO', 'H2O',
    'CO2',
    'H2O_181', 
    'C18O', 
    'C17O'
    ]

####################################################################################
# Covariance parameters
####################################################################################

cov_mode = 'GP'

cov_kwargs = dict(
    trunc_dist   = 3, 
    scale_GP_amp = True, 
    max_separation = 20, 

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

PT_mode = 'free_gradient'

PT_kwargs = dict(
    conv_adiabat = True, 

    ln_L_penalty_order = 3, 
    PT_interp_mode = PT_interp_mode, 

    enforce_PT_corr = False, 
    n_T_knots = N_knots,
)

####################################################################################
# Multinest parameters
####################################################################################

const_efficiency_mode = True
sampling_efficiency = 0.05
evidence_tolerance = 0.5
n_live_points = 100
# n_iter_before_update = n_live_points * 2
n_iter_before_update = 1
# generate a .txt version of this file

if __name__ == '__main__':
    from retrieval_base.config import Config
    import pathlib
    
    conf = Config(path=pathlib.Path(__file__).parent.absolute(), target=None, run=run)
    conf.save_json(file_params, globals())