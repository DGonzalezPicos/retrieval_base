import numpy as np
import os

file_params = 'config_freechem.py'

####################################################################################
# Files and physical parameters
####################################################################################

prefix = 'freechem_1'
prefix = f'./retrieval_outputs/{prefix}/test_'

config_data = {
    'K2166': {
        'w_set': 'K2166', 'wave_range': (2060, 2480), 
        # 'w_set': 'K2166', 'wave_range': (1900, 2500), 

        'file_target': './data/J0856.dat', 
        'file_std': './data/iSco_std.dat', 
        'file_wave': './data/iSco_std.dat', 
        'file_skycalc_transm': f'./data/skycalc_transm_K2166.dat', 
        
        'file_molecfit_transm': './data/J0856_molecfit_transm.dat', 
        'file_std_molecfit_transm': './data/iSco_std_molecfit_transm.dat', 

        'filter_2MASS': '2MASS/2MASS.Ks', 
        'pwv': 5.0, 
        # adjust values below....!
        'ra': 134.057762, 'dec': -13.70612, 'mjd': 60008.03764053,
        'ra_std': 247.552759, 'dec_std': -25.11518, 'mjd_std': 60007.24715561, 

        'T_std': 0, 'log_g_std': 2.3, 'rv_std': 31.00, 'vsini_std': 280, 
        
        'slit': 'w_0.4', 'lbl_opacity_sampling': 3, 
        'tell_threshold': 0.7, 'sigma_clip_width': 8, 
    
        'log_P_range': (-5,2),
        'n_atm_layers': 100, 
        }, 
    }

magnitudes = {
    '2MASS/2MASS.Ks': (12.49, 0.02), # 2MASS
}

####################################################################################
# Model parameters
####################################################################################
# solar to jupiter radii
r_star = 0.29
r_jup = r_star / 0.10045
# Define the priors of the parameters
free_params = {

    # Uncertainty scaling
    #'log_a': [(-18,-14), r'$\log\ a_1$'], 
    'log_a': [(-1,0.5), r'$\log\ a_\mathrm{K}$'], 
    'log_l': [(-2,-0.8), r'$\log\ l_\mathrm{K}$'], 

    # General properties
    # R = 0.29 [R_sun]
    # convert to jupiter radii
    # R = 0.29 * 9.73116 = 2.82 [R_jup]
    'R_p': [(0.5, 5.0), r'$R_\mathrm{p}$'], 
    'log_g': [(3.0,5.5), r'$\log\ g$'], 
    'epsilon_limb': [(0.1,0.98), r'$\epsilon_\mathrm{limb}$'], 

    # Velocities
    'vsini': [(2.,20.), r'$v\ \sin\ i$'], 
    'rv': [(-22,22), r'$v_\mathrm{rad}$'], 
    
    # Chemistry
    'log_12CO': [(-12,-2), r'$\log\ \mathrm{^{12}CO}$'], 
    'log_13CO': [(-12,-2), r'$\log\ \mathrm{^{13}CO}$'], 
    'log_C18O': [(-12,-2), r'$\log\ \mathrm{C^{18}O}$'], 
    
    'log_H2O': [(-12,-2), r'$\log\ \mathrm{H_2O}$'], 
    
    'log_Na': [(-12,-2), r'$\log\ \mathrm{Na}$'],
    'log_Mg': [(-12,-2), r'$\log\ \mathrm{Mg}$'],
    'log_K': [(-12,-2), r'$\log\ \mathrm{K}$'],
    'log_Ca':[(-12,-2), r'$\log\ \mathrm{Ca}$'],
    'log_Ti':[(-12,-2), r'$\log\ \mathrm{Ti}$'],
    'log_Fe':[(-12,-2), r'$\log\ \mathrm{Fe}$'],
    
    'log_CN':[(-12,-2), r'$\log\ \mathrm{CN}$'],
    'log_HCN':[(-12,-2), r'$\log\ \mathrm{HCN}$'],
    'log_HF': [(-12,-2), r'$\log\ \mathrm{HF}$'], 
    'log_HCl':[(-12,-2), r'$\log\ \mathrm{HCl}$'],
    'log_H2S':[(-12,-2), r'$\log\ \mathrm{H_2S}$'],

    # PT profile
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

    'T_0': [(3000,9000), r'$T_0$'], 
}

# Constants to use if prior is not given
# distance in pc to parallax
d_pc = 53.8 # pc
parallax = 1/d_pc
parallax_mas = parallax * 1000

dlnT_dlnP = [free_params[key] for key in free_params.keys() if 'dlnT_dlnP' in key]
N_knots = len(dlnT_dlnP)
# log_P_knots = np.linspace(-5,2,N_knots).tolist()
# log_P_knots = [-5., 
#                -2.5, 
#             #    -1.75, 
#                -1.,
#             #    -0.5, 
#                0.5, 
#             #    1., 
#                2.]
# log_P_knots = [-5, -3, -2.0, -1.25, -0.5, 0.25, 1.0, 2.0] # 8 knots
# log_P_knots = [-5, -3, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.25, 2.0] # 10 knots
# log_P_knots=[-5., -3., -2, -1., 0., 1., 2.]
log_P_knots = [-5, -3, -2.0, -1.25, -0.5, 0.25, 1.0, 2.0] # 8 knots

assert len(log_P_knots) == N_knots, 'Number of knots does not match number of dlnT_dlnP parameters'
N_knots = len(log_P_knots)
# print(f'Number of PT knots: {len(dlnT_dlnP)}')
print(f'PT knots: {log_P_knots}')
PT_interp_mode = 'linear'
constant_params = {
    # General properties
    'parallax': parallax_mas, 
    # 'epsilon_limb': 0.65, 

    # PT profile
    'log_P_knots': log_P_knots, 
}

####################################################################################
#
####################################################################################

scale_flux = True
scale_err  = True
apply_high_pass_filter = False

# cloud_mode = 'gray'
cloud_mode = None
cloud_species = None

####################################################################################
# Chemistry parameters
####################################################################################

#chem_mode  = 'free'
#chem_mode  = 'SONORAchem'
chem_mode  = 'free'

chem_kwargs = dict()

# Rayleigh scattering and continuum opacities
rayleigh_species=['H2','He']
continuum_opacities=['H2-H2', 'H2-He', 'H-']

line_species = [
    'CO_high', 
    'CO_36_high', 
    # 'CO_28', 
    # 'CO_27', 

    'H2O_pokazatel_main_iso', 
    
    
    'Na_allard',
    'Mg',
    'K',
    'Ca',
    'Ti',
    'Fe',
    
    'CN_main_iso',
    'HCN_main_iso',
    'HF_main_iso', 
    'HCl_main_iso',
    'H2S_ExoMol_main_iso',
    
    ]
species_to_plot_VMR = [
    '12CO', '13CO', 'H2O', 'Na',
    'Mg', 'K', 'Ca', 'Ti', 'Fe',
    'CN', 'HCN', 'HF', 'HCl', 'H2S',
    ]
species_to_plot_CCF = [
    '12CO', '13CO', 'H2O', 'Na',
    'Mg', 'K', 'Ca', 'Ti', 'Fe',
    'CN', 'HCN', 'HF', 'HCl', 'H2S',
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
    cov_kwargs['max_separation'] = \
        cov_kwargs['trunc_dist'] * 10**free_params['log_l'][0][1]
if free_params.get('l') is not None:
    cov_kwargs['max_separation'] = \
        cov_kwargs['trunc_dist'] * free_params['l'][0][1]

if (free_params.get('log_l_K2166') is not None) and \
    (free_params.get('log_l_J1226') is not None):
    cov_kwargs['max_separation'] = cov_kwargs['trunc_dist'] * \
        10**max([free_params['log_l_K2166'][0][1], \
                 free_params['log_l_J1226'][0][1]])

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
n_live_points = 200
n_iter_before_update = 200
