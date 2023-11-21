import numpy as np
import os

file_params = 'config_fiducial.py'

####################################################################################
# Files and physical parameters
####################################################################################

prefix = 'fiducial_ret_1'
prefix = f'./retrieval_outputs/{prefix}/test_'

config_data = {
    'K2166': {
        'w_set': 'K2166', 'wave_range': (2060, 2480), 
        # 'w_set': 'K2166', 'wave_range': (1900, 2500), 

        'file_target': './data/DHTauA.dat', 
        'file_std': './data/chiTau_std.dat', 
        'file_wave': './data/chiTau_std.dat', 
        'file_skycalc_transm': f'./data/skycalc_transm_K2166.dat', 
        
        'file_molecfit_transm': './data/DHTauA_molecfit_transm.dat', 
        'file_std_molecfit_transm': './data/chiTau_std_molecfit_transm.dat', 

        'filter_2MASS': '2MASS/2MASS.Ks', 
        'pwv': 5.0, 
        # adjust values below....!
        'ra': 67.422516, 'dec':26.54998 , 'mjd': 59945.15461034,
        'ra_std': 247.552759, 'dec_std': -25.11518, 'mjd_std': 60007.24715561, 

        'T_std': 17_000, 'log_g_std': 2.3, 'rv_std': 31.00, 'vsini_std': 280, 
        
        'slit': 'w_0.4', 'lbl_opacity_sampling': 3, 
        'tell_threshold': 0.7, 'sigma_clip_width': 8, 
    
        'log_P_range': (-5,2), 'n_atm_layers': 30, 
        }, 
    }

magnitudes = {
    '2MASS/2MASS.Ks': (8.20, 0.02), # Cutri et al. 2003
}

####################################################################################
# Model parameters
####################################################################################
# solar to jupiter radii
r_star = 0.29
r_jup = r_star / 0.10045
# Define the priors of the parameters
free_params = {
    # Data resolution
    #'res': [(20000,200000), r'res'], 
    #'log_res_J1226': [(4,5.2), r'$\log\ R_\mathrm{J}$'], 

    # Uncertainty scaling
    #'log_a': [(-18,-14), r'$\log\ a_1$'], 
    'log_a': [(-1,0.4), r'$\log\ a_\mathrm{K}$'], 
    'log_l': [(-2,-0.8), r'$\log\ l_\mathrm{K}$'], 
    #'log_a_K2166': [(-1,0.4), r'$\log\ a_\mathrm{K}$'], 
    #'log_l_K2166': [(-2,-0.8), r'$\log\ l_\mathrm{K}$'], 

    # General properties
    # R = 0.29 [R_sun]
    # convert to jupiter radii
    # R = 0.29 * 9.73116 = 2.82 [R_jup]
    'R_p': [(1.0, 12.0), r'$R_\mathrm{p}$'], 
    'log_g': [(3.0,5.5), r'$\log\ g$'], 
    'epsilon_limb': [(0.1,0.98), r'$\epsilon_\mathrm{limb}$'], 

    # Velocities
    'vsini': [(2.,25.), r'$v\ \sin\ i$'], 
    'rv': [(-25,25), r'$v_\mathrm{rad}$'], 

    # Cloud properties
    # 'log_opa_base_gray': [(-10,5), r'$\log\ \kappa_{\mathrm{cl},0}$'], 
    # 'log_P_base_gray': [(-5,3), r'$\log\ P_{\mathrm{cl},0}$'], 
    # 'f_sed_gray': [(0,20), r'$f_\mathrm{sed}$'], 
    #'cloud_slope': [(-10,10), r'$\xi_\mathrm{cl}$'], 

    # Chemistry
    'C/O': [(0.15,1), r'C/O'], 
    'Fe/H': [(-1,1), r'[Fe/H]'], 
    # 'log_P_quench_CO_CH4': [(-5,3), r'$\log\ P_\mathrm{quench}(\mathrm{C})$'], 
    #'log_P_quench_N2_NH3': [(-5,2), r'$\log\ P_\mathrm{quench}(\mathrm{N})$'], 
    'log_C13_12_ratio': [(-4,-1), r'$\log\ \mathrm{^{13}C/^{12}C}$'], 
    'log_O18_16_ratio': [(-6, -1), r'$\log\ \mathrm{^{18}O/^{16}O}$'], 
    # 'log_O17_16_ratio': [(-10,0), r'$\log\ \mathrm{^{17}C/^{16}O}$'], 
    # 'log_HF': [(-12,-2), r'$\log\ \mathrm{HF}$'], 
    # 'log_HCl': [(-12,-2), r'$\log\ \mathrm{HCl}$'], 

    #'log_12CO': [(-12,-2), r'$\log\ \mathrm{^{12}CO}$'], 
    #'log_13CO': [(-12,-2), r'$\log\ \mathrm{^{13}CO}$'], 
    #'log_C18O': [(-12,-2), r'$\log\ \mathrm{C^{18}O}$'], 
    #'log_C17O': [(-12,-2), r'$\log\ \mathrm{C^{17}O}$'], 

    #'log_H2O': [(-12,-2), r'$\log\ \mathrm{H_2O}$'], 
    #'log_CH4': [(-12,-2), r'$\log\ \mathrm{CH_4}$'], 
    #'log_NH3': [(-12,-2), r'$\log\ \mathrm{NH_3}$'], 
    #'log_H2S': [(-12,-2), r'$\log\ \mathrm{H_2S}$'], 
    #'log_HCN': [(-12,-2), r'$\log\ \mathrm{HCN}$'], 
    #'log_CO2': [(-12,-2), r'$\log\ \mathrm{CO_2}$'], 

    # PT profile
    'dlnT_dlnP_0': [(0.10, 0.40), r'$\nabla_{T,0}$'], 
    'dlnT_dlnP_1': [(0.08,0.22), r'$\nabla_{T,1}$'], 
    'dlnT_dlnP_2': [(0.05,0.30), r'$\nabla_{T,2}$'], 
    'dlnT_dlnP_3': [(0.00,0.20), r'$\nabla_{T,3}$'], 
    'dlnT_dlnP_4': [(-0.05,0.15), r'$\nabla_{T,4}$'], 
    'T_0': [(2000,12000), r'$T_0$'], 
}

# Constants to use if prior is not given
# distance in pc to parallax
# d_pc = 101.06 # pc
# parallax = 1/d_pc
# parallax_mas = parallax * 1000
parallax_mas = 7.4936
constant_params = {
    # General properties
    'parallax': parallax_mas, 
    # 'epsilon_limb': 0.65, 

    # PT profile
    'log_P_knots': [-5., -3., -1., 1., 2.], 
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
chem_mode  = 'eqchem'

chem_kwargs = dict()

# Rayleigh scattering and continuum opacities
rayleigh_species=['H2','He']
continuum_opacities=['H2-H2', 'H2-He', 'H-']

line_species = [
    'CO_high', 
    'CO_36_high', 
    'CO_28', 
    # 'CO_27', 

    'H2O_pokazatel_main_iso', 
    'Na_allard',
    # 'CH4_hargreaves_main_iso', 
    # 'NH3_coles_main_iso', 

    # 'H2S_ExoMol_main_iso', 
    # 'HF_main_iso', 
    # 'HCl_main_iso', 

    # 'HCN_main_iso', 
    # 'CO2_main_iso', 
    ]
species_to_plot_VMR = [
    '12CO', '13CO', 'H2O', 'Na'
    ]
species_to_plot_CCF = [
    '12CO', '13CO', 'H2O', 'Na'
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
    PT_interp_mode = 'quadratic', 

    enforce_PT_corr = False, 
    n_T_knots = 5, 
)

####################################################################################
# Multinest parameters
####################################################################################

const_efficiency_mode = True
sampling_efficiency = 0.05
evidence_tolerance = 0.5
n_live_points = 200
n_iter_before_update = 200
