import numpy as np
import os
import pathlib

file_params = 'config_freechem.py'

####################################################################################
# Files and physical parameters
####################################################################################

run = 'final_noGP'
prefix = f'./retrieval_outputs/{run}/test_'

config_data = {
    'K2166': {
        # 'w_set': 'K2166', 'wave_range': (2060, 2480), 
        # 'w_set': 'K2166', 'wave_range': (1900, 2500), 
        'w_set': 'K2166', 'wave_range': (1985, 2480), 


        'file_target': './data/J0856.dat', 
        # 'file_std': './data/iSco_std.dat', 
        'file_wave': './data/J0856_molecfit_transm.dat', 
        'file_skycalc_transm': None, 
        
        'file_molecfit_transm': './data/J0856_molecfit_transm.dat', 
        # 'file_std_molecfit_transm': './data/iSco_std_molecfit_transm.dat', 

        'filter_2MASS': '2MASS/2MASS.Ks', 
        # adjust values below....!
        'ra': 134.057762, 'dec': -13.70612, 'mjd': 60008.03764053,
        # 'ra_std': 247.552759, 'dec_std': -25.11518, 'mjd_std': 60007.24715561, 

        'T_std': 17_000, # i Sco = B3V
        'slit': 'w_0.4', 
        'lbl_opacity_sampling': 2, 
        'tell_threshold': 0.65,
        'tell_grow': 11,
        'sigma_clip_width': 12, 
    
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

# Define the priors of the parameters
free_params = {

    # Uncertainty scaling
    #'log_a': [(-18,-14), r'$\log\ a_1$'], 
    # 'log_a': [(-1,0.5), r'$\log\ a$'], 
    # 'log_l': [(-2,-0.8), r'$\log\ l$'], 
    
     # veiling power law
    'alpha': [(0.0, 2.), r'$\alpha$'],
    'beta': [(0.0, 3.0), r'$\beta$'],

    # General properties
    # R = 0.29 [R_sun]
    # convert to jupiter radii
    # R = 0.29 * 9.73116 = 2.82 [R_jup]
    # 'R_p': [(1.0, 10.0), r'$R_\mathrm{p}$'], 
    'log_g': [(2.0,5.5), r'$\log\ g$'], 
    'epsilon_limb': [(0.1,0.98), r'$\epsilon_\mathrm{limb}$'], 

    # Velocities
    'vsini': [(2,30), r'$v\ \sin\ i$'], 
    'rv': [(-40,40), r'$v_\mathrm{rad}$'], 
    
    # Chemistry
    'log_12CO': [(-12,-2), r'$\log\ \mathrm{^{12}CO}$'], 
    'log_13CO': [(-12,-2), r'$\log\ \mathrm{^{13}CO}$'], 
    # 'log_C18O': [(-12,-2), r'$\log\ \mathrm{C^{18}O}$'], 
    
    'log_H2O': [(-12,-2), r'$\log\ \mathrm{H_2O}$'], 
    'log_H2O_181': [(-12,-2), r'$\log\ \mathrm{H_2^{18}O}$'],
    'log_HF': [(-12,-2), r'$\log\ \mathrm{HF}$'], 

    'log_Na': [(-12,-2), r'$\log\ \mathrm{Na}$'],
    'log_Ca':[(-12,-2), r'$\log\ \mathrm{Ca}$'],
    'log_Ti':[(-12,-2), r'$\log\ \mathrm{Ti}$'],
    
    # 'log_Mg': [(-12,-2), r'$\log\ \mathrm{Mg}$'],
    # 'log_K': [(-12,-2), r'$\log\ \mathrm{K}$'],
    # 'log_Fe':[(-12,-2), r'$\log\ \mathrm{Fe}$'],
    
    # 'log_CN':[(-12,-2), r'$\log\ \mathrm{CN}$'],
    # 'log_HCN':[(-12,-2), r'$\log\ \mathrm{HCN}$'],
    # 'log_HF': [(-12,-2), r'$\log\ \mathrm{HF}$'], 
    # 'log_HCl':[(-12,-2), r'$\log\ \mathrm{HCl}$'],
    # 'log_H2S':[(-12,-2), r'$\log\ \mathrm{H_2S}$'],

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
}

# Constants to use if prior is not given
# distance in pc to parallax
d_pc = 53.8 # pc
parallax = 1/d_pc
parallax_mas = parallax * 1000

dlnT_dlnP = [free_params[key] for key in free_params.keys() if 'dlnT_dlnP' in key]
# log_P_knots = [-5, -3, -2.0, -1.625, -1.25, -0.875, -0.5, -0.125, 0.25, 0.625, 1.0, 2.0]
log_P_knots = [-5, -3, -2.0, -1.25, -0.5, 0.25, 1.0, 2.0] # 8 knots

N_knots = len(log_P_knots) # PT knots = 12 (NEW 2024-05-10)
assert len(dlnT_dlnP) == N_knots, 'Number of knots does not match number of dlnT_dlnP parameters'
PT_interp_mode = 'linear'
constant_params = {
    # General properties
    'parallax': parallax_mas, 
    # 'epsilon_limb': 0.65, 
    # 'R_p': 2.6, # from previous runs... does not matter because of flux scaling

    # PT profile
    # 'log_P_knots': np.linspace(-5,2,N_knots), 
     'log_P_knots': log_P_knots,
}

####################################################################################
#
####################################################################################

scale_flux = True
scale_err  = True
apply_high_pass_filter = False
normalize = True # normalize the spectrum per order (new 2024-05-07)
N_spline_knots = 1
N_veiling = 0

# cloud_mode = 'gray'
cloud_mode = None
cloud_species = None

# mask_lines = {'br_gamma': (2163, 2169)}
mask_lines = {}


####################################################################################
# Chemistry parameters
####################################################################################

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
    'H2O_181_HotWat78',

    # 'H2O_181',
    'HF_main_iso', 


    'Na_allard',
    'Ca',
    'Ti',
    # 'Mg',
    # 'K',
    # 'Fe',
    
    # 'CN_main_iso',
    # 'HCN_main_iso',
    # 'HCl_main_iso',
    # 'H2S_ExoMol_main_iso',
    
    ]
species_to_plot_VMR = [
    '12CO', '13CO', 'H2O', 'HF',
    'Na','Ca', 'Ti', 
    ]
species_to_plot_CCF = [
    '12CO', '13CO', 'H2O', 
    'HF',
    'Na', 'Ca', 'Ti',
    ]

####################################################################################
# Covariance parameters
####################################################################################

# cov_mode = 'GP'
cov_mode = None

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
    n_T_knots = len(constant_params['log_P_knots']), 
)

####################################################################################
# Multinest parameters
####################################################################################

const_efficiency_mode = True
sampling_efficiency = 0.05
evidence_tolerance = 0.5
n_live_points = 400
n_iter_before_update = int(n_live_points*3)
    
if __name__ == '__main__':
    from retrieval_base.config import Config
    import pathlib
    
    conf = Config(path=pathlib.Path(__file__).parent.absolute(), target=None, run=run)
    conf.save_json(file_params, globals())