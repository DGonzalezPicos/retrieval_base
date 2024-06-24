import numpy as np
import os

file_params = 'config_jwst.py'

####################################################################################
# Files and physical parameters
####################################################################################

run = 'jwst_KM_N1'
prefix = f'./retrieval_outputs/{run}/test_'

config_data = {
    'NIRSpec': {
        # 'w_set': 'G395H_F290LP', 'wave_range': (4100, 5300), 
        'w_set': 'NIRSpec',
        # 'wave_range': (1650, 3200), # g235h-f170lp
        'wave_range': (1650, 5300),
        # 'w_set': 'K2166', 'wave_range': (1900, 2500), 

        'file_target': './jwst/TWA28_g395h-f290lp.fits', 
        
        'lbl_opacity_sampling' : 20,
        'sigma_clip': 3,
        'sigma_clip_width': 50, 
    
        'log_P_range': (-5,2),
        'n_atm_layers': 35, 
        }, 
    }


####################################################################################
# Model parameters
####################################################################################

# Define the priors of the parameters
free_params = {

    # Uncertainty scaling
    # 'log_a_G': [(-2,0.6), r'$\log\ a$'], 
    'log_a_G235': [(-2,0.6), r'$\log\ a_{G235}$'],
    'log_a_G395': [(-2,0.6), r'$\log\ a_{G395}$'],
    'log_l': [(-2,-0.9), r'$\log\ l$'], 
    # 'beta_G' : [(1., 10.), r'$\beta$'], # (NEW 2024-06-11): manage underestimated errors without inflating the GP kernel

    # General properties
    # R = 0.29 [R_sun]
    # convert to jupiter radii
    # R = 0.29 * 9.73116 = 2.82 [R_jup]
    # 'R_p': [(1.0, 5.0), r'$R_\mathrm{p}$'], # use this for robust results
     'R_p': [(2.2, 3.5), r'$R_\mathrm{p}$'], # R_p ~ 2.82 R_jup
    'log_g': [(2.0,5.0), r'$\log\ g$'], 
    # 'epsilon_limb': [(0.1,0.98), r'$\epsilon_\mathrm{limb}$'], 
    
    # veiling parameters
    # 'log_r_0': [(-20, -14), r'$\log\ r_0$'], # veiling amplitude at wave=min(wave)
    # 'alpha': [(1.0, 20.0), r'$\alpha$'], # veiling power-law index, should be positive for dust emission
    'R_d': [(1.0, 100.0), r'$R_d [R_{Jup}]$'], # disk radius in R_jup
    # 'log_R_d' : [(-2, 4), r'$\log\ R_d$'], # disk radius in R_jup
    'T_d': [(300, 1000), r'$T_d$'], # disk temperature in K

    # Velocities
    # 'vsini': [(2,30), r'$v\ \sin\ i$'], 
    'rv': [(-20,20), r'$v_\mathrm{rad}$'], 
    
    # Chemistry
    'log_12CO': [(-12,-2), r'$\log\ \mathrm{^{12}CO}$'], 
    'log_13CO': [(-12,-2), r'$\log\ \mathrm{^{13}CO}$'], 
    # 'log_C18O': [(-12,-2), r'$\log\ \mathrm{C^{18}O}$'], 
    # 'log_C17O': [(-12,-2), r'$\log\ \mathrm{C^{17}O}$'],
    
    'log_H2O': [(-12,-2), r'$\log\ \mathrm{H_2O}$'], 
    # 'log_H2O_181': [(-12,-2), r'$\log\ \mathrm{H_2^{18}O}$'],
    'log_CO2': [(-12,-2), r'$\log\ \mathrm{CO_2}$'], # (NEW 2024-06-11): try to detect CO2 in KLM bands
    'log_Na': [(-12,-2), r'$\log\ \mathrm{Na}$'],
    'log_K': [(-12,-2), r'$\log\ \mathrm{K}$'], # (NEW 2024-06-11): try to detect K in KLM bands
    'log_Ca': [(-12,-2), r'$\log\ \mathrm{Ca}$'],
    'log_Ti': [(-12,-2), r'$\log\ \mathrm{Ti}$'],
    'log_HF': [(-12,-2), r'$\log\ \mathrm{HF}$'],
    'log_Mg': [(-12,-2), r'$\log\ \mathrm{Mg}$'],
    # 'log_Fe': [(-12,-2), r'$\log\ \mathrm{Fe}$'],
    'log_Al': [(-12,-2), r'$\log\ \mathrm{Al}$'],
    # 'log_Si': [(-12,-2), r'$\log\ \mathrm{Si}$'],
    'log_FeH': [(-12,-2), r'$\log\ \mathrm{FeH}$'],
    'log_H-' : [(-12,-6), r'$\log\ \mathrm{H^-}$'],

   'T_0': [(2000,8000), r'$T_0$'], 
    'log_P_RCE': [(-3,1), r'$\log\ P_\mathrm{RCE}$'],
    # 'dlog_P' : [(0.2, 1.6), r'$\Delta\log\ P$'],
    'dlog_P_1' : [(0.2, 1.6), r'$\Delta\log\ P_1$'], 
    'dlog_P_3' : [(0.2, 1.6), r'$\Delta\log\ P_3$'],
    'dlnT_dlnP_RCE': [(0.04, 0.34), r'$\nabla_{T,RCE}$'],
    'dlnT_dlnP_0':   [(0.04, 0.34), r'$\nabla_{T,0}$'],
    'dlnT_dlnP_1':   [(0.04, 0.34), r'$\nabla_{T,1}$'],
    'dlnT_dlnP_2':   [(0.04, 0.34), r'$\nabla_{T,2}$'],
    'dlnT_dlnP_3':   [(0.00, 0.34), r'$\nabla_{T,3}$'],
    'dlnT_dlnP_4':   [(0.00, 0.34), r'$\nabla_{T,4}$'],
    'dlnT_dlnP_5':   [(0.00, 0.34), r'$\nabla_{T,5}$'], # new points

    # 'f_slope': [(-0.1, 0.1), r'$f_\mathrm{slope}$'],
    'res_G235': [(1500, 4000), r'$\mathrm{R}_{G235}$'], # instrumental spectral resolution
    'res_G395': [(1500, 5000), r'$\mathrm{R}_{G395}$'], # instrumental spectral resolution
    # 'res_M': [(1500, 5000), r'$\mathrm{R}_M$'], # instrumental spectral resolution    
}
# Constants to use if prior is not given
# distance in pc to parallax
d_pc = 59.2 # pc
parallax = 1/d_pc
parallax_mas = parallax * 1000

# dlnT_dlnP = [free_params[key] for key in free_params.keys() if 'dlnT_dlnP' in key]
# log_P_knots = [-5, -3, -2.0, -1.25, -0.5, 0.25, 1.0, 2.0] # 8 knots
# N_PT_knots = len(log_P_knots) # PT knots = 8 (NEW 2024-05-08)
# assert len(dlnT_dlnP) == N_PT_knots, 'Number of knots does not match number of dlnT_dlnP parameters'
PT_interp_mode = 'linear'
PT_mode = 'RCE'

N_knots = 1 # spline knots (continuum fitting)

constant_params = {
    # General properties
    # 'R_p' : 1.0, 
    'parallax': parallax_mas, 
    'epsilon_limb': 0.5, 
    # 'log_g': 3.5,
    'vsini':1.,

    # PT profile
    # 'log_P_knots': [-6., -3., -1., 1., 2.], 
    # 'log_P_knots': log_P_knots,
    'N_knots': N_knots, # avoid using spline to fit the continuum
    # 'res_G235': 2800, # instrumental spectral resolution
    # 'res_G395': 3000, # instrumental spectral resolution
    # 'fit_radius': True,
}

####################################################################################
#
####################################################################################
scale_flux = False
scale_flux_eps = 0.00 # no scaling, set to 0.05 for a 5% deviation even with scale_flux=False
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
    # 'CO_28', 
    # 'CO_27', 

    'H2O_pokazatel_main_iso', 
    # 'H2O_181_HotWat78',
    'CO2_main_iso', # CO2 at ~ 4.1 um
    'Na_allard',
    'K', # (NEW 2024-06-11): try to detect K in KLM bands
    'Ca',
    'Ti',
    'Mg', # (NEW 2024-06-11): no Mg detected in KLM bands
    # 'Fe', # (NEW 2024-06-11): no Fe detected in KLM bands
    'Al',
    'HF_main_iso',
    # 'Si',
    'FeH_main_iso',
    ]
species_to_plot_VMR = [
    '12CO', '13CO', 'H2O',
    ]
species_to_plot_CCF = [
    '12CO', '13CO', 'H2O',
    ]

####################################################################################
# Covariance parameters
####################################################################################

cov_mode = 'GP'

cov_kwargs = dict(
    trunc_dist   = 0.7, # set to 3 for accuracy, 2 for speed
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

const_efficiency_mode = True
sampling_efficiency = 0.05
evidence_tolerance = 1.0
n_live_points = 100
n_iter_before_update = n_live_points * 2
# n_iter_before_update = 1
# generate a .txt version of this file

if __name__ == '__main__':
    from retrieval_base.config import Config
    import pathlib
    
    conf = Config(path=pathlib.Path(__file__).parent.absolute(), target=None, run=run)
    conf.save_json(file_params, globals())