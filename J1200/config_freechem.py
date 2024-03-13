import numpy as np
import os

file_params = 'config_freechem.py'

####################################################################################
# Files and physical parameters
####################################################################################

prefix = 'freechem_30'
# prefix = 'freechem_15' # best-fit for J1200
prefix = f'./retrieval_outputs/{prefix}/test_'

config_data = {
    'K2166': {
        # 'w_set': 'K2166', 'wave_range': (2060, 2480), 
        # 'w_set': 'K2166', 'wave_range': (1900, 2500), 
        'w_set': 'K2166', 'wave_range': (1985, 2480), 


        'file_target': './data/J1200.dat', 
        # 'file_std': './data/iSco_std.dat', 
        'file_wave': './data/J1200_molecfit_transm.dat', 
        'file_skycalc_transm': None, 
        
        'file_molecfit_transm': './data/J1200_molecfit_transm.dat', 
        'file_std_molecfit_transm': None, 
        'filter_2MASS': '2MASS/2MASS.Ks', 
        
        # adjust values below....!
        'ra': 180.153309, 'dec': -78.75272, 'mjd': 60008.15139086,
        'T_std': 17_000, # i Sco = B3V

        'slit': 'w_0.4', 
        'lbl_opacity_sampling': 3, # set to 5 for accuracy, 10 for speed
        'tell_threshold': 0.60,
        'tell_grow': 21,
        'sigma_clip_width': 12, 
    
        'log_P_range': (-5,2), 
        'n_atm_layers': 50, # set to 50 for accuracy, 25 for speed
        }, 
    }

magnitudes = {
    '2MASS/2MASS.Ks': (11.60, 0.02), # Cutri et al. 2003
}

####################################################################################
# Model parameters
####################################################################################

# Define the priors of the parameters
free_params = {

    # Uncertainty scaling
    #'log_a': [(-18,-14), r'$\log\ a_1$'], 
    'log_a': [(-1,0.5), r'$\log\ a$'], 
    'log_l': [(-2,-0.8), r'$\log\ l$'], 

    # General properties
    # R = 0.29 [R_sun]
    # convert to jupiter radii
    # R = 0.29 * 9.73116 = 2.82 [R_jup]
    'R_p': [(1.0, 10.0), r'$R_\mathrm{p}$'], 
    'log_g': [(2.5,5.5), r'$\log\ g$'], 
    'epsilon_limb': [(0.1,0.98), r'$\epsilon_\mathrm{limb}$'], 

    # Velocities
    'vsini': [(2,30), r'$v\ \sin\ i$'], 
    'rv': [(-40,40), r'$v_\mathrm{rad}$'], 
    
    # Chemistry
    'log_12CO': [(-12,-2), r'$\log\ \mathrm{^{12}CO}$'], 
    'log_13CO': [(-12,-2), r'$\log\ \mathrm{^{13}CO}$'], 
    # 'log_C18O': [(-12,-2), r'$\log\ \mathrm{C^{18}O}$'], 
    
    'log_H2O': [(-12,-2), r'$\log\ \mathrm{H_2O}$'], 
    # 'log_H2O_181': [(-12,-2), r'$\log\ \mathrm{H_2^{18}O}$'],
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
    'dlnT_dlnP_0': [(0.08,0.32), r'$\nabla_{T,0}$'], # 100 bar
    'dlnT_dlnP_1': [(0.06,0.22), r'$\nabla_{T,1}$'],  # 10 bar
    'dlnT_dlnP_2': [(0.06,0.28), r'$\nabla_{T,2}$'],  # 1 bar
    'dlnT_dlnP_3': [(0.06,0.28), r'$\nabla_{T,3}$'],  # 0.1 bar
    'dlnT_dlnP_4': [(0.06,0.28), r'$\nabla_{T,4}$'],  # 10 mbar
    'dlnT_dlnP_5': [(0.02,0.22), r'$\nabla_{T,5}$'],  # 10 mbar
    'dlnT_dlnP_6': [(0.00,0.18), r'$\nabla_{T,6}$'],  # 10 mbar
    'dlnT_dlnP_7': [(0.00,0.18), r'$\nabla_{T,7}$'],  # 10 mbar

    
    # 'dlnT_dlnP_5': [(0.02,0.15), r'$\nabla_{T,5}$'],  # 1 mbar
    # 'dlnT_dlnP_6': [(-0.04,0.20), r'$\nabla_{T,6}$'],  # 0.01 mbar
    'T_0': [(3000,10000), r'$T_0$'], 
}

# Constants to use if prior is not given
# distance in pc to parallax
d_pc = 101.06 # pc
parallax = 1/d_pc
parallax_mas = parallax * 1000

dlnT_dlnP = [free_params[key] for key in free_params.keys() if 'dlnT_dlnP' in key]
N_knots = len(dlnT_dlnP)
# log_P_knots = np.linspace(-5,2,N_knots).tolist()
log_P_knots = [-5., -3., -1.75, -1.,-0.5, 0.25, 1., 2.]
assert len(log_P_knots) == N_knots, 'Number of knots does not match number of dlnT_dlnP parameters'
# print(f'Number of PT knots: {len(dlnT_dlnP)}')
# print(f'PT knots: {log_P_knots}')
PT_interp_mode = 'linear'
constant_params = {
    # General properties
    'parallax': parallax_mas, 
    # 'epsilon_limb': 0.65, 
    # 'R_p': 6.80, # from previous runs... does not matter because of flux scaling

    # PT profile
    'log_P_knots': log_P_knots, 
}

####################################################################################
#
####################################################################################

scale_flux = True
scale_err  = True
apply_high_pass_filter = False
N_spline_knots = 1

# cloud_mode = 'gray'
cloud_mode = None
cloud_species = None

mask_lines = {'br_gamma': (2162.8, 2169.6)}

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
    n_T_knots = len(constant_params['log_P_knots']), 
)

####################################################################################
# Multinest parameters
####################################################################################

const_efficiency_mode = True
sampling_efficiency = 0.05
evidence_tolerance = 0.5
n_live_points = 200
n_iter_before_update = int(n_live_points*30)

# generate a .txt version of this file

if __name__ == '__main__':
    # print all global variables
    save_attrs = ['config_data', 'magnitudes', 'free_params',
                'd_pc', 'parallax', 'parallax_mas',
                'constant_params', 'scale_flux', 'scale_err',
                'apply_high_pass_filter', 'cloud_mode', 'cloud_species', 
                'mask_lines', 'chem_mode', 'chem_kwargs', 'rayleigh_species', 
                'continuum_opacities', 'line_species',
                'cov_mode', 'cov_kwargs', 'PT_mode', 'PT_kwargs',
                'const_efficiency_mode', 'sampling_efficiency',
                'evidence_tolerance', 'n_live_points', 'n_iter_before_update']

    import pathlib
    import json

    # get path of this file
    path = pathlib.Path(__file__).parent.absolute()
    outpath = path / f'{prefix[2:]}data'
    outfile = outpath / file_params.replace('.py', '.txt')
    outfile.parent.mkdir(parents=True, exist_ok=True)

    with open(outfile, 'w') as file:
        file.write(json.dumps({key: globals()[key] for key in save_attrs}))
        
    file.close()
    print(f'Wrote {outfile}')
        
    # # # test loading the file with json
    # with open(outfile, 'r') as file:
    #     load_file = json.load(file)