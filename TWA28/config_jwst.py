import numpy as np
import os

file_params = 'config_jwst.py'

####################################################################################
# Files and physical parameters
####################################################################################

prefix = 'jwst_2'
prefix = f'./retrieval_outputs/{prefix}/test_'

config_data = {
    'G395H_F290LP': {
        'w_set': 'G395H_F290LP', 'wave_range': (4100, 5300), 
        # 'w_set': 'K2166', 'wave_range': (1900, 2500), 

        'file_target': './jwst/TWA28_g395h-f290lp.fits', 
        
        'lbl_opacity_sampling' : 20,
        'sigma_clip': 3,
        'sigma_clip_width': 50, 
    
        'log_P_range': (-6,2), 'n_atm_layers': 30, 
        }, 
    }


####################################################################################
# Model parameters
####################################################################################

# Define the priors of the parameters
free_params = {

    # Uncertainty scaling
    'log_a': [(-2,0.2), r'$\log\ a$'], 
    'log_l': [(-2,-0.1), r'$\log\ l$'], 

    # General properties
    # R = 0.29 [R_sun]
    # convert to jupiter radii
    # R = 0.29 * 9.73116 = 2.82 [R_jup]
    # 'R_p': [(1.0, 10.0), r'$R_\mathrm{p}$'], 
    'log_g': [(2.5,5.0), r'$\log\ g$'], 
    # 'epsilon_limb': [(0.1,0.98), r'$\epsilon_\mathrm{limb}$'], 

    # Velocities
    # 'vsini': [(2,30), r'$v\ \sin\ i$'], 
    'rv': [(-40,40), r'$v_\mathrm{rad}$'], 
    
    # Chemistry
    'log_12CO': [(-8,-2), r'$\log\ \mathrm{^{12}CO}$'], 
    'log_13CO': [(-10,-2), r'$\log\ \mathrm{^{13}CO}$'], 
    # 'log_C18O': [(-12,-2), r'$\log\ \mathrm{C^{18}O}$'], 
    
    'log_H2O': [(-8,-2), r'$\log\ \mathrm{H_2O}$'], 
    # 'log_H2O_181': [(-12,-2), r'$\log\ \mathrm{H_2^{18}O}$'],

    # PT profile
    'dlnT_dlnP_0': [(0.01, 0.40), r'$\nabla_{T,0}$'], 
    'dlnT_dlnP_1': [(0.01,0.40), r'$\nabla_{T,1}$'], 
    'dlnT_dlnP_2': [(0.01,0.40), r'$\nabla_{T,2}$'], 
    # 'dlnT_dlnP_3': [(0.00,0.20), r'$\nabla_{T,3}$'], 
    # 'dlnT_dlnP_4': [(-0.05,0.15), r'$\nabla_{T,4}$'], 
    'dlnT_dlnP_3': [(-0.05,0.25), r'$\nabla_{T,3}$'], 
    'dlnT_dlnP_4': [(-0.05,0.25), r'$\nabla_{T,4}$'],
    'T_0': [(3000,10000), r'$T_0$'], 
    # 'f_slope': [(-0.1, 0.1), r'$f_\mathrm{slope}$'],
}
# Constants to use if prior is not given
# distance in pc to parallax
d_pc = 59.2 # pc
parallax = 1/d_pc
parallax_mas = parallax * 1000
constant_params = {
    # General properties
    'R_p' : 2.8, 
    'parallax': parallax_mas, 
    'epsilon_limb': 0.65, 
    # 'log_g': 4.0,
    'vsini':1.,

    # PT profile
    'log_P_knots': [-6., -3., -1., 1., 2.], 
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
    # 'H2O_181',
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
n_live_points = 100
n_iter_before_update = 50

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