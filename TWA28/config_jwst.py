import numpy as np
import os

file_params = 'config_jwst.py'

####################################################################################
# Files and physical parameters
####################################################################################

# run = 'ck_K_2'
# run = 'lbl12_KM_2'
lbl = 15
run = f'lbl{lbl}_G2G3_2'
prefix = f'./retrieval_outputs/{run}/test_'
grating = 'g235h+g395h'

config_data = {
    'NIRSpec': {
        # 'w_set': 'G395H_F290LP', 'wave_range': (4100, 5300), 
        'w_set': 'NIRSpec',
        # 'wave_range': (1650, 3200), # g235h-f170lp
        # 'wave_range': (1630, 5300), 
        # 'wave_range': (1630, 3250), 
        
        'lbl_opacity_sampling' : lbl,
        # 'lbl_opacity_sampling' : None,
        'sigma_clip': 3,
        'sigma_clip_width': 31, # (2024-07-16): 21 --> 31
        'Nedge': 34, # (2024-10-18): 20 --> 34
    
        'log_P_range': (-5,2),
        'n_atm_layers': 35, # (2024-10-12): update 35 --> 50
        'T_cutoff': (1200.0, 3600.0), # DGP (2024-10-14): new parameter
        'P_cutoff': (1e-4, 1e2), # DGP (2024-10-14): new parameter
        }, 
    }

# update wave_range
gratings_wave_range = {'g235h': (1630, 3200),
                       'g395h': (2800, 5300),
                       'g235h+g395h': (1630, 5300),
                       }
config_data['NIRSpec']['wave_range'] = gratings_wave_range[grating]



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
    # 'log_HDO': ([(-14,-2), r'$\log\ \mathrm{HDO}$'], 'HDO_voronin'),
    'log_HF': ([(-14,-2), r'$\log\ \mathrm{HF}$'], 'HF_high'), # DGP (2024-07-16): accidentally removed 
    'log_HCl': ([(-14,-2), r'$\log\ \mathrm{HCl}$'], 'HCl_main_iso'), # DGP (2024-07-16): try this one
    
    'log_CO2': ([(-14,-2), r'$\log\ \mathrm{CO_2}$'], 'CO2_main_iso'),
    # 'log_CN': ([(-14,-2), r'$\log\ \mathrm{CN}$'], 'CN_high'),
    
    'log_Na': ([(-14,-2), r'$\log\ \mathrm{Na}$'], 'Na_allard_high'),
    # 'log_K':  ([(-14,-2), r'$\log\ \mathrm{K}$'],  'K_high'),
    'log_Ca': ([(-14,-2), r'$\log\ \mathrm{Ca}$'], 'Ca_high'),
    # 'log_Ti': ([(-14,-2), r'$\log\ \mathrm{Ti}$'], 'Ti_high'),
    # 'log_Mg': ([(-14,-2), r'$\log\ \mathrm{Mg}$'], 'Mg_high'),
    # 'log_Mn': ([(-14,-2), r'$\log\ \mathrm{Mn}$'], 'Mn_high'),
    # 'log_Fe': ([(-14,-2), r'$\log\ \mathrm{Fe}$'], 'Fe'),
    # 'log_Al': ([(-14,-2), r'$\log\ \mathrm{Al}$'], 'Al_high'),
    
    'log_FeH': ([(-14,-2), r'$\log\ \mathrm{FeH}$'], 'FeH_main_iso'),
    # 'log_CrH': ([(-14,-2), r'$\log\ \mathrm{CrH}$'], 'CrH_main_iso'),
    'log_TiH': ([(-14,-2), r'$\log\ \mathrm{TiH}$'], 'TiH_main_iso'),
    # 'log_CaH': ([(-14,-2), r'$\log\ \mathrm{CaH}$'], 'CaH_XAB_main_iso'),
    'log_AlH': ([(-14,-2), r'$\log\ \mathrm{AlH}$'], 'AlH_AloHa_main_iso'),
    # 'log_MgH': ([(-14,-2), r'$\log\ \mathrm{MgH}$'], 'MgH_main_iso'),
    # 'log_NaH': ([(-14,-2), r'$\log\ \mathrm{NaH}$'], 'NaH_main_iso'), # DGP (2024-07-16)
    # 'log_ScH': ([(-14,-2), r'$\log\ \mathrm{ScH}$'], 'ScH_main_iso'), # DGP (2024-07-16): try

    'log_OH': ([(-14,-2), r'$\log\ \mathrm{OH}$'], 'OH_MYTHOS_main_iso'),
    # 'log_H2': ([(-12,-0.01), r'$\log\ \mathrm{H_2}$'], 'H2_main_iso'),
    
    'log_VO': ([(-14,-2), r'$\log\ \mathrm{VO}$'], 'VO_HyVO_main_iso'), # DGP (2024-07-16): 3.4 um bump?
    'log_TiO': ([(-14,-2), r'$\log\ \mathrm{TiO}$'], 'TiO_48_Exomol_McKemmish'),
    'log_SiO': ([(-14,-2), r'$\log\ \mathrm{SiO}$'], 'SiO_SiOUVenIR_main_iso'),
    'log_C2H2': ([(-14,-2), r'$\log\ \mathrm{C_2H_2}$'], 'C2H2_main_iso'),
    # 'log_AlO': ([(-14,-2), r'$\log\ \mathrm{AlO}$'], 'AlO_main_iso'),
    'log_H2S': ([(-14,-2), r'$\log\ \mathrm{H_2S}$'], 'H2S_Sid_main_iso'),
}
print(f' --> {len(opacity_params)} opacity parameters')
# Define the priors of the parameters
free_params = {

    # Uncertainty scaling
    # 'R_p': [(1.0, 5.0), r'$R_\mathrm{p}$'], # use this for robust results
     'R_p': [(1.6, 3.4), r'$R_\mathrm{p}$'], # R_p ~ 2.82 R_jup
    # 'R_p': [(2.72, 2.72), r'$R_\mathrm{p}$'], # R_p ~ 2.82 R_jup
    'log_g': [(2.5,4.5), r'$\log\ g$'], 
    # 'epsilon_limb': [(0.1,0.98), r'$\epsilon_\mathrm{limb}$'], 
    
    # veiling parameters
    # 'log_r_0': [(-20, -14), r'$\log\ r_0$'], # veiling amplitude at wave=min(wave)
    # 'alpha': [(1.0, 20.0), r'$\alpha$'], # veiling power-law index, should be positive for dust emission
    'R_d': [(0.0, 50.0), r'$R_d [R_{Jup}]$'], # disk radius in R_jup
    # 'R_d': [(14.0, 15.0), r'$R_d [R_{Jup}]$'], # disk radius in R_jup
    # 'log_R_d' : [(-2, 4), r'$\log\ R_d$'], # disk radius in R_jup
    'T_d': [(300.0, 1400.0), r'$T_d$'], # disk temperature in K
    # disk emission parameters
    # 'log_T_ex_12CO': [(1.8, 3.2), r'$T_\mathrm{ex}$'], # disk temperature in K
    # 'T_ex_12CO': [(400.0, 900.0), r'$T_\mathrm{ex}$'], # disk temperature in K
    # 'log_N_mol_12CO': [(15, 20), r'log $N_\mathrm{mol}$'], # disk temperature in K
    # 'log_A_au_12CO': [(-4, 1), r'$\log\ A_\mathrm{au}$'], # disk temperature in K
    # 'R_cav': [(1.0, 30.0), r'$R_\mathrm{cav}$'], # disk radius in R_jup
    # 'R_out': [(1.0, 200.0), r'$R_\mathrm{out}$'], # disk radius in R_jup
    # 'q': [(0.4, 1.2), r'$q$'], # disk temperature exponent
    # 'i_deg': [(0, 90), r'$i$'], # disk inclination in degrees
        
    
    'Av': [(0.0, 5.0), r'$A_v$'], # extinction in magnitudes
    
    'rv': [(4.,20.), r'$v_\mathrm{rad}$'],
    # 'log_H-' : [(-12,-6), r'$\log\ \mathrm{H^-}$'],

   'T_0': [(2000,8000), r'$T_0$'], 
    'log_P_RCE': [(-2.0,1.0), r'$\log\ P_\mathrm{RCE}$'],
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
}

# distance in pc to parallax
parallax_mas = 16.87 # Gaia DR3
d_pc = 1e3 / parallax_mas # ~ 59.17 pc

PT_interp_mode = 'linear'
PT_mode = 'RCE'

N_knots = 1 # spline knots (continuum fitting)

constant_params = {
    # General properties
    # 'R_p' : 1.0, 
    'parallax': parallax_mas, 
    'epsilon_limb': 0.5, 
    # 'log_g': 3.5,
    'vsini':0.,
    'T_star': 2430.0, # effective temperature in K, Cooper+2024 (Gaia DR3)

    # PT profile
    'N_knots': N_knots, # avoid using spline to fit the continuum
    
    # fix 12CO and H2O to the best-fit G235 values
    # 'log_12CO': -3.52,
    # 'log_H2O': -3.63,
    # 'rv': 12.16,
}

free_params.update({k:v[0] for k,v in opacity_params.items()})
# remove constant params from free_params dictionary
free_params = {k:v for k,v in free_params.items() if k not in list(constant_params.keys())}

# disk_species = ['H2O', '12CO', '13CO']
disk_species = ['12CO', '13CO']
T_ex_range = list(np.arange(300.0, 600.0+100.0, 100.0))
N_mol_range = list(np.logspace(15, 20, 6))
 
if len(disk_species) > 0:
    free_params.update({f'log_A_au_{sp}': [(-4, -1.0), f'$\log\ A_{{\mathrm{{au}}}} ({sp})$'] for sp in disk_species})
    free_params.update({f'log_N_mol_{sp}': [(15, 20), f'$\log\ N_{{\mathrm{{mol}}}} ({sp})$'] for sp in disk_species})
    free_params.update({f'T_ex_{sp}': [(300.0, 600.0), f'$\log\ T_{{\mathrm{{ex}}}} ({sp})$'] for sp in disk_species})



if grating == 'g235h+g395h':
    constant_params['gratings'] = ['g235h'] * 4 + ['g395h'] * 4
else:
    constant_params['gratings'] = [grating] * 4

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
# mask_lines = {'missing_opacity':(2050, 2080)}

####################################################################################
# Chemistry parameters
####################################################################################

#chem_mode  = 'free'
chem_mode  = 'free'
chem_kwargs = dict()

# Rayleigh scattering and continuum opacities
rayleigh_species=['H2','He']
continuum_opacities=['H2-H2', 'H2-He', 'H-']
line_species =[v[1] for _,v in opacity_params.items()]

# disk_species = []
# add H2 as line species, not a free parameter
# abundance of H2 calculated to sum(VMR) = 1
# line_species.append('H2_main_iso') # testing (2024-09-10)

species_to_plot_VMR , species_to_plot_CCF = [], []

####################################################################################
# Covariance parameters
####################################################################################

cov_mode = None

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
testing = True
const_efficiency_mode = True
sampling_efficiency = 0.05 if not testing else 0.20
# evidence_tolerance = 0.5
evidence_tolerance = 0.5 if not testing else 1.0
n_live_points = 200 if not testing else 100
n_iter_before_update = n_live_points * 3 if not testing else n_live_points * 2
# n_iter_before_update = 1
# generate a .txt version of this file

if __name__ == '__main__':
    from retrieval_base.config import Config
    import pathlib
    
    conf = Config(path=pathlib.Path(__file__).parent.absolute(), target=None, run=run)
    conf.save_json(file_params, globals())