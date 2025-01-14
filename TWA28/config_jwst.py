import numpy as np
import os
file_params = 'config_jwst.py'

####################################################################################
# Files and physical parameters
####################################################################################

target = 'TWA28'
lbl = 12
# run = f'lbl{lbl}_G2G3_8'
# run = f'lbl{lbl}_G1_2_freechem'
# grating = 'g235h+g395h'
# grating = 'g235h'
# gratings = ['g140h', 'g235h', 'g395h']
gratings = ['g140h']
grating_suffix = ''.join([str(g[:2]).upper() for g in gratings]) # e.g. G1G2
chem_mode = 'fastchem'
# chem_mode = 'freechem'

index = 1
run = f'lbl{lbl}_{grating_suffix}_{chem_mode}_{index}'
prefix = f'./retrieval_outputs/{run}/test_'

# Define PT profile
PT_interp_mode = 'linear' # ignored if PT_mode == 'fixed'
# PT_mode = 'RCE'
PT_mode = 'fixed'
PT_run = 'lbl12_G1G2G3_fastchem_1' # ignored if PT_mode != 'fixed'


config_data = {
    'NIRSpec': {
        'w_set': 'NIRSpec',

        'lbl_opacity_sampling' : lbl,
        'sigma_clip': 3,
        'sigma_clip_width': 31, # (2024-07-16): 21 --> 31
        'Nedge': 40, # (2024-10-18): 20 --> 40
        'log_P_range': (-5,2),
        'n_atm_layers': 60, # (2025-01-08): update 40 --> 60
        # 'T_cutoff': (1400.0, 3400.0), # DGP (2024-10-14): new parameter
        'T_cutoff': (1000.0, 3600.0), # DGP (2024-10-14): new parameter
        'P_cutoff': (1e-4, 1e1), # DGP (2024-10-14): new parameter
        }, 
    }

# distance in pc to parallax
parallax_mas_dict = dict(TWA28=16.87, TWA27A=15.46)
Teff_dict = dict(TWA28=2382.0, TWA27A=2430.0)
parallax_mas = parallax_mas_dict[target] # Gaia DR3, for TWA 28 (Manjavacas+2024)
d_pc = 1e3 / parallax_mas # ~ 59.17 pc

N_knots = 1 # spline knots (continuum fitting)

constant_params = {
    # General properties
    # 'R_p' : 1.0, 
    'parallax': parallax_mas, 
    'epsilon_limb': 0.5, 
    # 'log_g': 3.5,
    'vsini':0.,
    'T_star': Teff_dict[target], # effective temperature in K, Cooper+2024 (Gaia DR3)
    'M_star_Mjup': 20.0, # mass in Mjup, Manjavacas+2024

    # PT profile
    'N_knots': N_knots, # avoid using spline to fit the continuum
    
    # fix 12CO and H2O to the best-fit G235 values
    # 'log_12CO': -3.52,
    # 'log_H2O': -3.63,
    # 'rv': 12.16,
    'rv': 11.52, # FIXME: only keep it fixed for G140 retrieval
    'alpha_12CO': 0.69, # FIXME: only keep it fixed for G140 retrieval
    # 'alpha_H2O': 0.85,
}


####################################################################################
# Model parameters
####################################################################################
opacity_params = {
    'log_12CO': ([(-14,-2), r'$\log\ \mathrm{^{12}CO}$'], 'CO_high_Sam'),
    'log_13CO': ([(-14,-2), r'$\log\ \mathrm{^{13}CO}$'], 'CO_36_high_Sam'),
    'log_C18O': ([(-14,-2), r'$\log\ \mathrm{C^{18}O}$'], 'CO_28_high_Sam'),
    'log_C17O': ([(-14,-2), r'$\log\ \mathrm{C^{17}O}$'], 'CO_27_high_Sam'),
    
    'log_H2O': ([(-14,-2), r'$\log\ \mathrm{H_2O}$'], 'H2O_pokazatel_main_iso_Sam'),
    'log_H2O_181': ([(-14,-2), r'$\log\ \mathrm{H_2^{18}O}$'], 'H2O_181_HotWat78'),
    # 'log_HDO': ([(-14,-2), r'$\log\ \mathrm{HDO}$'], 'HDO_voronin'),
    'log_HF': ([(-14,-2), r'$\log\ \mathrm{HF}$'], 'HF_main_iso_new'), # DGP (2024-07-16): accidentally removed 
    'log_HCl': ([(-14,-2), r'$\log\ \mathrm{HCl}$'], 'HCl_main_iso'), # DGP (2024-07-16): try this one
    
    'log_CO2': ([(-14,-2), r'$\log\ \mathrm{CO_2}$'], 'CO2_main_iso'),
    # 'log_CN': ([(-14,-2), r'$\log\ \mathrm{CN}$'], 'CN_high'),
    
    # try new 2024-10-28
    'log_CH4': ([(-14,-2), r'$\log\ \mathrm{CH_4}$'], 'CH4_MM_main_iso'),
    'log_CH': ([(-14,-2), r'$\log\ \mathrm{CH}$'], 'CH_main_iso'),
    'log_NH3': ([(-14,-2), r'$\log\ \mathrm{NH_3}$'], 'NH3_coles_main_iso_Sam'),
    'log_HCN': ([(-14,-2), r'$\log\ \mathrm{HCN}$'], 'HCN_main_iso'),
    'log_NH': ([(-14,-2), r'$\log\ \mathrm{NH}$'], 'NH_kNigHt_main_iso'),
    'log_SH': ([(-14,-2), r'$\log\ \mathrm{SH}$'], 'SH_main_iso'),
    
    
    'log_Na': ([(-14,-4), r'$\log\ \mathrm{Na}$'], 'Na_Sam'),
    'log_K':  ([(-14,-4), r'$\log\ \mathrm{K}$'],  'K_static'),
    'log_Ca': ([(-14,-4), r'$\log\ \mathrm{Ca}$'], 'Ca_high'),
    'log_Ti': ([(-14,-4), r'$\log\ \mathrm{Ti}$'], 'Ti_high'),
    'log_Sc': ([(-14,-5), r'$\log\ \mathrm{Sc}$'], 'Sc_high'),
    'log_Mg': ([(-14,-5), r'$\log\ \mathrm{Mg}$'], 'Mg_high'),
    'log_Mn': ([(-14,-5), r'$\log\ \mathrm{Mn}$'], 'Mn_high'),
    'log_Fe': ([(-14,-5), r'$\log\ \mathrm{Fe}$'], 'Fe_high'),
    'log_Al': ([(-14,-5), r'$\log\ \mathrm{Al}$'], 'Al_high'),
    'log_Cr': ([(-14,-5), r'$\log\ \mathrm{Cr}$'], 'Cr_high'),
    'log_Cs': ([(-14,-5), r'$\log\ \mathrm{Cs}$'], 'Cs_high'),
    'log_V':  ([(-14,-5), r'$\log\ \mathrm{V}$'],  'V_high'),
    'log_Li': ([(-14,-5), r'$\log\ \mathrm{Li}$'], 'Li_high'),
    
    'log_FeH': ([(-14,-2), r'$\log\ \mathrm{FeH}$'], 'FeH_main_iso_Sam'),
    'log_CrH': ([(-14,-2), r'$\log\ \mathrm{CrH}$'], 'CrH_main_iso'),
    'log_TiH': ([(-14,-2), r'$\log\ \mathrm{TiH}$'], 'TiH_main_iso'),
    'log_CaH': ([(-14,-2), r'$\log\ \mathrm{CaH}$'], 'CaH_XAB_main_iso'),
    'log_AlH': ([(-14,-2), r'$\log\ \mathrm{AlH}$'], 'AlH_AloHa_main_iso'),
    'log_MgH': ([(-14,-2), r'$\log\ \mathrm{MgH}$'], 'MgH_main_iso'),
    'log_NaH': ([(-14,-2), r'$\log\ \mathrm{NaH}$'], 'NaH_main_iso'), # DGP (2024-07-16)
    'log_ScH': ([(-14,-2), r'$\log\ \mathrm{ScH}$'], 'ScH_main_iso'), # DGP (2024-07-16): try

    'log_OH': ([(-14,-2), r'$\log\ \mathrm{OH}$'], 'OH_MYTHOS_main_iso'),
    'log_H2': ([(-12,-0.01), r'$\log\ \mathrm{H_2}$'], 'H2_main_iso'),
    
    'log_VO': ([(-14,-2), r'$\log\ \mathrm{VO}$'], 'VO_HyVO_main_iso'), # DGP (2024-07-16): 3.4 um bump?
    'log_TiO': ([(-14,-2), r'$\log\ \mathrm{TiO}$'], 'TiO_48_Exomol_McKemmish'),
    'log_ZrO': ([(-14,-2), r'$\log\ \mathrm{ZrO}$'], 'ZrO_ZorrO_main_iso'),
    'log_SiO': ([(-14,-2), r'$\log\ \mathrm{SiO}$'], 'SiO_SiOUVenIR_main_iso'),
    'log_C2H2': ([(-14,-2), r'$\log\ \mathrm{C_2H_2}$'], 'C2H2_main_iso'),
    'log_AlO': ([(-14,-2), r'$\log\ \mathrm{AlO}$'], 'AlO_main_iso'),
    'log_MgO': ([(-14,-2), r'$\log\ \mathrm{MgO}$'], 'MgO_Sid_main_iso'),
    'log_H2S': ([(-14,-2), r'$\log\ \mathrm{H_2S}$'], 'H2S_Sid_main_iso'),
}

species_wave = {
    '12CO': [[1500, 1900], [2200, 3200], [4200, 5400]],
    '13CO': [[2200, 3200], [4200, 5400]],
    'C18O': [[2200, 3200], [4200, 5400]],
    'C17O': [[4200, 5400]],
    'H2O': [[0.0, np.inf]],
    'H2O_181': [[0.0, np.inf]],
    
    
    'HF': [[1200, np.inf]],
    # 'HCl': [[3050, np.inf]],

    'CO2': [[3700, 5400]],
    # 'HCN': [[0.0, np.inf]],
    
    'Na': [[0, np.inf]],
    # 'K': [[0, 1900], [2800, 3100], [3600,4100]],
    'K': [[0, np.inf]],
    'Ca': [[0, 2400]],
    'Ti': [[0, np.inf]],
    # 'Sc': [[0, 2600]], # add this back for final retrieval, potential opacity source at 1.35, 1.62 um
    'Mg': [[0, 2600]],
    # 'Mn': [[1200, 1600]], # add this back for final retrieval
    'Mn': [[0, np.inf]],
    'Fe': [[0, 2200]],
    'Al': [[1000, 1800]],
    # 'Cr': [[0, 2200], [3800, 4100]],
    # 'Cs': [[0, 1200], [1300, 1600],[2850,4000]],
    'FeH': [[0, 2400]],
    # 'V': [[0, 2300]],
    'CrH': [[0, 1650]],
    'TiH': [[0, 2000]], # add this back for final retrieval
    'CaH': [[0, 1400], [3800, 5300]], # add this back for final retrieval
    'AlH': [[1600, np.inf]],
    'MgH': [[0, 2000]],
    'NaH': [[0, 1400]],
    'ScH':[[0,1900.0]], # add this back for final retrieval
    'OH' : [[0, np.inf]],
    'VO': [[0, 1800],[4800, 5300]],
    'TiO': [[0,np.inf]],
    '46TiO': [[0, np.inf]],
    'SiO': [[2650,3100],[3900, 5200]],
    # 'H2S': [[1800, np.inf]],
}
    
    
all_species = [k[4:] for k in opacity_params.keys()]
# add line_species that are not in species_wave with (0, inf) = full range
# species_wave.update({s: [[0, np.inf]] for s in all_species if s not in species_wave})

opacity_params = {k:v for k,v in opacity_params.items() if k[4:] in species_wave.keys()}
assert len(opacity_params) > 0, 'No opacity parameters'
print(f' --> {len(opacity_params)} opacity parameters')
line_species_dict = {k[4:] : v[-1] for k,v in opacity_params.items()}
# replace keys in species_wave with line_species
species_wave = {line_species_dict[k]:v for k,v in species_wave.items()}

# Define the priors of the parameters
free_params = {

    # Uncertainty scaling
    # 'R_p': [(1.0, 5.0), r'$R_\mathrm{p}$'], # use this for robust results
     'R_p': [(1.8, 3.8), r'$R_\mathrm{p}$'], # R_p ~ 2.82 R_jup
    # 'R_p': [(2.4, 4.8), r'$R_\mathrm{p}$'], # R_p ~ 2.82 R_jup
    # 'R_p': [(2.72, 2.72), r'$R_\mathrm{p}$'], # R_p ~ 2.82 R_jup
    # 'log_g': [(2.5,4.5), r'$\log\ g$'], 
    # 'epsilon_limb': [(0.1,0.98), r'$\epsilon_\mathrm{limb}$'], 
    
    'rv': [(-30.0,30.0), r'$v_\mathrm{rad}$'],
    # 'log_H-' : [(-12,-6), r'$\log\ \mathrm{H^-}$'],
}
if PT_mode  == 'RCE':
    RCE_params = {'T_0': [(3000,8000), r'$T_0$'], 
    'log_P_RCE': [(-3.0,1.0), r'$\log\ P_\mathrm{RCE}$'],
    # 'dlog_P' : [(0.2, 1.6), r'$\Delta\log\ P$'],
    'dlog_P_1' : [(0.2, 1.6), r'$\Delta\log\ P_1$'], 
    'dlog_P_3' : [(0.2, 1.6), r'$\Delta\log\ P_3$'],
    'dlnT_dlnP_RCE': [(0.04, 0.38), r'$\nabla_{T,RCE}$'],
    'dlnT_dlnP_1':   [(0.04, 0.38), r'$\nabla_{T,1}$'],
    'dlnT_dlnP_0':   [(0.04, 0.38), r'$\nabla_{T,0}$'],
    'dlnT_dlnP_2':   [(0.04, 0.38), r'$\nabla_{T,2}$'],
    'dlnT_dlnP_3':   [(0.00, 0.38), r'$\nabla_{T,3}$'],
    'dlnT_dlnP_4':   [(0.00, 0.38), r'$\nabla_{T,4}$'],
    'dlnT_dlnP_5':   [(0.00, 0.38), r'$\nabla_{T,5}$'], # new points
    }
    
    free_params.update(RCE_params)
    
if PT_mode == 'fixed':
    constant_params['PT_run'] = PT_run # load PT profile from previous best fit
    constant_params['PT_target'] = target
    
# Surface gravity
log_g = [(3.0,4.5), r'$\log\ g$'] # uncomment this to fit log_g as a free parameter
# log_g = 4.49 # from PT_run
if isinstance(log_g, float):
    constant_params['log_g'] = log_g
else:
    free_params['log_g'] = log_g
    

# if grating == 'g235h' or grating==('g235h+g395h'):
if ('g235h' in gratings) or ('g395h' in gratings):
    # add disk params
    free_params['R_d'] =  [(0.0, 50.0), r'$R_d [R_{Jup}]$']
    free_params['T_d'] =  [(300.0, 1000.0), r'$T_d$']
    
else:
    # add disk params from best fit of g235h+g395h
    constant_params['R_d'] =  15.87 # from lbl12_G1G2G3_fastchem_1
    constant_params['T_d'] =  605.77 # from lbl12_G1G2G3_fastchem_1

fc_species_dict={
    'H2': 'H2',
    'He': 'He',
    'H': 'H',
    'e-': 'e-',
    'H2O': 'H2O1',
    '12CO': 'C1O1',
    'CH4': 'C1H4',
    'C2H2': 'C2H2',
    'CO2': 'C1O2',
    'H2S': 'H2S1',
    'CH': 'C1H1',
    'NH': 'H1N1',
    'NH3': 'H3N1',
    'HCN': 'C1H1N1_1',
    'SH': 'H1S1',
    'PH': 'H1P1',
  
    'SiS': 'S1Si1',
    'SiH': 'H1Si1',
    'HCl':'Cl1H1',
    'CaH': 'Ca1H1',
    'MgH': 'H1Mg1',
    'FeH': 'Fe1H1',
    'TiH': 'H1Ti1',
    'NaH': 'H1Na1',
    'AlH': 'Al1H1',
    'CrH': 'Cr1H1',
    # 'ScH': 'H
    'VO': 'O1V1',
    'TiO': 'O1Ti1',
    'SiO': 'O1Si1',

    'Na': 'Na',
    'K': 'K',
    'Fe': 'Fe',
    'Mg': 'Mg',
    'Ca': 'Ca',
    'Si': 'Si',
    'Ti': 'Ti',
    'Cr': 'Cr',
    'Al': 'Al',
    # 'O': 'O1',
    'OH': 'H1O1',
    'CN': 'C1N1',
    'HF': 'F1H1',
    # 'Sc': 'Sc1'  # Assuming 'Sc' follows the same pattern, though it's not explicitly listed
}

fc_species = list(fc_species_dict.keys()) # available species in chemistry table

isotopologues_dict = {
                        '13CO': ['log_12CO/13CO', [(1., 3.), r'$\log\ \mathrm{^{12}CO/^{13}CO}$']],
                        'C18O': ['log_12CO/C18O', [(1.5, 4.), r'$\log\ \mathrm{C^{16}O/C^{18}O}$']],
                        'C17O': ['log_12CO/C17O', [(1.5, 4.), r'$\log\ \mathrm{C^{16}O/C^{17}O}$']],
                        'H2O_181': ['log_H2O/H2O_181', [(1.5, 4.), r'$\log\ \mathrm{H_2^{16}O/H_2^{18}O}$']],
                        '46TiO': ['log_TiO/46TiO', [(1.0, 2.5), r'$\log\ \mathrm{^{48}TiO/^{46}TiO}$']],
                        '47TiO': ['log_TiO/47TiO', [(1.0, 2.5), r'$\log\ \mathrm{^{48}TiO/^{47}TiO}$']],
                        '49TiO': ['log_TiO/49TiO', [(1.0, 2.5), r'$\log\ \mathrm{^{48}TiO/^{49}TiO}$']],
}

# two_point_species = ['K']
two_point_species = []
for log_k, v in opacity_params.items():
    k = log_k[4:]
    
    if chem_mode == 'fastchem':
        if k in fc_species:
            # pass
            # add deviation parameter `alpha` for each species: log X = log X_0 + alpha
            # free_params[f'alpha_{k}'] = [(-3.0, 3.0), f'$\\alpha_{{{k}}}$']
            free_params[f'alpha_{k}'] = [(-3.0, 3.0), f'$\\alpha_{{{k}}}$']
        elif k in isotopologues_dict.keys():
            # add isotope ratio as free parameter
            free_params[isotopologues_dict[k][0]] = isotopologues_dict[k][1]
        else:
            free_params[log_k] = v[0]
            
    if chem_mode == 'free' or chem_mode == 'freechem':
        
        if k in two_point_species:
            free_params[log_k+'_1'] = [v[0][0], v[0][1][:-1] + '_1$']
            free_params[log_k+'_2'] = [v[0][0], v[0][1][:-1] + '_2$']
            free_params[log_k+'_P'] = [(-3.0, 1.0), r'$\log\ P$'+f'({k})$']
        else:
            free_params[log_k] = v[0]
        
        
        

print(f' --> {free_params} free parameters')
# free_params.update({k:v[0] for k,v in opacity_params.items()})
# remove constant params from free_params dictionary
free_params = {k:v for k,v in free_params.items() if k not in list(constant_params.keys())}

# disk_species = ['H2O', '12CO', '13CO']
constant_params['gratings'] = []
gratings_n = {'g140h': 4, 'g235h': 4, 'g395h': 4}
constant_params['gratings'] += [[g]*gratings_n[g] for g in gratings]
# flatten list of lists
constant_params['gratings'] = [item for sublist in constant_params['gratings'] for item in sublist]

if 'g395h' in gratings:
    # constant_params['gratings'] = ['g235h'] * 4 + ['g395h'] * 4
    
    # disk_species = ['12CO', '13CO', 'H2O']
    disk_species = ['12CO']
    T_ex_range = np.arange(300.0, 1000.0+50.0, 50.0).tolist()
    N_mol_range = np.logspace(15, 20, 6*2).tolist()
    
    disk_kwargs = dict(nr=20, ntheta=60)

    if len(disk_species) > 0:
        # free_params.update({f'log_A_au_{sp}': [(-5.0, -1.0), f'$\log\ A_{{\mathrm{{au}}}} ({sp})$'] for sp in disk_species})
        free_params.update({f'log_N_mol_{sp}': [(15.0, 20.0), f'$\log\ N_{{\mathrm{{mol}}}} ({sp})$'] for sp in disk_species})
        free_params.update({f'T_ex_{sp}': [(min(T_ex_range), max(T_ex_range)), f'$T_{{\mathrm{{ex}}}} ({sp})$'] for sp in disk_species})

        # free_params.update({'rv_disk': [(-50.0,50.0), r'$v_\mathrm{rad,disk}$']}) # new parameter 2024-10-28
        free_params.update({'R_cav': [(0.5, 30.0), r'$R_\mathrm{cav}$']}) # disk radius in R_jup
        free_params.update({'R_out': [(0.5, 200.0), r'$R_\mathrm{out}$']}) # disk radius in R_jup
        free_params.update({'i_deg': [(0.0, 90.0), r'$i$ (deg)']}) # disk inclination in degrees
        free_params.update({'nu': [(-1.0, 1.0), r'$\nu$']}) # angular asymmetry parameter
    
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

# Rayleigh scattering and continuum opacities
rayleigh_species=['H2','He']
continuum_opacities=['H2-H2', 'H2-He', 'H-']
# add free parameter for H-
free_params['log_Hminus'] = [(-12.0, -7.0), r'$\log\ H^-$']

line_species =list(set([v[1] for _,v in opacity_params.items()]))
line_species_dict = {k[4:]: v[1] for k,v in opacity_params.items()}

#chem_mode  = 'free'
# chem_mode  = 'free'

chem_kwargs = dict()
if chem_mode == 'fastchem':
    chem_kwargs['fastchem_grid_file'] = '../data/fastchem_grid_twx.h5'
    # chem_kwargs['line_species'] = line_species_dict


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
    conv_adiabat = False, 

    ln_L_penalty_order = 3, 
    PT_interp_mode = PT_interp_mode, 

    enforce_PT_corr = False, 
    # n_T_knots = N_PT_knots,
    sonora=dict(teff=2400, log_g=4.0),
)
if PT_mode == 'fixed':
    PT_kwargs['PT_target'] = target
    PT_kwargs['PT_run'] = PT_run
    
    

####################################################################################
# Multinest parameters
####################################################################################
testing = True
const_efficiency_mode = True
sampling_efficiency = 0.05 if not testing else 0.20
# evidence_tolerance = 0.5
evidence_tolerance = 0.5 if not testing else 1.0
n_live_points = 400 if not testing else 200
n_iter_before_update = n_live_points * 3 if not testing else n_live_points * 2
# n_iter_before_update = 1
# generate a .txt version of this file

if __name__ == '__main__':
    from retrieval_base.config import Config
    import pathlib
    
    conf = Config(path=pathlib.Path(__file__).parent.absolute(), target=None, run=run)
    conf.save_json(file_params, globals())
    print(f' Number of dimensions: {len(free_params)}')