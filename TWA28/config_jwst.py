import numpy as np
import os
file_params = 'config_jwst.py'

####################################################################################
# Files and physical parameters
####################################################################################

target = 'TWA28'
lbl = 10
# lbl = 15
# run = f'lbl{lbl}_G2G3_8'
# run = f'lbl{lbl}_G1_2_freechem'
# grating = 'g235h+g395h'
# # gratings = ['g235h']
gratings = ['g235h', 'g395h']
# gratings = ['g140h', 'g235h', 'g395h']
# gratings = ['g140h']
grating_suffix = ''.join([str(g[:2]).upper() for g in gratings]) # e.g. G1G2
chem_mode = 'fastchem'
# chem_mode = 'freechem'
# cov_mode = 'None'
cov_mode = 'newGP' # NEW 2025-02-27: use new GP mode, keep OLDCovariance for compatibility
cov_mode_label = f'_{cov_mode}' if cov_mode != 'None' else ''

index = 1
run = f'freeslab_lbl{lbl}_{grating_suffix}_{index}'
# run = 'test_g395h'
prefix = f'./retrieval_outputs/{run}/test_'

# Define PT profile
PT_interp_mode = 'linear' # ignored if PT_mode == 'fixed'
PT_mode = 'RCE'
# PT_mode = 'fixed'
PT_run = 'lbl12_G1G2G3_fastchem_1' # ignored if PT_mode != 'fixed'


config_data = {
    'NIRSpec': {
        'w_set': 'NIRSpec',

        'lbl_opacity_sampling' : lbl,
        'n_order_factor': 6, # NEW 2025-02-27: number of chunks to divide each order into
        'apply_psf_correction': False, # NEW 2025-03-10: apply PSF correction
        'sigma_clip': 0, # NEW 2025-02-27: disable sigma clipping
        'sigma_clip_max_iter': 6,
        'sigma_clip_width': 31, # (2025-02-15): 31
        'Nedge': 10, # (2025-02-27): 40 --> 20, new data already has edge effects discarded
        'log_P_range': (-5,2),
        'n_atm_layers': 40, # (2025-01-08): update 40 --> 60
        # 'T_cutoff': (1400.0, 3400.0), # DGP (2024-10-14): new parameter
        'T_cutoff': (1200.0, 3400.0), # DGP (2024-10-14): new parameter
        'P_cutoff': (1e-3, 1e1), # DGP (2024-10-14): new parameter
        'flux_unit_factor': 1e14, # DGP (2025-02-06): new parameter
        }, 
    }
# from JWST docs NIRSpec
wave_range_gratings = dict(
                   g140h= (970.0, 1830.0),
                   g235h= (1630.0, 3060.0),
                   g395h= (2840.0, 5300.0),
                   )
cenwave_gratings = {k:np.mean(v) for k,v in wave_range_gratings.items()}
wave_range_list = [wave_range_gratings[grating] for grating in gratings]
wave_range = [float(np.min(wave_range_list)), float(np.max(wave_range_list))]

# distance in pc to parallax
parallax_mas_dict = dict(TWA28=16.87, TWA27A=15.46)

# distance from Gaia, old distances from Ducourant+2008 were off...
distance_pc_dict = dict(TWA28=59.9, 
                        TWA27A=64.4)
Teff_dict = dict(TWA28=2382.0, TWA27A=2430.0)
mass_dict = dict(TWA28=(20.9, 6.0), TWA27A=(19.9, 5.0))

parallax_mas = parallax_mas_dict[target] # Gaia DR3, for TWA 28 (Manjavacas+2024)
# d_pc = 1e3 / parallax_mas # ~ 59.17 pc
d_pc = distance_pc_dict[target]

N_knots = 1 # spline knots (continuum fitting)

constant_params = {
    # General properties
    # 'R_p' : 1.0, 
    # 'parallax': parallax_mas, 
    'd_pc': d_pc,
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
    # 'log_FeH': ([(-14,-2), r'$\log\ \mathrm{FeH}$'], 'FeH_main_iso'),
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
    # 'log_TiO': ([(-14,-2), r'$\log\ \mathrm{TiO}$'], 'TiO_48_Exomol_McKemmish'), # (2025-01-14): change to ALL_ISO
    'log_TiO': ([(-14,-2), r'$\log\ \mathrm{TiO}$'], 'TiO_all_iso_Exomol_McKemmish'), # (2025-01-14): change to ALL_ISO
    # 'log_46TiO': ([(-14,-2), r'$\log\ \mathrm{46TiO}$'], 'TiO_46_Exomol_McKemmish'),
    'log_ZrO': ([(-14,-2), r'$\log\ \mathrm{ZrO}$'], 'ZrO_ZorrO_main_iso'),
    'log_SiO': ([(-14,-2), r'$\log\ \mathrm{SiO}$'], 'SiO_SiOUVenIR_main_iso'),
    'log_C2H2': ([(-14,-2), r'$\log\ \mathrm{C_2H_2}$'], 'C2H2_main_iso'),
    'log_AlO': ([(-14,-2), r'$\log\ \mathrm{AlO}$'], 'AlO_main_iso'),
    'log_MgO': ([(-14,-2), r'$\log\ \mathrm{MgO}$'], 'MgO_Sid_main_iso'),
    'log_H2S': ([(-14,-2), r'$\log\ \mathrm{H_2S}$'], 'H2S_Sid_main_iso'),
    'log_NO':  ([(-14,-2), r'$\log\ \mathrm{NO}$'], 'NO_XABC_main_iso'),
    'log_SiH': ([(-14,-2), r'$\log\ \mathrm{SiH}$'], 'SiH_SiGHTLY_main_iso'),
}


species_wave = {
    '12CO': [[1500, 1900], [2200, 3200], [4200, 5400]],
    '13CO': [[2200, 3200], [4200, 5400]],
    'C18O': [[2200, 2420], [4200, 5400]],
    # 'C18O': [[4200, np.inf]],
    'C17O': [[4200, 5400]], # 
    'H2O': [[0.0, np.inf]],
    'H2O_181': [[0.0, np.inf]],
    
    
    'HF': [[2100, 2950.0]],
    'HCl': [[3050, 4915]], #

    'CO2': [[2800, 3200],[3900, 5400]],
    # 'CH4': [[2900.0, 3900.0]], # TODO: add this back for final retrieval
    # 'NH3': [[2700.0, np.inf]], # TODO: add this back for final retrieval
    # 'HCN': [[2800.0, np.inf]], # TODO: add this back for final retrieval
    'CH4': [[1580, np.inf]], # check from here... species contribution plot
    # 'NH3': [[0.0, np.inf]],
    # 'HCN': [[0.0, np.inf]], # unclear, keep?

    'Na': [[0, 2400.0], [3390.0, 3600.0], [3900.0,4100.0]],
    # 'K': [[0, 1900], [2800, 3100], [3600,4100]],
    'K': [[0, 1900.0], [2440, 4100]],
    'Ca': [[0, 2400.0]],
    'Ti': [[0, 2400.0]],
    # 'Sc': [[0, 2600]], # 2025-02-19: not detected...
    'Mg': [[0, 2160]],
    # 'Mn': [[1200, 1600]], # add this back for final retrieval
    # 'Mn': [[0, 2400.0]],
    'Fe': [[0, 2160]],
    'Al': [[1000, 1800]],
    # 'Cr': [[0, 2200], [3800, 4100]],
    # 'Cs': [[0, 1200], [1300, 1600],[2850,4000]],
    # 'SH': [[0, np.inf]],
    'FeH': [[0, 1850]],
    # 'V': [[0, 2300]],
    'CrH': [[0, 1400]],
    # 'TiH': [[0, 2000]], # add this back for final retrieval
    # 'CaH': [[0, 1400]], # Feb 18: not detected...
    # 'MgH': [[0, 2000]],
    'NaH': [[0, 1400]],
    # 'ScH':[[0,1900.0]], # Feb 18: not detected...
    'OH' : [[0, 5300.0]],
    'VO': [[0, 1450.0], [4500.0, 5300.0]],
    'TiO': [[0,1450],[4500.0, 5300.0]],
    # '46TiO': [[0, np.inf]],
    'SiO': [[2650,5300]],
    # 'H2S': [[0.0, np.inf]],# Feb 18: not detected... alpha < -1.2 (+0.32, -0.42)
    # 'AlH': [[3000, 4600]],
    'AlH': [[2920, 4600]],
    # 'CH': [[0.0, 2200], [3000, np.inf]],
    # 'SiH': [[4500, 5300]],
    # 'SiH': [[0.0, np.inf]],
    # 'MgO': [[3000, 5300]],
    # 'AlO': [[3000, 5300]],
    'AlO': [[0.0, 4500]],
}

#FIXME: only for testing
# all_species = [k[4:] for k,v in opacity_params.items() if not v[-1].endswith('_high')]
# ignore = ['13CO','C18O','C17O','H2O_181']
# species_wave = {k:[[2800, 4100]] for k in all_species if k not in ignore}

# include_only = ['FeH', 'H2O'] # FIXME: manually add species here
# if len(include_only) > 0:
#     species_wave = {k:v for k,v in species_wave.items() if k in include_only}
ignore_species = []
for species in species_wave:
    wmin = np.min(species_wave[species])
    wmax = np.max(species_wave[species])
    # print(f'{species}: {wmin} - {wmax}')
    if wmax < wave_range[0] or wmin > wave_range[1]:
        # print(f'{species} is not covered by {gratings[0]}')
        ignore_species.append(species)
    
del wmin, wmax
print(f' --> {len(ignore_species)} species ignored: {ignore_species}')
species_wave = {k:v for k,v in species_wave.items() if k not in ignore_species}

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
     'R_p': [(2.2, 3.8), r'$R_\mathrm{p}$'], # R_p ~ 2.82 R_jup
    # 'R_p': [(5.72, 5.73), r'$R_\mathrm{p}$'], # R_p ~ 2.82 R_jup
    #  'mass': [(10.0, 40.0), r'$M [M_\mathrm{Jup}]$'],
    # 'mass': [mass_dict[target], r'$M [M_\mathrm{Jup}]$'],
    # 'R_p': [(2.4, 4.8), r'$R_\mathrm{p}$'], # R_p ~ 2.82 R_jup
    # 'R_p': [(2.72, 2.72), r'$R_\mathrm{p}$'], # R_p ~ 2.82 R_jup
    # 'log_g': [(2.5,4.5), r'$\log\ g$'], 
    # 'epsilon_limb': [(0.1,0.98), r'$\epsilon_\mathrm{limb}$'], 
    
    'rv': [(-30.0,30.0), r'$v_\mathrm{rad}$'],
    # 'b': [(0.0, 3.0), r'$b$'], # error scaling parameter as in var2_eff = var2_0 * 10**b
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

gaussian_params = []
invgamma_params = []
# if 'mass' in free_params.keys():
#     gaussian_params = ['mass']
    
    
if 'mass' not in free_params.keys():
    log_g = [(3.0,4.5), r'$\log\ g$'] # uncomment this to fit log_g as a free parameter
    # log_g = 4.49 # from PT_run
    if isinstance(log_g, float):
        constant_params['log_g'] = log_g
    else:
        free_params['log_g'] = log_g

    

# if grating == 'g235h' or grating==('g235h+g395h'):
# if ('g235h' in gratings) or ('g395h' in gratings):
if 'g395h' in gratings:
    # add disk params
    # free_params['R_d'] =  [(0.0, 50.0), r'$R_d [R_{Jup}]$']
    free_params['log_R_d'] = [(0.0, 2.0), r'$R_d [R_{Jup}]$']
    free_params['T_d'] =  [(300.0, 900.0), r'$T_d$']
    # free_params['log_T_d'] = [(2.0, 3.2), r'$T_d$']
    # use gaussian priors from low res NIRSpec+Spitzer fit
    # free_params['R_d'] = [(14.8, 2.0), r'$R_d [R_{Jup}]$']
    # free_params['T_d'] = [(631.0, 20.0), r'$T_d$']
    # gaussian_params.append('R_d')
    # gaussian_params.append('T_d')
    
else:
    # add disk params from best fit of g140h+g235h+g395h
    constant_params['R_d'] =  13.80 # from freeslab_lbl10_G1G2G3_0
    constant_params['T_d'] =  654.37 # from freeslab_lbl10_G1G2G3_0

fc_species_dict={
    'H2': 'H2',
    'He': 'He',
    'H': 'H',
    'e-': 'e-',
    'H2O': 'H2O1',
    '12CO': 'C1O1',
    # 'CH4': 'C1H4', # remove from here to use freechem for this species
    # 'C2H2': 'C2H2',
    'CO2': 'C1O2',
    # 'H2S': 'H2S1',
    # 'CH': 'C1H1',
    # 'NH': 'H1N1',
    # 'NH3': 'H3N1',
    # 'HCN': 'C1H1N1_1',
    # 'SH': 'H1S1',
    'PH': 'H1P1',
  
    'SiS': 'S1Si1',
    # 'SiH': 'H1Si1',
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
                        '46TiO': ['log_TiO/46TiO', [(0.0, 2.5), r'$\log\ \mathrm{^{48}TiO/^{46}TiO}$']],
                        '47TiO': ['log_TiO/47TiO', [(0.0, 2.5), r'$\log\ \mathrm{^{48}TiO/^{47}TiO}$']],
                        '49TiO': ['log_TiO/49TiO', [(0.0, 2.5), r'$\log\ \mathrm{^{48}TiO/^{49}TiO}$']],
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
            # free_params[f'alpha_{k}'] = [(-3.0, 3.0), f'$\\alpha_{{{k}}}$']
            free_params[f'alpha_{k}'] = [(0.0, 1.0), f'$\\alpha_{{{k}}}$']
            gaussian_params.append(f'alpha_{k}')
            
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
        
        
        

# free_params.update({k:v[0] for k,v in opacity_params.items()})
# remove constant params from free_params dictionary
free_params = {k:v for k,v in free_params.items() if k not in list(constant_params.keys())}

# disk_species = ['H2O', '12CO', '13CO']
constant_params['gratings'] = []

n_order_factor = config_data['NIRSpec']['n_order_factor']
gratings_n = {'g140h': n_order_factor, 'g235h': n_order_factor, 'g395h': n_order_factor}
constant_params['gratings'] += [[g]*gratings_n[g] for g in gratings]
# flatten list of lists
constant_params['gratings'] = [item for sublist in constant_params['gratings'] for item in sublist]

if 'g395h' in gratings:
    # constant_params['gratings'] = ['g235h'] * 4 + ['g395h'] * 4
    
    disk_species = ['12CO', '13CO', 'H2O']
    # disk_species = ['12CO', '13CO']
    # disk_species = ['12CO']
    # T_ex_range = np.arange(500.0, 1150.0+51.0, 50.0).tolist()
    # T_ex_range = np.arange(200.0, 1000.0+100.0, 100.0).tolist()
    T_ex_range = np.arange(500.0, 2000.0+100.0, 100.0).tolist() # updated 2025-06-23
    # T_ex_range = [400.0, 600.0, 1200.0]
    N_mol_min, N_mol_max = 14.0, 18.0
    N_mol_range = np.logspace(N_mol_min, N_mol_max, 6*2).tolist()
    # N_mol_range = np.array([10**14.0, 10**16.0, 10**18.0]).tolist()
    # T_ex_range = np.arange(300.0, 1350.0+50.0, 50.0).tolist()
    # N_mol_range = np.logspace(15, 22, 6*2).tolist()
    
    
   

    # define disk emission parameters (and outer radius)
    hot_cold_model = False
    # slabs = dict(T_ex = [1200.0, 800.0, 600.0],
    #              N_mol = [10**18.0, 10**17.0, 10**16.0])
    slabs = {}
    n_slabs = len(slabs)

    labels = ['_hot', '_cold'] if hot_cold_model else ['']
    
    disk_kwargs = dict(nr=18, ntheta=36, hot_cold_model=hot_cold_model, n_slabs=n_slabs)
    constant_params.update(disk_kwargs)
    if hot_cold_model:
         # define disk geometry parameters
        free_params.update({'log_R_cav': [(0.0, 2.0), r'$R_\mathrm{cav}$']}) # disk inner radius in R_jup
        free_params.update({'i_deg': [(0.0, 90.0), r'$i$ (deg)']}) # disk inclination in degrees
        for label in labels:
            free_params.update({'log_N_mol_12CO'+label: [(N_mol_min, N_mol_max), r'$\log\ N_{{\mathrm{{mol}}}} (\mathrm{^{12}CO})$'+label]})
            free_params.update({'log_T_ex_12CO'+label: [(np.log10(min(T_ex_range)), np.log10(max(T_ex_range)),), r'$\log\ T_{{\mathrm{{ex}}}} (\mathrm{^{12}CO})$'+label]})
            free_params.update({'log_R_out'+label: [(0.5, 3.0), r'$R_\mathrm{out}$'+label]}) # disk outer radius in R_jup
    elif n_slabs > 0:
        for i in range(n_slabs):
            constant_params[f'N_mol_{i}'] = slabs['N_mol'][i]
            constant_params[f'T_ex_{i}'] = slabs['T_ex'][i]
            # free_params[f'log_A_au_{i}'] = [(-5.0, 2.0), r'$A_\mathrm{au}$'+f'_{i}']
            free_params[f'log_R_jup_{i}'] = [(-1.0, 3.0), r'$\log\ R_\mathrm{jup}$'+f'_{i}']
            
    else:
        free_params['log_N_mol'] = [(N_mol_min, N_mol_max), r'$\log\ N_{{\mathrm{{mol}}}}$']
        free_params['log_T_ex'] = [(np.log10(min(T_ex_range)), np.log10(max(T_ex_range)),), r'$\log\ T_{{\mathrm{{ex}}}}$']
        free_params['log_R_jup'] = [(0.0, 3.0), r'$\log\ R_\mathrm{jup}$']
        free_params['rv_disk'] = [(-60.0, 60.0), r'$v_\mathrm{disk}$']
    # free_params.update({'nu': [(-1.0, 1.0), r'$\nu$']}) # angular asymmetry parameter
    
####################################################################################
#
####################################################################################
scale_flux = False
scale_flux_eps = 0.00 # no scaling, set to 0.05 for a 5% deviation even with scale_flux=False
scale_err  = True
# if scale_err == False:
#     free_params['beta2'] = [(1.0, 10.0), r'b$^2$']
#     invgamma_params.append('beta2')
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
# add free parameter for H- opacity
if 'g140h' in gratings:
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
trunc_dist = 4.0
# max_separation = 5
# max_separation_gratings = {'g140h': 5, 'g235h': 5, 'g395h': 5}
# length_scale_factors = {k:1.0 for k in gratings}
# if cov_mode == 'GP' or cov_mode == 'SGP':
    
#     # log_l_prior_gratings = {'g140h': (-0.4, 0.18), 'g235h': (-0.4, 0.42), 'g395h': (-0.4, 0.64)}
#     free_params['log_l_G'] = [(-0.40, 0.20), r'$\log\ l_G$']
#     max_separation = 10.0**free_params['log_l_G'][0][1] * trunc_dist
#     # free_params['log_l_G'] = [(0.0, 0.1), r'$\log\ l_G$']
#     for grating in gratings:
#         # free_params[f'log_a_{grating}_G'] = [(-1.0, 0.6), r'$\log\ a_{G}$' + f'({grating})']
#         # free_params[f'a_{grating}_G'] = [(3.0, 2.0), r'$a_{G}$' + f'({grating})']
#         # invgamma_params.append(f'a_{grating}_G')
#         # free_params[f'log_a_{grating}_G'] = [(0.0, 0.1), r'$\log\ a_{G}$']
#         length_scale_factors[grating] = cenwave_gratings[grating] / cenwave_gratings['g140h']
#         constant_params[f'a_{grating}_G'] = 1.0
#         # free_params[f'log_l_{grating}_G'] = [log_l_prior_gratings[grating], r'$\log\ l_{G}$' + f'({grating})']
#         # max_separation_gratings[grating] = 10.0**log_l_prior_gratings[grating][1] * trunc_dist
# global error scaling per grating

for grating in gratings:
    free_params[f'b_{grating}'] = [(0.0, 3.0), r'$\log\ b$' + f'({grating})']

free_params['log_l_G'] = [(1.4, 2.6), r'$\log\ l_G$ [km/s]'] # from 30 to ~200 km/s ~ 5 pixels
# free_params['log_a_G'] = [(-1.0, 1.0), r'$\log\ a_G$']
constant_params['a_G'] = 1.0 # fix it to 1.0
cov_kwargs = {
    'scale_amplitude': True,
    'max_length_scale': 10.0**free_params['log_l_G'][0][1],
    'truncate': trunc_dist,
    'local_sigma': 30.0,  # width of local kernel (km/s), 120 km/s ~ 3 pixels
    'local_threshold': 4.0, # number of standard deviations to use for local covariance
    'max_outliers': 5, # maximum number of outliers to flag
}

# add all items in cov_kwargs to constant_params
constant_params.update(cov_kwargs)

# cov_kwargs = dict(
#     # trunc_dist   = 2, # set to 3 for accuracy, 2 for speed
#     scale_GP_amp = True, 
#     max_separation = max_separation,
#     # max_separation_gratings = max_separation_gratings,
#     trunc_dist = trunc_dist,
#     length_scale_factors = length_scale_factors,
#     # Prepare the wavelength separation and
#     # average squared error arrays and keep 
#     # in memory
#     prepare_for_covariance = True
# )

# if free_params.get('log_l') is not None:
#     cov_kwargs['max_separation'] =  cov_kwargs['trunc_dist']
#     cov_kwargs['max_separation'] *= 10**free_params['log_l'][0][1]
    
####################################################################################
# PT parameters
####################################################################################


PT_kwargs = dict(
    conv_adiabat = False, 
    PT_interp_mode = PT_interp_mode, 
    # sonora=dict(teff=2400, log_g=4.0),
)
if PT_mode == 'fixed':
    PT_kwargs['PT_target'] = target
    PT_kwargs['PT_run'] = PT_run
    
    

####################################################################################
# Multinest parameters
####################################################################################
testing = False
const_efficiency_mode = True
sampling_efficiency = 0.05 if not testing else 0.05
# evidence_tolerance = 0.5
evidence_tolerance = 0.5 if not testing else 0.5
n_live_points = 800 if not testing else 400
n_iter_before_update = n_live_points * 2 if not testing else n_live_points * 1
# n_iter_before_update = 1
# generate a .txt version of this file
print(f' --> {free_params} free parameters')
del wave_range_gratings, wave_range_list

if __name__ == '__main__':
    from retrieval_base.config import Config
    import pathlib
    
    conf = Config(path=pathlib.Path(__file__).parent.absolute(), target=None, run=run)
    conf.save_json(file_params, globals())
    print(f' Number of dimensions: {len(free_params)}')