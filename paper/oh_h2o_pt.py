from retrieval_base.retrieval import Retrieval
import retrieval_base.figures as figs
from retrieval_base.config import Config
from retrieval_base.auxiliary_functions import pickle_load, pickle_save, read_spirou_sample_csv
# import config_freechem as conf
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import corner
import pandas as pd
from datetime import datetime

import scienceplots
# reset to default
plt.style.use('default')
# plt.style.use(['latex-sans'])
plt.style.use(['sans'])
# enable latex
# plt.rcParams['text.usetex'] = True
plt.rcParams.update({
    "font.size": 11,
})

base_path = '/home/dario/phd/retrieval_base/'
nat_path = '/home/dario/phd/nat/figures/'
out_path = '/home/dario/phd/red_dwarf_isotopes/data/'
run ='fc5'
fs =14
# target = 'gl205'

def save_atmospheric_data_to_csv(target, Chem, PT, species=['OH','H2O']):
    """
    Save atmospheric data to CSV file for reproducibility.
    
    Parameters:
    -----------
    target : str
        Target name
    Chem : Chemistry object
        Chemistry object containing VMR envelopes
    PT : PT_profile object  
        PT profile object containing temperature envelopes and emission contribution
    species : list
        List of species to save (default: ['OH','H2O'])
    """
    
    # Create output directory for target
    target_dir = pathlib.Path(out_path) / target
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Get atmospheric data
    pressure = Chem.pressure  # bar
    
    # Initialize data dictionary
    data = {}
    
    # Add pressure column
    data['pressure_bar'] = pressure
    
    # Add OH and H2O VMR envelopes (q16, q50, q84)
    for species_name in species:
        if species_name in Chem.VMRs_envelopes:
            env = Chem.VMRs_envelopes[species_name]
            data[f'{species_name}_vmr_q16'] = env[0]  # 16th percentile
            data[f'{species_name}_vmr_q50'] = env[1]  # 50th percentile (median)
            data[f'{species_name}_vmr_q84'] = env[2]  # 84th percentile
        else:
            print(f'Warning: {species_name} not found in VMRs_envelopes')
            # Fill with NaN if species not available
            data[f'{species_name}_vmr_q16'] = np.full_like(pressure, np.nan)
            data[f'{species_name}_vmr_q50'] = np.full_like(pressure, np.nan)
            data[f'{species_name}_vmr_q84'] = np.full_like(pressure, np.nan)
    
    # Add temperature envelopes (q16, q50, q84)
    if hasattr(PT, 'temperature_envelopes'):
        temp_env = PT.temperature_envelopes
        # temperature_envelopes has shape (7, n_layers) with quantiles [0.15%, 2.5%, 16%, 50%, 84%, 97.5%, 99.85%]
        data['temperature_q16'] = temp_env[2]  # 16th percentile
        data['temperature_q50'] = temp_env[3]  # 50th percentile (median)  
        data['temperature_q84'] = temp_env[4]  # 84th percentile
    else:
        print(f'Warning: temperature_envelopes not found for {target}')
        data['temperature_q16'] = np.full_like(pressure, np.nan)
        data['temperature_q50'] = np.full_like(pressure, np.nan)
        data['temperature_q84'] = np.full_like(pressure, np.nan)
    
    # Add integrated emission contribution function
    if hasattr(PT, 'int_contr_em') and 'spirou' in PT.int_contr_em:
        data['emission_contribution'] = PT.int_contr_em['spirou']
    else:
        print(f'Warning: integrated emission contribution not found for {target}')
        data['emission_contribution'] = np.full_like(pressure, np.nan)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create header with metadata
    header_lines = [
        "# Supplementary data for thermal dissociation of water in M dwarf atmospheres",
        f"# Target: {target}",
        f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "# Authors: Darío González Picos, Ignas Snellen and Sam de Regt",
        "# Contact: picos@strw.leidenuniv.nl",
        "#",
        "# Data description:",
        "# This file contains atmospheric profiles from Bayesian retrieval analysis",
        "# showing thermal dissociation of water molecules in M dwarf stellar atmospheres.",
        "#",
        "# Column descriptions:",
        "# pressure_bar: Atmospheric pressure in bar",
        "# OH_vmr_q16: OH volume mixing ratio 16th percentile (1-sigma lower bound)",
        "# OH_vmr_q50: OH volume mixing ratio 50th percentile (median)",
        "# OH_vmr_q84: OH volume mixing ratio 84th percentile (1-sigma upper bound)",
        "# H2O_vmr_q16: H2O volume mixing ratio 16th percentile (1-sigma lower bound)", 
        "# H2O_vmr_q50: H2O volume mixing ratio 50th percentile (median)",
        "# H2O_vmr_q84: H2O volume mixing ratio 84th percentile (1-sigma upper bound)",
        "# temperature_q16: Temperature 16th percentile in Kelvin (1-sigma lower bound)",
        "# temperature_q50: Temperature 50th percentile in Kelvin (median)",
        "# temperature_q84: Temperature 84th percentile in Kelvin (1-sigma upper bound)",
        "# emission_contribution: Integrated emission contribution function",
        "#",
        "# Units:",
        "# Pressure: bar",
        "# Volume mixing ratios: dimensionless",
        "# Temperature: Kelvin", 
        "# Emission contribution: dimensionless (normalized)",
        "#",
        "# Notes:",
        "# - Each row corresponds to a different atmospheric layer",
        "# - Pressure increases from top to bottom of atmosphere",
        "# - VMR uncertainties represent 1-sigma (68% confidence) intervals",
        "# - Temperature uncertainties represent 1-sigma (68% confidence) intervals",
        "#"
    ]
    
    # Save to CSV file
    csv_file = target_dir / 'supplementary_fig_thermal_dissociation.csv'
    
    # Write header and data
    with open(csv_file, 'w') as f:
        # Write header lines
        for line in header_lines:
            f.write(line + '\n')
        
        # Write DataFrame to CSV (append mode, no header since we wrote custom header)
        df.to_csv(f, index=False, float_format='%.6e')
    
    print(f'Saved atmospheric data to {csv_file}')
    return csv_file

def plot_target(target, ax, color, label=None, species=['OH','H2O'], cache=True, ax_ice=None):

    if target not in os.getcwd():
        os.chdir(base_path + target)
        
    outputs = pathlib.Path(base_path) / target / 'retrieval_outputs'
    config_file = 'config_freechem.txt'

    posterior_samples_file = outputs / run / outputs / f'posterior_samples.npy'
    posterior_labels_file = outputs / run / outputs / f'posterior_labels.npy'
    Chem_file = outputs / run / 'test_data/bestfit_Chem.pkl'
    PT_file = outputs / run / 'test_data/bestfit_PT.pkl'

    conf = Config(path=base_path, target=target, run=run)(config_file)


    if not Chem_file.exists() or not cache:
        ret = Retrieval(
                    conf=conf, 
                    evaluation=False,
                    )

        _, samples = ret.PMN_analyze()
        # samples = samples.T
        ret.get_PT_mf_envelopes(samples)
        ret.Chem.get_VMRs_posterior()
        pickle_save(Chem_file, ret.Chem)
        
    # oh = 
    Chem = pickle_load(Chem_file)
    PT = pickle_load(PT_file)
    p = Chem.pressure
    
    # Save atmospheric data to CSV for reproducibility
    save_atmospheric_data_to_csv(target, Chem, PT, species=species)
    
    non_detections = []
        
    for i, s in enumerate(species):
        env = Chem.VMRs_envelopes[s]
        log_env = np.log10(env[:,len(env)//2])
        if log_env[-1] - log_env[0] > 3:
            print(f' Unconstrained {target}: {log_env[-1] - log_env[0]} for {species[i]}')
            ax[i].plot(env[1], p, color=color, lw=1, ls='--', label=label, zorder=-1) # 1sigma upper limits
            non_detections.append(species[i])
        else:   
            ax[i].fill_betweenx(p, env[0], env[2], color=color, alpha=0.3, lw=0)
            ax[i].plot(env[1], p, color=color, lw=1)
        
    
    ax[0].set(yscale='log', xscale='log', ylim=(np.max(p), np.min(p)))
    ax[1].set(xscale='log', ylim=(np.max(p), np.min(p)))
    
    ax[-1].fill_betweenx(p, PT.temperature_envelopes[1], PT.temperature_envelopes[-2], color=color, alpha=0.5, lw=0)
    ax[-1].plot(PT.temperature_envelopes[3], p, color=color, lw=1)
    
    if ax_ice is not None:
        ice = PT.int_contr_em['spirou']
        # ice /= np.median(ice)
        ax_ice.fill_between(ice, p, color=color, alpha=0.01, lw=0)
        ax_ice.plot(ice, p, color=color, lw=1.0, alpha=0.8, zorder=color[0], ls='dotted')
    
    return non_detections
    
species = ['OH','H2O']
n = len(species) + 1

fig, ax = plt.subplots(1,n, figsize=(7+n,3), sharey=True, gridspec_kw={'wspace': 0.04})
ax_ice = ax[-1].twiny()

df = read_spirou_sample_csv()
names = df['Star'].to_list()
ignore_targets = ['gl3622']

teff =  dict(zip(names, [float(t.split('+-')[0]) for t in df['Teff (K)'].to_list()]))
norm = plt.Normalize(3000.0, 3900.0)
cmap = plt.cm.coolwarm_r

non_detections = {k: (False, teff[k]) for k in names}
for i, name in enumerate(names):
    print(f'---> {name}')

    target = name.replace('Gl ', 'gl')
    if target in ignore_targets:
        print(f'---> Skipping {target}...')
        continue
    color = cmap(norm(teff[name]))
    # print(f' Target = {target}, color = {color}')
    non_detections_i = plot_target(target, ax, color, label=name, species=species, cache=True, ax_ice=ax_ice)
    if 'OH' in non_detections_i:
        non_detections[name] = (True, teff[name])
    else:
        non_detections[name] = (False, teff[name])
    # if i > 1:
    #     break # testing
        
# show number of detections compared to number of targets
detections = {k: (not(v[0]), v[1]) for k, v in non_detections.items()}
n_detections = sum([1 for k in detections.keys() if detections[k]])
n_targets = len(detections)
print(f' Number of detections: {n_detections} / {n_targets}')
# show minimum temperature of non-detections
min_teff = min([v[1] for v in detections.values() if v[0]])
print(f' Minimum temperature of detections: {min_teff}')

ax[0].set(ylabel='Pressure / bar')

xlims = [ax[0].get_xlim(), ax[1].get_xlim()]
# xmin = min([x[0] for x in xlims])
xmin = 1e-8
xmax = max([x[1] for x in xlims])
ax[0].set_xlim(xmin, xmax)
ax[0].set_xlabel(r'$X$'+'(OH)')
ax[1].set_xlim(xmin, xmax)
ax[1].set_xlabel(r'$X$'+'(H'+r'$_2$'+r'$^{16}$'+'O)')

ax[-1].set(xlabel='Temperature / K', xlim=(1200.0, 6200.0))
ax_ice_max = ax_ice.get_xlim()[1]
ax_ice.set_xlim(0, ax_ice_max*4.0)
# remove xticks from ax_ice
ax_ice.set_xticks([])

# add plot labels: a,b,c
for i, axi in enumerate(ax):
    axi.text(0.90, 0.90, f'{chr(97+i)}', transform=axi.transAxes, fontsize=12, fontweight='bold')


# plt.show()
fig_name = nat_path + 'oh_h2o_pt.pdf'
fig.savefig(fig_name, bbox_inches='tight')
print(f' Saved {fig_name}')
plt.close(fig)