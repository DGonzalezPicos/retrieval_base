from retrieval_base.retrieval import pre_processing, Retrieval
from retrieval_base.parameters import Parameters
from retrieval_base import auxiliary_functions as af

import numpy as np
import matplotlib.pyplot as plt
# set fontsize to 16
plt.style.use('/home/dario/phd/retrieval_base/HBDs/my_science.mplstyle')
# plt.rcParams.update({'font.size': 24})

import pathlib
import pickle
import corner
import pandas as pd
import json

import pyfastchem
from astropy import constants as const



path = pathlib.Path('/home/dario/phd/retrieval_base')
# out_path = path / 'HBDs'
out_path = pathlib.Path('/home/dario/phd/Hot_Brown_Dwarfs_Retrievals/figures/')

targets = dict(
                J1200='freechem_10', 
               TWA28='freechem_6',  # freechem_7 goes down to logP = -6
               J0856='freechem_9'
               )
colors = dict(J1200='royalblue', TWA28='seagreen', J0856='indianred')


mode = 'basic' # basic, rainout

## FastChem
#we read in a p-T structure for a brown dwarf
path_fc = pathlib.Path('/home/dario/phd/fastchem/')
#here, we currently use the standard one from FastChem
output_dir = path_fc / 'output'

plot_species_labels = [
    ('12CO', 'C1O1'),
    ('H2O', 'H2O1'),
     ('Ca','Ca'),
     ('Na','Na'),
     ('Ti','Ti'),
     ('HF', 'F1H1')
    ]
alpha_min = 0.62

## PLotting
fig, ax = plt.subplots(len(targets),2,
                                     figsize=(7,7), 
                                     sharey=True,
                                     sharex='col',
                                     gridspec_kw=dict(wspace=0.12,
                                                      hspace=0.05,
                                                      width_ratios=[1,3]))

for i, (target, retrieval_id) in enumerate(targets.items()):
    
    data_path = pathlib.Path('/home/dario/phd/retrieval_base') / f'{target}'
    print(data_path)
    
    # bestfit_params = 
    retrieval_path = data_path / f'retrieval_outputs/{retrieval_id}'
    assert retrieval_path.exists(), f'Retrieval path {retrieval_path} does not exist.'
    
    PT = pickle.load(open(retrieval_path / 'test_data/bestfit_PT.pkl', 'rb'))
    
    # choose axes
    ax_PT = ax[i, 0]
    ax_chem = ax[i, 1]
    # remove empty axis
    # [ax[2*i+1, z].axis('off') for z in range(2)]
    
    ax_PT.fill_betweenx(PT.pressure, PT.temperature_envelopes[0], PT.temperature_envelopes[-1], color=colors[target], alpha=0.2)
    ax_PT.fill_betweenx(PT.pressure, PT.temperature_envelopes[1], PT.temperature_envelopes[-2], color=colors[target], alpha=0.4)
    # ax_PT.plot(PT.temperature, PT.pressure, color=colors[target], lw=2.5, label=target)
    # ax_PT.plot(PT.temperature_posterior[-1,:], PT.pressure, color=colors[target], lw=2.5, label=target)
    ax_PT.plot(PT.temperature_envelopes[3], PT.pressure, color=colors[target], lw=2.5, label=target)

    # plot integrated contribution function
    icf = np.load(retrieval_path / 'test_data/bestfit_int_contr_em_K2166.npy')
    print(f'shape of icf = {icf.shape}')
    ax_PT_icf = ax_PT.twiny()
    # make the zero on the right side of the ax_PTis
    ax_PT_icf.plot(icf, PT.pressure, color=colors[target], lw=2.5, label='ICF', ls='-', alpha=0.3)
    ax_PT_icf.fill_betweenx(PT.pressure, icf, 0., color=colors[target], alpha=0.1)
    alpha_0 = af.weigh_alpha(icf, PT.pressure, PT.temperature_envelopes[3], ax=ax_PT, plot=True, alpha_min=alpha_min,
                             T_max=10_000)
    # alpha_1 = af.weigh_alpha(icf, PT.pressure, PT.temperature_envelopes[0], ax=ax_PT, 
    #                          plot=True, alpha_min=1.-np.array(alpha_0))
    
    
    
    ax_PT_icf.invert_xaxis()
    # ax_PT_icf.set(xlim=(1.2*icf.max(), 0.))
    ax_PT_icf.set(xlim=(0., 1.2*icf.max()))
    # remove xticks from ax_PT_icf
    ax_PT_icf.set_xticks([])
    
    
    # plot free chemistry retrieved abundances
    chem = pickle.load(open(retrieval_path / 'test_data/bestfit_Chem.pkl', 'rb'))
    # chem.VMRs_envelopes()
    # line_species = ['12CO', 'H2O']
    colors_list = []
    MMW = chem.mass_fractions_posterior['MMW'][3].mean()

    
    
    for ls in plot_species_labels:
        # atomic number
        if ls[0] == 'CO':
            ls[0] = '12CO'
            
        line_species = chem.read_species_info(ls[0], 'pRT_name')
        mass         = chem.read_species_info(ls[0], 'mass')
        color        = chem.read_species_info(ls[0], 'color')
        colors_list.append(color)

        VMR_envelopes = chem.mass_fractions_envelopes[line_species] * (MMW/mass)
        
        if ls[0] == '12CO':
            VMR_12CO = VMR_envelopes[3]
            
        VMR_envelopes = VMR_envelopes / VMR_12CO
        
        ax_chem.plot(VMR_envelopes[3], PT.pressure, label=ls[0], lw=2.5, ls='-', color=color)
        
        ax_chem.fill_betweenx(PT.pressure, VMR_envelopes[2], VMR_envelopes[4], alpha=0.2, color=color)
            
        


    # fastchem 
    #make a copy of the solar abundances from FastChem
    #create a FastChem object
    fastchem = pyfastchem.FastChem(
    str(path_fc / 'input/element_abundances/asplund_2009.dat'), 
    str(path_fc / 'input/logK/logK.dat'),
    str(path_fc / 'input/logK/logK_condensates.dat'),
    1)

    #create the input and output structures for FastChem
    input_data = pyfastchem.FastChemInput()
    output_data = pyfastchem.FastChemOutput()
    
    if mode == 'cond':
        #use equilibrium condensation
        input_data.equilibrium_condensation = True
    elif mode == 'rainout':
        #this would turn on the rainout condensation approach
        input_data.rainout_condensation = True
    
    fastchem.setParameter('accuracyChem', 1e-5)


    solar_abundances = np.array(fastchem.getElementAbundances())

    #we need to know the indices for O and C from FastChem
    index_C = fastchem.getElementIndex('C')
    index_O = fastchem.getElementIndex('O')
    
    element_abundances = np.copy(solar_abundances)
  
    #set the C abundance as a function of the C/O ratio
    # chem.CO = 0.9
    metallicity = np.copy(chem.FeH)
    
    print(f'C/O = {chem.CO:.2f}, Fe/H = {metallicity:.2f}')
    element_abundances[index_C] = element_abundances[index_O] * chem.CO
    #scale the element abundances, except those of H and He
    for element in range(0, fastchem.getElementNumber()):
        if fastchem.getElementSymbol(element) != 'H' and fastchem.getElementSymbol(element) != 'He':
            element_abundances[element] *= 10**(metallicity)

    fastchem.setElementAbundances(element_abundances)
    
    input_data.temperature = PT.temperature_envelopes[3]
    input_data.pressure = PT.pressure   
    #run FastChem on the entire p-T structure
    fastchem_flag = fastchem.calcDensities(input_data, output_data)

    assert np.amin(output_data.element_conserved[:]) == 1, "Element conservation failed"

    #convert the output into a numpy array
    number_densities = np.array(output_data.number_densities)
    number_densities_cond = np.array(output_data.number_densities_cond)
    
    #check the species we want to plot and get their indices from FastChem
    plot_species_indices = []
    plot_species_symbols = []

    for idx, species in enumerate(plot_species_labels):
        index = fastchem.getGasSpeciesIndex(species[1])

        if index != pyfastchem.FASTCHEM_UNKNOWN_SPECIES:
            plot_species_indices.append(index)
            plot_species_symbols.append(plot_species_labels[idx])
        else:
            print("Species", species, "to plot not found in FastChem")


    #convert the output into a numpy array
    number_densities = np.array(output_data.number_densities)


    #total gas particle number density from the ideal gas law 
    #used later to convert the number densities to mixing ratios
    gas_number_density = PT.pressure*1e6 / (const.k_B.cgs * PT.temperature_envelopes[3])
    
    for s_i in range(0, len(plot_species_symbols)):
        
        VMR_fc = number_densities[:, plot_species_indices[s_i]]/gas_number_density
        if s_i == 0:
            VMR_fc_12CO = np.copy(VMR_fc)
            
        ax_chem.plot(VMR_fc / VMR_fc_12CO, PT.pressure,
                     lw=2.5, ls='--', color=colors_list[s_i])
        
        

    alpha_0 = af.weigh_alpha(icf, PT.pressure, np.ones_like(PT.pressure), ax=ax_chem, plot=True, alpha_min=alpha_min,
                                T_max=1e2)
    ax_chem.set(xscale='log', xlim=(5e-7, 10.0), ylim=(PT.pressure.max(), PT.pressure.min()))
                
    if i == 0:
        ax_chem.legend(ncol=len(plot_species_indices)//2, loc=(0.04, 1.01), 
                       frameon=False, prop={'size': 14})
    # remove xticks from ax_chem
    ax_chem.set_xticks([], minor=True)
    ax_chem.set_xticks([], minor=False)

    ax_chem_twin = ax_chem.twinx().twiny()
    ax_chem_twin.set(xscale='log', xlim=ax_chem.get_xlim(),
                    ylim=ax_chem.get_ylim())
    ax_chem_twin.tick_params(which='both', pad=6)

    # place xticks and xlabel at the bottom
    # disable minor xticks
    ax_chem_twin.xaxis.set_minor_locator(plt.NullLocator())
    # set yticks labels to invisible
    ax_chem_twin.xaxis.set_ticks_position('both')
    if (i+1) == len(targets):
        ax_chem_twin.set_xlabel(r'VMR(X$_i$) / VMR ($^{12}$CO)', labelpad=5,)
    else:
        ax_chem_twin.xaxis.set_ticklabels([])
        
    ax_chem_twiny = ax_chem.twinx()
    ax_chem_twiny.set(yscale='log', ylim=ax_chem.get_ylim(),
                    xlim=ax_chem.get_xlim())
    
    ax_chem_twiny.yaxis.set_ticks_position('both')
    ax_chem_twiny.yaxis.set_ticklabels([])
    
    ax_chem_twin.yaxis.set_ticks([])
    ax_chem_twin.yaxis.set_ticklabels([])
    ax_chem_twin.xaxis.set_ticks_position('both')
    ax_chem_twin.xaxis.tick_bottom()
    ax_chem_twin.xaxis.set_label_position('bottom')
            
    xlabel = 'Temperature (K)' if (i+1) == len(targets) else ''
    ylabel = 'Pressure (bar)' if i == 1 else ''
    # set xticks labels to
    # xticks = [1000, 2000, 3000, 4000]
    xticks = [2000, 4000]
    xticks_labels = [f'{int(x/1000)}' for x in xticks]
    ax_PT.xaxis.set_ticks(xticks)
    # ax_PT.xaxis.set_ticklabels(xticks_labels)
    ax_PT.set(ylim=(PT.pressure.max(), PT.pressure.min()), 
              ylabel=ylabel, 
              xlabel=xlabel,
              xlim=(1000,5000),
            yscale='log')
    # disable xticks on ax_PT
    # ax_PT.set_xticks([], minor=True)
    # ax_PT.set_xticks([], minor=False)
    
    ax_PT_twinx = ax_PT.twinx()
    ax_PT_twinx.set(yscale='log', ylim=ax_PT.get_ylim(),
                    xlim=ax_PT.get_xlim())
    ax_PT_twinx.yaxis.set_ticks_position('both')
    ax_PT_twinx.yaxis.set_ticklabels([])
    
    ax_PT_twiny = ax_PT.twiny() 
    ax_PT_twiny.set(xlim=ax_PT.get_xlim(),
                    ylim=ax_PT.get_ylim())
    ax_PT_twiny.xaxis.set_ticks(xticks)
    # ax_PT_twiny.xaxis.set_ticklabels(xticks_labels)
    ax_PT_twiny.xaxis.set_ticks_position('bottom')
    # remove text from xticks
    ax_PT_twiny.xaxis.set_ticklabels([])
    # no handle, only text 
    # ax_PT.legend(frameon=False, prop={'weight':'bold', 'size': 16}, loc='upper right')
    ax_PT.text(s=target, x=0.90, y=0.84, transform=ax_PT.transAxes, weight='bold', fontsize=16, zorder=10,
               ha='right', va='center')

    ax_PT.minorticks_off()


save = True
if save:
    fig.savefig(out_path / f'fig_eq_chem_{mode}.pdf', bbox_inches='tight', dpi=300)
    print(f'Saved figure in {out_path / f"fig_eq_chem_{mode}.pdf"}')
    plt.close(fig)
else:
    plt.show()