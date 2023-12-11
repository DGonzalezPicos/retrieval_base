from retrieval_base.retrieval import pre_processing, Retrieval
from retrieval_base.parameters import Parameters

import numpy as np
import matplotlib.pyplot as plt
# set fontsize to 16
plt.rcParams.update({'font.size': 16})
import pathlib
import pickle
import corner
import pandas as pd
import json

def create_legend(colors, ax, loc=(1.02, 0.0), ls=None):
    ''' Create a legend for the plot '''
    
    if ls is None:
        ls = dict(zip(colors.keys(), ['-']*len(colors)))
    # create a list of Line2D objects
    lines = []
    for name, color in colors.items():
        lines.append(plt.Line2D([0], [0], color=color, lw=2, label=name, ls=ls[name]))
        
    # create the legend
    ax.legend(lines, list(colors.keys()), loc=loc, bbox_to_anchor=loc)
    return ax

path = pathlib.Path('/home/dario/phd/retrieval_base')
out_path = path / 'HBDs'

targets = dict(J1200='freechem_4', TWA28='freechem_1', J0856='freechem_1')
colors = dict(J1200='royalblue', TWA28='seagreen', J0856='indianred')

prefix = 'freechem'
plot_PT = True
plot_chemistry = True

plot_vsini = False

df = pd.read_csv(out_path / 'dataframe10-k.csv')
species = df.name.str.split().str[0].unique()
# assign a unique color to each species (26)
colors_s = plt.cm.tab20(np.linspace(0, 1, species.size))
colors_s = dict(zip(species, colors_s))
ls = dict(zip(species, ['-', '--']*(len(species)//2)))

fig, ax = plt.subplots(len(targets)+1,1, figsize=(16,len(targets)*3.), sharex=True)
ax_transm = ax[-1].twinx()


if plot_PT:
    fig_PT, ax_PT = plt.subplots(1,1, figsize=(8,8))
    
# if plot_1213CO:
#     fig_1213CO, ax_1213CO = plt.subplots(1,1, figsize=(8,8))


order, det = 2,2

def wrapper(order,det, plot_chemistry=False, plot_PT=False):
    fig_corner = None
    handles = []

    for i, (target, retrieval_id) in enumerate(targets.items()):
        data_path = pathlib.Path('/home/dario/phd/retrieval_base') / f'{target}'
        print(data_path)
        
        
        
        # bestfit_params = 
        retrieval_path = data_path / f'retrieval_outputs/{retrieval_id}'
        assert retrieval_path.exists(), f'Retrieval path {retrieval_path} does not exist.'
        # m_spec = np.load(retrieval_path / 'test_data/bestfit_m_spec_K1266.pkl')
        m_spec = pickle.load(open(retrieval_path / 'test_data/bestfit_m_spec_K2166.pkl', 'rb'))
        d_spec = pickle.load(open(retrieval_path / 'test_data/d_spec_K2166.pkl', 'rb'))
        transm = np.load(retrieval_path / 'test_data/d_spec_transm_K2166.npy')
        loglike = pickle.load(open(retrieval_path / 'test_data/bestfit_LogLike_K2166.pkl', 'rb'))
        
        
        
        # load json file with bestfit parameters
        with open(retrieval_path / 'test_data/bestfit.json', 'r') as f:
            bestfit_params = json.load(f)
            
        equal_weighted_file = retrieval_path / 'test_post_equal_weights.dat'
        posterior = np.loadtxt(equal_weighted_file)
        posterior = posterior[:,:-1]
        
        params = bestfit_params['params']
        RV = params['rv']
        print(f'Posterior shape = {posterior.shape}')
        samples = dict(zip(params.keys(), posterior.T))
        print(samples)
        # TODO: plot log_g, vsini to the posterior
        
        # print(params.keys())
        # RV = bestfit_params['params']['RV']
        
        if plot_PT:
            PT = pickle.load(open(retrieval_path / 'test_data/bestfit_PT.pkl', 'rb'))
            ax_PT.fill_betweenx(PT.pressure, PT.temperature_envelopes[0], PT.temperature_envelopes[-1], color=colors[target], alpha=0.2)
            ax_PT.fill_betweenx(PT.pressure, PT.temperature_envelopes[1], PT.temperature_envelopes[-2], color=colors[target], alpha=0.4)
            ax_PT.plot(PT.temperature, PT.pressure, color=colors[target], lw=2.5, label=target)
            
            if i == 0:
                # plot integrated contribution function
                icf = np.load(retrieval_path / 'test_data/bestfit_int_contr_em_K2166.npy')
                print(f'shape of icf = {icf.shape}')
                ax_icf = ax_PT.twiny()
                ax_icf.plot(icf, PT.pressure, color='k', lw=2.5, label='ICF', ls='--', alpha=0.7)
                ax_icf.set(xlim=(0, 1.1*icf.max()), xlabel='Contribution function')
                
            if i == len(targets)-1:
                ax_PT.set(ylim=(PT.pressure.max(), PT.pressure.min()), ylabel='Pressure [bar]', xlabel='Temperature [K]',
                        yscale='log')
                ax_PT.legend(frameon=False, fontsize=24)
                fig_PT.savefig(out_path / f'plots/bestfit_{prefix}_PT.png', bbox_inches='tight', dpi=300)
            
        if plot_chemistry:
            
            
            chem = pickle.load(open(retrieval_path / 'test_data/bestfit_Chem.pkl', 'rb'))
            print(chem.__dict__.keys())
            # cornerplot with C/O, Fe/H and 12C/13C
            chem.C12C13_posterior = np.median(chem.mass_fractions_posterior['CO_high'] / chem.mass_fractions_posterior['CO_36_high'],axis=-1)
            
            samples = np.array([chem.CO_posterior, chem.FeH_posterior, chem.C12C13_posterior]).T
            
            # Define the arguments for the hist function, make them filled and thick edge black
            hist_args = {"color": colors[target], "alpha": 0.4, "fill": True, "edgecolor": "k",
                         "linewidth": 1.5, "histtype": "stepfilled"}
            limits = [(0.50, 0.68), (-1.0, 1.0), (1, 200)]  # replace with your actual limits
            
            
            fig_corner = corner.corner(samples, labels=['C/O', 'Fe/H', '12C/13C'], 
                          quantiles=[0.5], 
                          show_titles=False, 
                          color=colors[target],
                          bins=20,
                          alpha=0.4,
                          plot_density=True,
                        plot_contours=True,
                          fig=fig_corner,
                          hist_kwargs=hist_args,
                          range=limits,
                          smooth=1.0,)
            # create custom handle with the corresponding name and color for legend
            handles.append(plt.Line2D([0], [0], color=colors[target], lw=1.5, label=target))
            if i == len(targets)-1:
                # overplot line showing solar system values on the corner plot
                solar_system = {'C/O': 0.54, 'Fe/H': 0.0, '12C/13C': 89}
                corner.overplot_lines(fig_corner, list(solar_system.values()), ls='-', color='magenta', lw=1.5)
                handles.append(plt.Line2D([0], [0], color='magenta', lw=1.5, label='Solar'))
                # overplot the ISM 12C/13C value
                corner.overplot_lines(fig_corner, [None, None, 68.], ls='--', color='orange', lw=1.5)
                handles.append(plt.Line2D([0], [0], color='orange', lw=1.5, ls='--', label='ISM'))
                # add custom legend for the targets and the solar system
                fig_corner.legend(ncol=2, handles=handles, frameon=False, fontsize=18, 
                                  loc='upper right')
                
                
                
                fig_corner.savefig(out_path / f'plots/cornerplot_{prefix}_chemistry.png', bbox_inches='tight', dpi=300)
            
            
            
        
        x = d_spec.wave[order,det]
        sample_rate = np.mean(np.diff(x))
        # print(f'sample rate = {sample_rate:.3e} nm')
        y = d_spec.flux[order,det]
        err = d_spec.err[order,det] * loglike.beta[order,det,None]
        median_err = np.nanmedian(err)

        m = m_spec.flux[order,det] * loglike.f[order,det,None]
        ax[i].plot(x, y, lw=1.5, label='data', color='k')
        ax[i].fill_between(x, y-err, y+err, color='k', alpha=0.2)
        ax[i].plot(x, m, lw=2.5, label='model', ls='-', color=colors[target])
        ax[i].set(xlim=(x.min(), x.max()))
        if i == 0:
            ax[i].set_ylabel('Flux\n[erg/s/cm$^2$/nm]')
        ax[i].text(0.92, 0.85, target, transform=ax[i].transAxes, fontsize=18, weight='bold')

        res = y - m
        # shift residuals to rest-frame 
        res_shift = np.interp(x, x*(1-RV/2.998e5), res)
        
        # ax[-1].plot(x, res, lw=1., label='residuals', color=colors[target], alpha=0.8)
        ax[-1].plot(x, res_shift, lw=1., label='residuals (shifted)', color=colors[target], alpha=0.8)
    
    # print('Creating transmission plot')
    ax_transm.plot(x, transm[order,det], lw=2., label='transmission', color='k', alpha=0.1)

        # select species within wavelength range
    # select lines within the wavelength range
    lines = df[(df.wavelength >= x.min()) & (df.wavelength <= x.max())]
    # sort lines by equivalent_width
    # lines = lines.sort_values(by='equivalent_width', ascending=False)
    # # select only the first 50 lines
    # lines = lines.iloc[:100]
    # sort lines by absorptance
    lines = lines.sort_values(by='absorptance', ascending=False)
    # select only lines with absorptance > 0.05
    lines = lines[lines.absorptance > 0.03]
    # generate a range of alpha values proportional to absorptance
    alpha_lines = lines.absorptance / lines.absorptance.max()    
    
    species_in_range = []
    for j, line in enumerate(lines.itertuples()):
        ax[-1].axvline(line.wavelength, color=colors_s[line.name], alpha=alpha_lines.iloc[j], ls=ls[line.name])
        species_in_range.append(line.name)
        
    species_in_range = np.unique(species_in_range)
    # add manual legend on the right side outside plot
    # ax.legend(species, loc='center left', bbox_to_anchor=(1, 0.5))
    colors_in_range = {k: colors_s[k] for k in species_in_range}
    create_legend(colors_in_range, ax[-1], ls=ls)
        
        
        
        # ax[-1].axhspan(-median_err, median_err, color='k', alpha=0.2)
    ax[-1].axhline(0, ls='-', color='k', lw=1.5)
    ax[-1].set(xlabel='Wavelength [nm]', ylabel='Residuals\n[erg/s/cm$^2$/nm]')
    ax[-1].set_ylim(-2.1e-15, 2.1e-15)

    # ax[0].legend()
    plt.show()
    fig.savefig(out_path / f'plots/bestfit_{prefix}_spectra_order{order}_det{det}.png', bbox_inches='tight', dpi=300)
    
    # if plot_chemistry:
    #     return m_spec, d_spec, loglike, chem

    return m_spec, d_spec, loglike
    
orders = [0,1,2,3,4]
detectors = [0,1,2]

# orders = [1]
# detectors = [1]

for order in orders:
    for det in detectors:
        # m_spec, d_spec, loglike, chem = wrapper(order,det)
        plot_chemistry = True if (order+det) == 0 else False
        plot_PT = True if (order+det) == 0 else False
        m_spec, d_spec, loglike = wrapper(order,det, plot_chemistry=plot_chemistry, plot_PT=plot_PT)