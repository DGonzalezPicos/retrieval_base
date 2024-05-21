from retrieval_base.retrieval import pre_processing, Retrieval
from retrieval_base.parameters import Parameters
from retrieval_base.chemistry import Chemistry
from retrieval_base.config import Config
import retrieval_base.figures as figs
atomic_mass = {k:v[2] for k,v in Chemistry.species_info.items()}

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('/home/dario/phd/retrieval_base/HBDs/my_science.mplstyle')
# set fontsize to 16
fs = 26
plt.rcParams.update({'font.size': fs})

import pathlib
import pickle
import corner
import pandas as pd
import json


path = pathlib.Path('/home/dario/phd/retrieval_base')
# out_path = path / 'HBDs'
out_path = pathlib.Path('/home/dario/phd/Hot_Brown_Dwarfs_Retrievals/figures/')

target = 'J0856'
runs = ['final_full', 
        'final_noGP',
        ]
colors = dict(
            final_full='indianred', 
            final_noGP='#5ccdcd',
            )
config_file = 'config_freechem.txt'

path = pathlib.Path('/home/dario/phd/retrieval_base')
data_path = path / f'{target}'
print(data_path)
PT_list, CO_list, C_ratio_list = [], [], []

for r, run in enumerate(runs):
    
    # bestfit_params = 
    retrieval_path = data_path / f'retrieval_outputs/{run}'
    assert retrieval_path.exists(), f'Retrieval path {retrieval_path} does not exist.'
    
    # load json file with bestfit parameters
    with open(retrieval_path / 'test_data/bestfit.json', 'r') as f:
        bestfit_params = json.load(f)
        
    conf = Config(path=path, target=target, run=run)(config_file)
    PT = pickle.load(open(retrieval_path / 'test_data/bestfit_PT.pkl', 'rb'))
    equal_weighted_file = retrieval_path / 'test_post_equal_weights.dat'
    posterior = np.loadtxt(equal_weighted_file)
    posterior = posterior[:,:-1]
    # create dict with keys from conf.free_params.keys() and values from posterior
    posterior_dict = {k: posterior[:,i] for i, k in enumerate(conf.free_params.keys())}

    params = bestfit_params['params']
    chem = pickle.load(open(retrieval_path / 'test_data/bestfit_Chem.pkl', 'rb'))
    
    exclude_keys = ['dlnT_dlnP_0', 'dlnT_dlnP_1', 'dlnT_dlnP_2', 'dlnT_dlnP_3', 'dlnT_dlnP_4', 
                    'dlnT_dlnP_5', 'dlnT_dlnP_6', 'dlnT_dlnP_7', 'dlog_P', 'T_0',
                    'log_a', 'log_l']

    
    # # samples = np.array([posterior_dict[k] for k in posterior_dict.keys() if k not in exclude_keys]).T
    # labels = [v[-1] for k,v in conf.free_params.items() if k not in exclude_keys]
    
    sort_keys = ['log_12CO', 'log_H2O', 'log_13CO', 'log_H2O_181', 'log_HF', 'log_Na','log_Ca','log_Ti',
                 'log_g', 'vsini', 'epsilon_limb', 'rv', 'alpha', 'beta', ]
    samples = np.array([posterior_dict[k] for k in sort_keys]).T
    
    
    labels = [conf.free_params[k][-1] for k in sort_keys]

    # replace keys
    replace = {
        '$\Delta\log\ P$' : '$\log\Delta P$',
        '$\log\ \mathrm{H_2O}$' : '$\log\ \mathrm{H_2^{16}O}$',
        '$\alpha$' : '$r_0$',
        '$\beta$' : '$\\alpha$',
                }
    labels = [replace.get(l, l) for l in labels]
    # print(stop)
    assert len(labels) == samples.shape[-1], f'Number of labels = {len(labels)} does not match number of samples = {samples.shape[-1]}'
    
    print(f'Number of samples = {samples.shape[0]}')

    

    pad_factor = 0.05
    # lims= [(np.min(samples[:,i]), np.max(samples[:,i])) for i in range(samples.shape[-1])]
    p = 0.05
    lims = [(np.percentile(samples[:,i], p), np.percentile(samples[:,i], 100-p)) for i in range(samples.shape[-1])]
    pad = [(lims[i][1] - lims[i][0]) * pad_factor for i in range(samples.shape[-1])]
    lims = [(lims[i][0] - pad[i], lims[i][1] + pad[i]) for i in range(samples.shape[-1])]
    # lims[2] = (0, 200)
    if r == 0:
        fig = None

    fig = corner.corner(samples, labels=labels, color=colors[run],
                        range=lims if r == 0 else None,
                        quantiles=[0.16, 0.5, 0.84],
                        show_titles=True, title_kwargs={"fontsize": fs},
                        fontsize=fs,
                        plot_density=True,
                        plot_datapoints=False,
                        plot_contours=True,
                        fill_contours=True,
                        smooth=1.5, 
                        bins=30, 
                        max_n_ticks=3,
                        hist_kwargs={'density': False,
                                     'fill': True,
                                     'alpha': 0.7,
                                     'edgecolor': 'k',
                                    'linewidth': 1.0,
                                     },
                        fig=fig,
                        )
    
    
    # collect all titles
    titles = [axi.title.get_text() for axi in fig.axes]
    for i, title in enumerate(titles):
        if len(title) > 30:
            title_split = title.split('=')
            titles[i] = title_split[0] + ' =\n' + title_split[1]
    # remove the original titles
    # if i ==1:
    for i, axi in enumerate(fig.axes):
        fig.axes[i].title.set_visible(False)
        # increase fontsize of x and y labels
        fig.axes[i].xaxis.label.set_fontsize(fs)
        fig.axes[i].yaxis.label.set_fontsize(fs)
        # increse fontsize of tick labels
        fig.axes[i].tick_params(axis='both', which='major', labelsize=20)
        fig.axes[i].tick_params(axis='both', which='minor', labelsize=20)
        
        
    # add new titles
    
    for j, title in enumerate(titles):
        if title == '':
            continue
        
        # first only the name of the parameter
        s = title.split('=')
        # fig.axes[j].text(0.5, 1.30, title, fontsize=22,
        if r == 0:
            fig.axes[j].text(0.5, 1.55, s[0], fontsize=fs,
                            ha='center', va='bottom',
                            transform=fig.axes[j].transAxes,
                            color='k',
                            weight='normal')
        fig.axes[j].text(0.5, 1.55-(0.25*(r+1)), s[1], fontsize=fs,
                        ha='center', va='bottom',
                        transform=fig.axes[j].transAxes,
                        color=colors[run],
                        weight='normal')
        
    CO_list.append(chem.VMRs_posterior['C/O'])
    C_ratio_list.append(chem.VMRs_posterior['12_13CO'])
    PT_list.append(PT)
    
        
    # if r==0:
ax_PT = fig.add_axes([0.66,0.50,0.3,0.28])
ax_CO = fig.add_axes([0.55,0.82,0.2,0.14])
l, b, w, h = ax_CO.get_position().bounds

ax_C_ratio = fig.add_axes([l+w+0.02, b, w, h])



# plot two histograms
hist_kwargs = {"color": colors[run], "alpha": 0.65, "fill": True, "edgecolor": "k",
                        "linewidth": 2.0, "histtype": "stepfilled", "density": True,}
hist_kwargs_edge = hist_kwargs.copy()
hist_kwargs_edge['fill'] = False
    
for PT, CO, C_ratio, run in zip(PT_list, CO_list, C_ratio_list, runs):
    # remove attribute of int_contr_em
    if run == 'final_noGP':
        delattr (PT, 'int_contr_em')
    figs.fig_PT(
            PT=PT, 
            ax=ax_PT, 
            ax_grad=None,
            fig=fig,
            bestfit_color=colors[run],
            envelopes_color=colors[run],
            int_contr_em_color='gray',
            # text_color='gray',
            # weigh_alpha=True,
            show_photosphere=False,
            show_knots=False,
            show_text=False,
            xlim=(1000, 5000), # fix view
            # xlim_grad=(-0.02, 0.34),
            # fig_name=self.prefix+f'plots/PT_grad_profile.pdf',
        )
    hist_kwargs['color'] = colors[run]
    ax_CO.hist(CO, bins=30, **hist_kwargs)
    ax_CO.hist(CO, bins=30, **hist_kwargs_edge)
    ax_C_ratio.hist(C_ratio, bins=30, **hist_kwargs)
    ax_C_ratio.hist(C_ratio, bins=30, **hist_kwargs_edge)
    
ax_CO.set(xlabel='C/O')
ax_C_ratio.set(xlabel=r'$^{12}$C/$^{13}$C')
    
ax_list = [ax_CO, ax_C_ratio]
for axi in ax_list:
    axi.yaxis.set_visible(False)
    axi.xaxis.set_ticks_position('bottom')
    axi.spines['top'].set_visible(False)
    axi.spines['right'].set_visible(False)
    axi.spines['left'].set_visible(False)
        
ax_C_ratio.set_xticks(np.arange(30, 190+1, 30))
# increase padding of x-axis labels
ax_PT.tick_params(axis='x', pad=18)
        
for axi in fig.axes:
    for item in ([axi.title, axi.xaxis.label, axi.yaxis.label] +
                axi.get_xticklabels() + axi.get_yticklabels()):
        item.set_fontsize(fs)
    
# add custom legend with the labels and colors of the two runs
handles = [plt.Rectangle((0,0),1,1, color=colors[run]) for run in runs]
labels = ['GP', 'No GP']
ax_CO.legend(handles, labels, loc=(-0.5, 0.5), frameon=False, fontsize=fs*1.2)


    
        
fig.subplots_adjust(top=0.95)
save= True
if save:
    fig.savefig(out_path / f'cornerplot_{target}_GP.pdf', bbox_inches='tight', dpi=300)   
    print(f'Saved {out_path / f"cornerplot_{target}_GP.pdf"}')
    plt.close() 