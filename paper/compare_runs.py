from retrieval_base.retrieval import Retrieval
import retrieval_base.figures as figs
from retrieval_base.config import Config
# import config_freechem as conf
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import corner

base_path = '/home/dario/phd/retrieval_base/'
target = 'gl1151'

if target not in os.getcwd():
    os.chdir(base_path + target)
    
outputs = pathlib.Path(base_path) / target / 'retrieval_outputs'
config_file = 'config_freechem.txt'

# print(FIXMEEEE)
# run 16 was fitting for `resolution`, `log_g`, `Z` --> not useful for comparison with run 17
# run 18 is now running identical to run 17 but with fixed resolution=69k
runs_dict = {
    # 'sphinx1':('SPHINX nl=40', 'darkorange'),
        # 'fc1':('FastChem nl=40', 'indianred'),
        'fc1':('FastChem 1', 'royalblue'),
        'fc2':('FastChem 2', 'forestgreen'),
}
runs = list(runs_dict.keys())
legend_labels = [v[0] for v in runs_dict.values()]
colors = {run: v[1] for run, v in runs_dict.items()}

fs= 14


ignore_params = ['Z', 'resolution']
# ignore all params starting with dln
RCE_params = ['T_0', 'log_P_RCE', 'dlog_P_1', 'dlog_P_3',
              'dlnT_dlnP_RCE']
RCE_params += [f'dlnT_dlnP_{i}' for i in range(6)]
if any(['sphinx' in run for run in runs]):
    ignore_params += RCE_params
    ignore_params += ['log_HF', 'alpha_HF']
    ignore_params += ['log_CN', 'alpha_CN']
    ignore_params += ['Teff']
    
    
# ignore_params += RCE_params
PT_list, C_ratio_list = [], []

for r, run in enumerate(runs):

    conf = Config(path=base_path, target=target, run=run)(config_file)

    ret = Retrieval(
                conf=conf, 
                evaluation=False,
                )

    bestfit_params, posterior = ret.PMN_analyze()
    # samples = samples.T
    
    labels = list(ret.Param.param_keys) # change to math mode
    posterior_dict = dict(zip(labels, posterior.T))
    
    samples = np.array([posterior_dict[label] for label in labels if label not in ignore_params]).T
    labels = [label for label in labels if label not in ignore_params]
    assert samples.shape[-1] == len(labels), f'{samples.shape[-1]} != {len(labels)}'
    print(samples.shape)
    print(labels)
    C_ratio_list.append(10.0**posterior_dict['log_12CO/13CO'])
    
    ret.get_PT_mf_envelopes(posterior)
    PT_list.append(ret.PT)

    
    pad_factor = 0.1
    p = 0.001
    lims = [(np.percentile(samples[:,i], p), np.percentile(samples[:,i], 100-p)) for i in range(samples.shape[-1])]
    pad = [(lims[i][1] - lims[i][0]) * pad_factor for i in range(samples.shape[-1])]
    lims = [(lims[i][0] - pad[i], lims[i][1] + pad[i]) for i in range(samples.shape[-1])]
    if r == 0:
    # fig = 
        # fig = plt.figure(figsize=(24, 28))
        fig = None
        shape_0 = samples.shape[-1]
    assert shape_0 == samples.shape[-1], f'{shape_0} != {samples.shape[-1]}'
    
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
    # apply tight layout
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
        # fig.axes[i].xaxis.label.labelpad(20)
        xlabel_pos = fig.axes[i].xaxis.label.get_position()
        fig.axes[i].xaxis.label.set_position((xlabel_pos[0], xlabel_pos[1]-0.07))
        
        ylabel_pos = fig.axes[i].yaxis.label.get_position()
        fig.axes[i].yaxis.label.set_position((ylabel_pos[0]-0.15, ylabel_pos[1]))
        # increse fontsize of tick labels
        fig.axes[i].tick_params(axis='both', which='major', labelsize=20)
        fig.axes[i].tick_params(axis='both', which='minor', labelsize=20)
        # increase padding of xlabels

        
        
    # add new titles
    
    for j, title in enumerate(titles):
        if title == '':
            continue
        
        # first only the name of the parameter
        s = title.split('=')
        # fig.axes[j].text(0.5, 1.30, title, fontsize=22,
        y0 = 1.35
        if r == 0:
            fig.axes[j].text(0.5, y0, s[0], fontsize=fs,
                            ha='center', va='bottom',
                            transform=fig.axes[j].transAxes,
                            color='k',
                            weight='normal')
        fig.axes[j].text(0.5, y0-(0.15*(r+1)), s[1], fontsize=fs,
                        ha='center', va='bottom',
                        transform=fig.axes[j].transAxes,
                        color=colors[run],
                        weight='normal')
        
        
  
ax_PT = fig.add_axes([0.52,0.56,0.2,0.24])
l, b, w, h = ax_PT.get_position().bounds
ax_grad = fig.add_axes([l+w+0.03, b, 0.1, h])

ax_C_ratio = fig.add_axes([0.52,0.82,0.30+0.03,0.14])



# plot two histograms
hist_kwargs = {"color": colors[run], "alpha": 0.65, "fill": True, "edgecolor": "k",
                        "linewidth": 2.0, "histtype": "stepfilled", "density": True,}
hist_kwargs_edge = hist_kwargs.copy()
hist_kwargs_edge['fill'] = False
    
for PT, C_ratio, run in zip(PT_list, C_ratio_list, runs):
    print(f' Plotting {run}...')
    
    figs.fig_PT(
            PT=PT, 
            ax=ax_PT, 
            ax_grad=ax_grad,
            fig=fig,
            bestfit_color=colors[run],
            envelopes_color=colors[run],
            int_contr_em_color=colors[run],
            # text_color='gray',
            show_photosphere=True,
            show_knots=False,
            # show_text=False,
            xlim=(2000, 7000), # fix view
            # xlim_grad=(-0.02, 0.34),
            # fig_name=self.prefix+f'plots/PT_grad_profile.pdf',
        )
    
    ax_PT.set_ylim(1e2, 1e-5)
    ax_grad.set_ylim(1e2, 1e-5)
    # remove ylabel from ax_grad and yticks
    ax_grad.set_ylabel('')
    ax_grad.set_yticks([])
    ax_grad.set_xlim(-0.05, 0.50)
    
    hist_kwargs = {"color": colors[run], "alpha": 0.65, "fill": True, "edgecolor": "k",
                        "linewidth": 2.0, "histtype": "stepfilled", "density": True,}
    hist_kwargs_edge = hist_kwargs.copy()
    hist_kwargs_edge['fill'] = False
    ax_C_ratio.hist(C_ratio, bins=30, **hist_kwargs)
    ax_C_ratio.hist(C_ratio, bins=30, **hist_kwargs_edge)
    C_ratio_q = np.percentile(C_ratio, [16, 50, 84])
    print(f' C_ratio_q = {C_ratio_q[1]:.1f} + {C_ratio_q[2] - C_ratio_q[1]:.1f} - {C_ratio_q[1] - C_ratio_q[0]:.1f}')
        

ax_C_ratio.axvline(89.0, color='magenta', ls='--', lw=3, alpha=0.7)
ax_C_ratio.set_xlabel(r'$^{12}$C/$^{13}$C', fontsize=fs*1.4)

ISM = (68, 15)
ax_C_ratio.axvspan(ISM[0]-ISM[1], ISM[0]+ISM[1], color='gray', alpha=0.15)
        
# add custom legend
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[run], markersize=10) for run in runs]
fig.legend(handles, legend_labels, loc='upper right', fontsize=fs*2, title='Runs', title_fontsize=fs*2)
fig.subplots_adjust(top=0.97, right=0.97, left=0.05, bottom=0.05)
        
runs_label = '_'.join(runs)
fig_name = base_path + 'paper/latex/figures/corner_' + target + '_' + runs_label + '.pdf'
fig.savefig(fig_name)
print(f'Figure saved as {fig_name}')
plt.close(fig)