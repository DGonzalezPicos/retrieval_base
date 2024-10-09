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
target = 'gl880'

if target not in os.getcwd():
    os.chdir(base_path + target)
    
outputs = pathlib.Path(base_path) / target / 'retrieval_outputs'
config_file = 'config_freechem.txt'

# print(FIXMEEEE)
# run 16 was fitting for `resolution`, `log_g`, `Z` --> not useful for comparison with run 17
# run 18 is now running identical to run 17 but with fixed resolution=69k
runs_id = [17,18]
runs = [f'sphinx{r}' for r in runs_id]
fs= 14
# colors = {'sphinx16': 'r', 'sphinx17': 'b'}
colors = ['indianred', 'royalblue']

ignore_params = ['log_g', 'Z', 'resolution']
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
    
    fig = corner.corner(samples, labels=labels, color=colors[r],
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
                        color=colors[r],
                        weight='normal')
        
        
runs_label = '_'.join(runs)
fig_name = base_path + 'paper/latex/figures/corner_' + target + '_' + runs_label + '.pdf'
fig.savefig(fig_name)
print(f'Figure saved as {fig_name}')
plt.close(fig)