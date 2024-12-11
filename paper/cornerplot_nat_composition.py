from retrieval_base.retrieval import Retrieval
import retrieval_base.figures as figs
from retrieval_base.config import Config
from retrieval_base.auxiliary_functions import pickle_load, pickle_save
# import config_freechem as conf
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import corner

# reset to default
plt.style.use('default')
# plt.style.use(['latex-sans'])
plt.style.use(['sans'])
# enable latex
# plt.rcParams['text.usetex'] = True
plt.rcParams.update({
    "font.size": 8,
})

base_path = '/home/dario/phd/retrieval_base/'
nat_path = '/home/dario/phd/nat/figures/'

# target = 'gl205'
target = 'gl699'

if target not in os.getcwd():
    os.chdir(base_path + target)
    
outputs = pathlib.Path(base_path) / target / 'retrieval_outputs'
config_file = 'config_freechem.txt'

run ='fc5'
fs =14
color = 'brown'

mode = 'composition'
params_dict = dict(composition=['alpha_12CO',
                'alpha_H2O',
                'alpha_Na',
                'alpha_Ca',
                'alpha_Ti',
                'alpha_Mg',
                'alpha_Fe',
                'alpha_OH',
                'alpha_HF',
                'alpha_CN',
                'log_Sc',
                'log_12CO/13CO',
                'log_12CO/C18O',
                'log_12CO/C17O',
                'log_H2O/H2O_181',
])
my_params = params_dict[mode]
posterior_samples_file = outputs / run / outputs / f'posterior_samples.npy'
posterior_labels_file = outputs / run / outputs / f'posterior_labels.npy'
Chem_file = outputs / run / 'test_data/bestfit_Chem.pkl'

conf = Config(path=base_path, target=target, run=run)(config_file)


cache = True
if all([posterior_samples_file.exists(), posterior_labels_file.exists(), Chem_file]) and cache:
    samples = np.load(posterior_samples_file)
    labels  = np.load(posterior_labels_file)
    
else:
    ret = Retrieval(
                conf=conf, 
                evaluation=False,
                )

    _, samples = ret.PMN_analyze()
    # samples = samples.T
    ret.get_PT_mf_envelopes(samples)
    ret.Chem.get_VMRs_posterior()
    pickle_save(Chem_file, ret.Chem)

    labels = list(ret.Param.param_keys) # change to math mode
    # save samples and labels
    np.save(posterior_samples_file, samples)
    np.save(posterior_labels_file, labels)
    
posterior_dict = dict(zip(labels, samples.T))
samples = np.array([posterior_dict[label] for label in my_params]).T
labels = my_params
latex_labels = {k:conf.free_params[k][-1] for k in labels}
if 'alpha_H2O' in labels:
    latex_labels['alpha_H2O'] = r'$\alpha(\mathrm{H_2^{16}O})$'

assert samples.shape[-1] == len(labels), f'{samples.shape[-1]} != {len(labels)}'

pad_factor = 0.1
p = 0.001
lims = [(np.percentile(samples[:,i], p), np.percentile(samples[:,i], 100-p)) for i in range(samples.shape[-1])]
pad = [(lims[i][1] - lims[i][0]) * pad_factor for i in range(samples.shape[-1])]
lims = [(lims[i][0] - pad[i], lims[i][1] + pad[i]) for i in range(samples.shape[-1])]

fig = plt.figure(figsize=(20, 22))


fig = corner.corner(samples, 
                    labels=list(latex_labels.values()),
                    color=color,
                    range=lims,
                    quantiles=[0.16, 0.5, 0.84],
                    show_titles=True, title_kwargs={"fontsize": fs},
                    fontsize=fs,
                    plot_density=True,
                    plot_datapoints=False,
                    plot_contours=True,
                    fill_contours=True,
                    smooth=1.0, 
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
    fig.axes[i].xaxis.label.set_fontsize(0)
    fig.axes[i].yaxis.label.set_fontsize(0)
    # fig.axes[i].xaxis.label.labelpad(20)
    xlabel_pos = fig.axes[i].xaxis.label.get_position()
    fig.axes[i].xaxis.label.set_position((xlabel_pos[0], xlabel_pos[1]-0.07))
    
    ylabel_pos = fig.axes[i].yaxis.label.get_position()
    fig.axes[i].yaxis.label.set_position((ylabel_pos[0]-0.15, ylabel_pos[1]))
    # increse fontsize of tick labels
    fig.axes[i].tick_params(axis='both', which='major', labelsize=fs)
    fig.axes[i].tick_params(axis='both', which='minor', labelsize=fs)
    # increase padding of xlabels

    
    
# add new titles

for j, title in enumerate(titles):
    if title == '':
        continue
    
    # first only the name of the parameter
    s = title.split('=')
    # fig.axes[j].text(0.5, 1.30, title, fontsize=22,
    y0 = 1.25
    r = 0
    fig.axes[j].text(0.5, y0, s[0], fontsize=fs,
                    ha='center', va='bottom',
                    transform=fig.axes[j].transAxes,
                    color='k',
                    weight='normal')
    fig.axes[j].text(0.5, y0-(0.22*(r+1)), s[1], fontsize=fs,
                            ha='center', va='bottom',
                            transform=fig.axes[j].transAxes,
                            color='k',
                            weight='normal')
    
    
# add VMR plot
Chem = pickle_load(outputs / run / 'test_data/bestfit_Chem.pkl')
PT = pickle_load(outputs / run / 'test_data/bestfit_PT.pkl')

assert hasattr(PT, 'temperature_envelopes'), 'No temperature envelopes found'

ax_chem = fig.add_axes([0.58,0.64,0.38,0.30])

species_to_plot = ['12CO',
 '13CO',
 'C18O',
#  'C17O',
 'H2O',
#  'H2O_181',
 'HF',
 'Na',
#  'Ca',
 'Ti',
#  'Mg',
#  'Fe',
 'Sc',
 'OH',
 'CN',
 ] 
ax_chem = figs.fig_VMR(Chem, ax=ax_chem, fig=fig, pressure=Chem.pressure, species_to_plot=species_to_plot,
                       xlim=(1e-10, 1e-3),
                       leg_kwargs={'fontsize': fs*1.3, 'loc': (-0.3, 0.40), 'ncol': 1},
)
# update fs of ax_chem labels and ticks to fs
ax_chem.tick_params(axis='both', which='major', labelsize=fs)
ax_chem.tick_params(axis='both', which='minor', labelsize=fs)
ax_chem.xaxis.label.set_fontsize(fs)
ax_chem.yaxis.label.set_fontsize(fs)


    
# set margins for the figure
plt.subplots_adjust(left=0.035, right=0.99, top=0.97, bottom=0.03)

fig_name = nat_path + 'corner_' + target + '.pdf'
fig.savefig(fig_name)
print(f' Saved {fig_name}')
plt.close(fig)