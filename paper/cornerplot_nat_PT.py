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

target = 'gl205'
# target = 'gl699'

if target not in os.getcwd():
    os.chdir(base_path + target)
    
outputs = pathlib.Path(base_path) / target / 'retrieval_outputs'
config_file = 'config_freechem.txt'

run ='fc5'
fs =14
color = 'brown'

mode = 'PT'
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
                ],
                PT=['log_g',
                    'vsini', 'rv', 'T_0', 'log_P_RCE', 'dlog_P_1', 
                    'dlog_P_3', 'dlnT_dlnP_0', 'dlnT_dlnP_1', 
                    'dlnT_dlnP_RCE', 'dlnT_dlnP_2', 
                    'dlnT_dlnP_3', 'dlnT_dlnP_4', 'dlnT_dlnP_5'
                ]
)
my_params = params_dict[mode]
posterior_samples_file = outputs / run / outputs / f'posterior_samples.npy'
posterior_labels_file = outputs / run / outputs / f'posterior_labels.npy'
PT_file = outputs / run / 'test_data/bestfit_PT.pkl'

conf = Config(path=base_path, target=target, run=run)(config_file)


cache = True
if all([posterior_samples_file.exists(), posterior_labels_file.exists(), PT_file]) and cache:
    samples = np.load(posterior_samples_file)
    labels  = np.load(posterior_labels_file)
    
else:
    ret = Retrieval(
                conf=conf, 
                evaluation=False,
                )

    _, samples = ret.PMN_analyze()
    # samples = samples.T
    # ret.get_PT_mf_envelopes(samples)
    # ret.Chem.get_VMRs_posterior()
    # pickle_save(Chem_file, ret.Chem)

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
PT = pickle_load(outputs / run / 'test_data/bestfit_PT.pkl')
# Chem = ret.Chem
# Chem.get_VMRs_posterior(profile_id=30)

# ax_PT = fig.add_axes([0.58,0.64,0.38,0.30])
ax_PT = fig.add_axes([0.52,0.66,0.28,0.28])
l, b, w, h = ax_PT.get_position().bounds
ax_grad = fig.add_axes([l+w+0.012, b, 0.14, h])

figs.fig_PT(
            PT=PT, 
            ax=ax_PT, 
            ax_grad=ax_grad,
            fig=fig,
            bestfit_color=color,
            envelopes_color=color,
            int_contr_em_color='k',
            # text_color='gray',
            show_photosphere=False,
            show_knots=True,
            # show_text=False,
            xlim=(1400, 6400), # fix view
            # xlim_grad=(-0.02, 0.34),
            # fig_name=self.prefix+f'plots/PT_grad_profile.pdf',
        )
# update fs of ax_PT labels and ticks to fs
axes = [ax_PT, ax_grad]
for axi in axes:
    axi.tick_params(axis='both', which='major', labelsize=fs)
    axi.tick_params(axis='both', which='minor', labelsize=fs)
    axi.xaxis.label.set_fontsize(fs)
    axi.yaxis.label.set_fontsize(fs)


    
# set margins for the figure
plt.subplots_adjust(left=0.035, right=0.99, top=0.97, bottom=0.03)

fig_name = nat_path + 'corner_' + target + f'_{mode}.pdf'
fig.savefig(fig_name)
print(f' Saved {fig_name}')
plt.close(fig)