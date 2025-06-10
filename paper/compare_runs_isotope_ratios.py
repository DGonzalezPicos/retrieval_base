from retrieval_base.retrieval import Retrieval
import retrieval_base.figures as figs
from retrieval_base.config import Config
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib

base_path = '/home/dario/phd/retrieval_base/'
target = 'gl880'

if target not in os.getcwd():
    os.chdir(base_path + target)
    
outputs = pathlib.Path(base_path) / target / 'retrieval_outputs'
config_file = 'config_freechem.txt'

runs_dict = {
    'fc5': ('Fiducial', 'royalblue'),
    'fc6': ('Non-isothermal priors', 'forestgreen'),
}
runs = list(runs_dict.keys())
legend_labels = [v[0] for v in runs_dict.values()]
colors = {run: v[1] for run, v in runs_dict.items()}

fs = 10

ignore_params = ['Z', 'resolution']
RCE_params = ['T_0', 'log_P_RCE', 'dlog_P_1', 'dlog_P_3', 'dlnT_dlnP_RCE']
RCE_params += [f'dlnT_dlnP_{i}' for i in range(6)]

if any(['sphinx' in run for run in runs]):
    ignore_params += RCE_params
    ignore_params += ['log_HF', 'alpha_HF']
    ignore_params += ['log_CN', 'alpha_CN']
    ignore_params += ['alpha_Si']
    ignore_params += ['log_12CO/C17O']
    ignore_params += ['alpha_Mg', 'log_Sc']
    ignore_params += ['Teff']

PT_list = []
C_ratio_list = []
O_ratio_list = []

# Process each run to get PT profiles and isotope ratios
for r, run in enumerate(runs):
    print(f'Processing run {run}...')
    
    conf = Config(path=base_path, target=target, run=run)(config_file)
    
    ret = Retrieval(
        conf=conf, 
        evaluation=False,
    )
    
    bestfit_params, posterior = ret.PMN_analyze()
    
    labels = list(ret.Param.param_keys)
    posterior_dict = dict(zip(labels, posterior.T))
    
    # Extract isotope ratios
    C_ratio_list.append(10.0**posterior_dict['log_12CO/13CO'])
    
    
    # Check if oxygen isotope ratio exists
    if 'log_12CO/C18O' in posterior_dict:
        O_ratio_list.append(10.0**posterior_dict['log_12CO/C18O'])
    else:
        O_ratio_list.append(None)
    
    ret.get_PT_mf_envelopes(posterior)
    PT_list.append(ret.PT)

# Create figure with subplots
fig = plt.figure(figsize=(9, 6))
plt.subplots_adjust(hspace=0.2)

# PT profile subplot
ax_PT = fig.add_subplot(2, 3, 1)
ax_grad = fig.add_subplot(2, 3, 2)

# Isotope ratio subplots
ax_C_ratio = fig.add_subplot(2, 3, 4)
if any(o is not None for o in O_ratio_list):
    ax_O_ratio = fig.add_subplot(2, 3, 5)

# Plot PT profiles and gradients
for PT, run in zip(PT_list, runs):
    print(f'Plotting PT profile for {run}...')
    
    figs.fig_PT(
        PT=PT, 
        ax=ax_PT, 
        ax_grad=ax_grad,
        fig=fig,
        bestfit_color=colors[run],
        envelopes_color=colors[run],
        int_contr_em_color=colors[run],
        show_photosphere=True,
        show_knots=False,
        xlim=(2000, 7000),
    )

# Configure PT and gradient axes
ax_PT.set_ylim(1e2, 1e-5)
ax_PT.set_xlabel('Temperature (K)', fontsize=fs)
ax_PT.set_ylabel('Pressure (bar)', fontsize=fs)
# ax_PT.set_title('PT Profile', fontsize=fs)

ax_grad.set_ylim(1e2, 1e-5)
ax_grad.set_ylabel('')
ax_grad.set_yticks([])
ax_grad.set_xlim(-0.05, 0.50)
ax_grad.set_xlabel('Temperature Gradient', fontsize=fs)
# ax_grad.set_title('Temperature Gradient', fontsize=fs)

# Plot carbon isotope ratios
hist_kwargs = {
    "alpha": 0.5, 
    "fill": True, 
    "edgecolor": "k",
    "linewidth": 1.0, 
    "histtype": "stepfilled", 
    "density": True,
}

for C_ratio, run in zip(C_ratio_list, runs):
    hist_kwargs_run = hist_kwargs.copy()
    hist_kwargs_run["color"] = colors[run]
    
    ax_C_ratio.hist(C_ratio, bins=30, **hist_kwargs_run)
    
    # Calculate and print percentiles
    C_ratio_q = np.percentile(C_ratio, [16, 50, 84])
    print(f'{run} C_ratio: {C_ratio_q[1]:.1f} +{C_ratio_q[2] - C_ratio_q[1]:.1f} -{C_ratio_q[1] - C_ratio_q[0]:.1f}')

# Add reference lines for carbon ratios
ax_C_ratio.axvline(89.0, color='magenta', ls='--', lw=2, alpha=0.7, label='Solar')
plot_ISM = False

if plot_ISM:
    ISM = (68, 15)
    ax_C_ratio.axvspan(ISM[0]-ISM[1], ISM[0]+ISM[1], color='gray', alpha=0.15, label='ISM')

ax_C_ratio.set_xlabel(r'$^{12}$C/$^{13}$C', fontsize=fs)
ax_C_ratio.set_ylabel('Density', fontsize=fs)
# ax_C_ratio.set_title('Carbon Isotope Ratio', fontsize=fs)

# Plot oxygen isotope ratios if available
if any(o is not None for o in O_ratio_list):
    for O_ratio, run in zip(O_ratio_list, runs):
        if O_ratio is not None:
            hist_kwargs_run = hist_kwargs.copy()
            hist_kwargs_run["color"] = colors[run]
            
            ax_O_ratio.hist(O_ratio, bins=30, **hist_kwargs_run)
            
            # Calculate and print percentiles
            O_ratio_q = np.percentile(O_ratio, [16, 50, 84])
            print(f'{run} O_ratio: {O_ratio_q[1]:.1f} +{O_ratio_q[2] - O_ratio_q[1]:.1f} -{O_ratio_q[1] - O_ratio_q[0]:.1f}')
    
    # Add reference lines for oxygen ratios
    ax_O_ratio.axvline(499.0, color='magenta', ls='--', lw=3, alpha=0.7, label='Solar')
    ax_O_ratio.set_xlabel(r'$^{16}$O/$^{18}$O', fontsize=fs)
    ax_O_ratio.set_ylabel('Density', fontsize=fs)
    # ax_O_ratio.set_title('Oxygen Isotope Ratio', fontsize=fs)

# Add legend
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[run], markersize=6) for run in runs]
# add custom legend to ax[0]
# fig.legend(handles, legend_labels, 
#         #    loc=(0.05, 1.01),
#            fontsize=fs, title='Runs', title_fontsize=fs)
ax_PT.legend(handles, legend_labels, loc='upper right', 
             fontsize=fs*0.6, title='Runs', title_fontsize=fs*0.6)
ax_C_ratio.legend()

# fig.subplots_adjust(top=0.92, right=0.85)
xlim = {'carbon': (60, 100), 'oxygen': (200, 800)}
ax_C_ratio.set_xlim(xlim['carbon'])
ax_O_ratio.set_xlim(xlim['oxygen'])
# Adjust layout
plt.tight_layout()
# Save figure
runs_label = '_'.join(runs)
fig_name = base_path + 'paper/figures/isotope_ratios_' + target + '_' + runs_label + '.png'
fig.savefig(fig_name, dpi=300, bbox_inches='tight')
print(f'Figure saved as {fig_name}')
plt.close(fig) 