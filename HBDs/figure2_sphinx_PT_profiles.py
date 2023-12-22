
# import config_freechem as conf
import matplotlib.pyplot as plt
import pathlib
plt.style.use('/home/dario/phd/retrieval_base/HBDs/my_science.mplstyle')
# plt.rcParams['text.usetex'] = False
# make minor ticks in y-axis invisible
# plt.rcParams['xtick.minor.visible'] = False
plt.rcParams['xtick.minor.size'] = 4.
plt.rcParams['xtick.minor.width'] = 1.5
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['ytick.minor.size'] = 4.
# plt.rcParams['xtick.major.size'] = 3.
# plt.rcParams['xtick.major.width'] = 1.5

import numpy as np
import os

from retrieval_base.retrieval import pre_processing, Retrieval
from retrieval_base.parameters import Parameters
from retrieval_base.PT_profile import PT_profile_free_gradient
from retrieval_base.auxiliary_functions import quantiles

outdir = pathlib.Path(__file__).parent.absolute()
print(f'Outdir is {outdir}')
def load_sphinx_thermal_profile(Teff=2600.0, log_g=4.0, logZ=0.0, C_O=0.50):
        
    path = pathlib.Path('/home/dario/phd/SPHINX_MODELS_MLT_1/ATMS/')
    sign = '+' if logZ >= 0 else '-'
    file = path / f'Teff_{Teff:.1f}_logg_{log_g}_logZ_{sign}{abs(logZ)}_CtoO_{C_O}_atms.txt'
    assert file.exists(), f'File {file} does not exist.'
    t, p = np.loadtxt(file, unpack=True)
    return t, p

# read data
path =  pathlib.Path('/home/dario/phd/SPHINX_MODELS_MLT_1/ATMS/')
files = sorted(path.glob('*.txt'))
print(f'Number of files: {len(list(files))}')
# Teff_2000.0_logg_4.0_logZ_+0.0_CtoO_0.3_atms.txt

target = 'J0856'
# change working directory
os.chdir(f'/home/dario/phd/retrieval_base/{target}')
print(f'Current working directory is {os.getcwd()}')
import config_freechem as conf

n_layers = conf.config_data['K2166']['n_atm_layers']
logP_knots = np.array(conf.constant_params['log_P_knots'])[::-1]
pressure = np.logspace(np.min(logP_knots), np.max(logP_knots), n_layers)
# logP_knots = np.linspace()
P_knots = 10.0**(logP_knots)
N_knots = len(logP_knots)
print(f' logP_knots: {logP_knots}')

# for gaussian prior plot
yticks = logP_knots.max()-logP_knots

Teff = 2600.0
logg = 4.0
logZ = 0.0
sign = '+' if logZ >= 0 else '-'

C_O = 0.5
fig, ax = plt.subplots(1,2, figsize=(8, 6), sharey=True, 
                       gridspec_kw={'width_ratios': [3, 2], 'wspace': 0.02})
# remove minor ticks in y-axis (left and right)
ax[0].tick_params(axis='y', which='minor', left=False, right=False)
ax[0].tick_params(axis='y', which='major', right=False)
ax[1].tick_params(axis='y', which='minor', left=False, right=False)
ax[1].tick_params(axis='y', which='major', left=True, right=False)


Teff_grid = np.arange(2000.0, 3000.1, 100.0)
logg_grid = [4.0, 4.50]

cmap = getattr(plt.cm, 'inferno')
colors = cmap(np.linspace(0, 1, Teff_grid.size))

# generate colorbar

im = ax[1].scatter([], [], c=[], cmap=cmap, vmin=Teff_grid.min(), vmax=Teff_grid.max())
# make colorbar wider, do not include minor ticks
cbar = fig.colorbar(im, ax=ax[1], label='Effective temperature (K)', pad=0.03, aspect=15)
cbar.ax.set(ylabel='Effective temperature (K)')
cbar.ax.tick_params(which='minor', length=0)

for i, Teff in enumerate(Teff_grid):
    for logg in logg_grid:
        ls = '-' if logg == 4.0 else ':'
        label = f'$\\log g={logg}$' if i == 0 else None
        # file = path / f'Teff_{Teff:.1f}_logg_{logg}_logZ_{sign}{abs(logZ)}_CtoO_{C_O}_atms.txt'

        # t, p = np.loadtxt(file, unpack=True)
        t,p = load_sphinx_thermal_profile(Teff, logg, logZ)
        # ax.plot(t, p, label=f'Teff={Teff_i}, logg={logg}, logZ={logZ}, C/O={C_O}')
        ax[0].plot(t,p, ls=ls, color=colors[i], label=label, lw=3.)
        dlnT_dlnP = np.gradient(np.log10(t), np.log10(p))
        
       
        ax[1].plot(dlnT_dlnP, p, ls=ls, color=colors[i], lw=3., alpha=0.9)


PT = PT_profile_free_gradient(pressure=pressure)
ret = Retrieval(conf=conf, evaluation=False)

# uniform random sample

temp_list = []
temp_knots = []

n_samples = int(1e4)
for i in range(n_samples):
    ret.Param(np.random.uniform(size=len(ret.Param.param_keys)))

    sample = {k:ret.Param.params[k] for k in ret.Param.param_keys}
    t_knots = [sample[f'dlnT_dlnP_{i}'] for i in range(N_knots)]
    sample_PT = {'T_knots': np.ones(N_knots),
                'P_knots': P_knots,
                'log_P_knots': logP_knots,
                'T_0': sample['T_0'],
                'dlnT_dlnP_knots': t_knots}
    
    temp_knots.append(t_knots)
    temp_list.append(ret.PT(sample_PT))


# calculate envelopes
sigma2quantile= {1: 0.6826894921370859,
                    2: 0.9544997361036416,
                    3: 0.9973002039367398,
                    5: 0.9999994266968562}

temp_quantiles = []
# temp_quantiles.append(np.quantile(temp_list, sigma2quantile[1], axis=0))

from scipy.interpolate import CubicSpline
for s in [1,3]:
    quantile = sigma2quantile[s]
    lower_bounds = np.quantile(temp_list, 1.0 - quantile, axis=0)
    upper_bounds = np.quantile(temp_list, quantile, axis=0)
   
    # temp_quantiles.append(left_envelope)
    # ax[0].fill_betweenx(pressure, left_envelope, right_envelope, alpha=0.2, color='blue')
    
    ax[0].fill_betweenx(pressure, lower_bounds, upper_bounds, alpha=0.2, color='blue')


    knots_lower = np.quantile(temp_knots, quantile, axis=0)
    knots_upper = np.quantile(temp_knots, 1.0 - quantile, axis=0)
    left_envelope = CubicSpline(logP_knots[::-1], knots_lower[::-1])(np.log10(pressure))
    right_envelope = CubicSpline(logP_knots[::-1], knots_upper[::-1])(np.log10(pressure))
    ax[1].fill_betweenx(pressure, left_envelope, right_envelope, alpha=0.2, color='blue')
    
plot_horizontal_lines = False
if plot_horizontal_lines:
    # plot horizontal lines 
    for i in range(N_knots):
        [axi.axhline(10**logP_knots[i], color='k', ls='-', alpha=0.5) for axi in ax]
    

# set yticks to logP_knots

# add latex xlabel with nabla
# ax[1].set(xlabel='dlnT/dlnP', yscale='log')
ax[0].set(xlim=(0.0, 9000.), ylim=(1e2, 1e-5),
          xlabel='Temperature (K)', ylabel='Pressure (bar)',
          yscale='log')
# ax[0].set_yticks(10**logP_knots)
# ax[0].set_yticklabels([f'$10^{{{int(x)}}}$' for x in logP_knots])


ax[1].set(xlabel=r'$\nabla T$', yscale='log')
# add custom legend with the 1-sigma and 3-sigma envelopes
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

legend_elements = [Patch(facecolor='blue', edgecolor='k', alpha=0.2, label='3-$\sigma$ envelope'),
                     Patch(facecolor='blue', edgecolor='k', alpha=0.5, label='1-$\sigma$ envelope'),
                     Line2D([0], [0], color='k', lw=3., ls='-', label='log g=4.0'),
                     Line2D([0], [0], color='k', lw=3., ls=':', label='log g=4.5')][::-1]
# add to legend
leg = ax[0].legend(handles=legend_elements, loc='upper right', fontsize=15,
                   frameon=False, labelspacing=0.8, borderpad=0.5, handletextpad=0.5, ncol=1)

# ax[0].legend()
# ax[0].set_title('Self-consistent SPHINX-M models', fontsize=16, x=1.08, y=1.05)  
plt.show()
fig.savefig(outdir / 'fig2_SPHINX_M_PT.pdf', bbox_inches='tight')