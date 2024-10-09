from retrieval_base.retrieval import Retrieval
import retrieval_base.figures as figs
from retrieval_base.config import Config
from retrieval_base.auxiliary_functions import spirou_sample, read_spirou_sample_csv
# import config_freechem as conf
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib

base_path = '/home/dario/phd/retrieval_base/'

target = 'gl880'

def main(target, x, xerr=None, label='', ax=None, run=None, **kwargs):
    if target not in os.getcwd():
        os.chdir(base_path + target)

    outputs = pathlib.Path(base_path) / target / 'retrieval_outputs'
    # find dirs in outputs
    # print(f' outputs = {outputs}')
    dirs = [d for d in outputs.iterdir() if d.is_dir() and 'sphinx' in d.name and '_' not in d.name]
    runs = [int(d.name.split('sphinx')[-1]) for d in dirs]
    print(f' {target}: Found {len(runs)} runs: {runs}')
    assert len(runs) > 0, f'No runs found in {outputs}'
    if run is None:
        run = 'sphinx'+str(max(runs))
    else:
        run = 'sphinx'+str(run)
        assert run in [d.name for d in dirs], f'Run {run} not found in {dirs}'
    # print('Run:', run)
    
    carbon_isotope_posterior_file = base_path + target + '/retrieval_outputs/' + run + '/carbon_isotope_posterior.npy'
    if not os.path.exists(carbon_isotope_posterior_file):

        config_file = 'config_freechem.txt'
        conf = Config(path=base_path, target=target, run=run)(config_file)

        ret = Retrieval(
                    conf=conf, 
                    evaluation=False,
                    )

        bestfit_params, posterior = ret.PMN_analyze()

        param_keys = list(ret.Param.param_keys)
        log_carbon_ratio_id = param_keys.index('log_12CO/13CO')

        # make scatter plot with one point corresponding to Teff vs log(12CO/13CO)
        # take uncertainties from posterior quantiles 

        carbon_isotope_posterior = 10.0**posterior[:, log_carbon_ratio_id]
        np.save(carbon_isotope_posterior_file, carbon_isotope_posterior)
        print(f' {target}: Saved carbon isotope posterior to {carbon_isotope_posterior_file}')
    else:
        carbon_isotope_posterior = np.load(carbon_isotope_posterior_file)
        
    q=[0.16, 0.5, 0.84]
    carbon_isotope_quantiles = np.quantile(carbon_isotope_posterior, q)
        

    ax_new = ax is None
    ax = ax or plt.gca()
    print(f' {target}: log 12C/13C = {carbon_isotope_quantiles[1]:.2f} +{carbon_isotope_quantiles[2]-carbon_isotope_quantiles[1]:.2f} -{carbon_isotope_quantiles[1]-carbon_isotope_quantiles[0]:.2f}\n')
    # add black edge to points
    xerr = [x,x] if xerr is None else xerr
    if isinstance(xerr, (int, float)):
        xerr = [x-xerr, x+xerr]
    if isinstance(xerr, list):
        xerr = [[x-xerr[0]], [xerr[1]-x]]
        
    ax.errorbar(x, carbon_isotope_quantiles[1], 
                xerr=xerr,
                yerr=[[carbon_isotope_quantiles[1]-carbon_isotope_quantiles[0]], [carbon_isotope_quantiles[2]-carbon_isotope_quantiles[1]]],
                fmt='o', 
                label=label.replace('gl', 'Gl '),
                alpha=0.9,
                    # markerfacecolor='none',  # Make the inside of the marker transparent (optional)
                 markeredgecolor='black', # Black edge color
                markeredgewidth=0.8,     # Thickness of the edge
                color=kwargs.get('color', 'k'),
    )
    if x<xlim[1]:
        # add text with target name next to point, offset text from point
        ax.annotate(label.replace('gl', 'Gl '), (x, carbon_isotope_quantiles[1]), textcoords="offset points", xytext=(3,3), ha='left',
                    fontsize=8, color=kwargs.get('color', 'k'), alpha=0.9)
        
    

    if ax_new:
        ax.set_ylabel(r'$^{12}$C/$^{13}$C')
        
    return carbon_isotope_quantiles
        


df = read_spirou_sample_csv()
names = df['Star'].to_list()
teff =  dict(zip(names, [float(t.split('+-')[0]) for t in df['Teff (K)'].to_list()]))
prot = dict(zip(names, [float(t.split('+-')[0]) for t in df['Period (days)'].to_list()]))
prot_err = dict(zip(names, [float(t.split('+-')[1]) for t in df['Period (days)'].to_list()]))
runs = dict(zip(spirou_sample.keys(), [spirou_sample[k][1] for k in spirou_sample.keys()]))

# create colormap with teff in K
norm = plt.Normalize(min(teff.values()), 4000.0)
cmap = plt.cm.plasma

fig, ax = plt.subplots(1,1, figsize=(5,5), tight_layout=True)

zoom_in = False
xlim = [0.0, 500.0]
if zoom_in:
    xlim[1] = 150.0

for name in names:
    target = name.replace('Gl ', 'gl')
    color = cmap(norm(teff[name]))

    C_ratio_t = main(target, prot[name], xerr=prot_err[name],ax=ax, label=name, run=runs[target[2:]], color=color)
    
    

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Only needed for color bar
cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.03, aspect=20, location='right')
cbar.set_label(r'T$_{\mathrm{eff}}$ (K)')

solar = [89.0, 3.0] # review this value...
ax.axhspan(solar[0]-solar[1], solar[0]+solar[1], color='deepskyblue', alpha=0.3, label='Solar',lw=0)

ism = [69.0, 15.0]
ax.axhspan(ism[0]-ism[1], ism[0]+ism[1], color='green', alpha=0.2, label='ISM',lw=0)
ylim_min = 30.0
ylim_max = 200.0

ax.set_ylim(ylim_min, ylim_max)

ax.set_xlim(xlim)

ax.legend(ncol=4, frameon=False, fontsize=8, loc=(0.0, 1.01))
ax.set_xlabel(r'P$_{\mathrm{rot}}$ (days)')
ax.set_ylabel(r'$^{12}$C/$^{13}$C')
# plt.show()

zoom_in_label = '_zoom' if zoom_in else ''
fig_name = base_path + f'paper/latex/figures/carbon_isotope_period{zoom_in_label}.pdf'
fig.savefig(fig_name)
print(f'Figure saved as {fig_name}')
plt.close(fig)