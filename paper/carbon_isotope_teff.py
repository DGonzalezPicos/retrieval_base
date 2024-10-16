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

def main(target, x, xerr=None, label='', ax=None, run=None, xytext=None,**kwargs):
    if target not in os.getcwd():
        os.chdir(base_path + target)

    outputs = pathlib.Path(base_path) / target / 'retrieval_outputs'
    # find dirs in outputs
    # print(f' outputs = {outputs}')
    dirs = [d for d in outputs.iterdir() if d.is_dir() and 'fc' in d.name and '_' not in d.name]
    print(f' dirs = {dirs}')
    runs = [int(d.name.split('fc')[-1]) for d in dirs]
    print(f' runs = {runs}')
    print(f' {target}: Found {len(runs)} runs: {runs}')
    assert len(runs) > 0, f'No runs found in {outputs}'
    if run is None:
        run = 'fc'+str(max(runs))
    else:
        run = 'fc'+str(run)
        assert run in [d.name for d in dirs], f'Run {run} not found in {dirs}'
    # print('Run:', run)
    # check that the folder 'test_output' is not empty
    test_output = outputs / run / 'test_output'
    assert test_output.exists(), f'No test_output folder found in {test_output}'
    if len(list(test_output.iterdir())) == 0:
        print(f' {target}: No files found in {test_output}')
        return None
    
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
        
        
    y = carbon_isotope_quantiles[1]
    lolims=False
    fmt = 'o'
    if target=='gl699' and plot_lower_limit_gl699:
        y -= carbon_isotope_quantiles[0]
        lolims = carbon_isotope_quantiles[2]-carbon_isotope_quantiles[1]
        fmt = 's'
    ax.errorbar(x, y, 
                xerr=xerr,
                yerr=[[carbon_isotope_quantiles[1]-carbon_isotope_quantiles[0]], [carbon_isotope_quantiles[2]-carbon_isotope_quantiles[1]]],
                fmt=fmt, 
                label=label.replace('gl', 'Gl '),
                alpha=0.9,
                    # markerfacecolor='none',  # Make the inside of the marker transparent (optional)
                markeredgecolor='black', # Black edge color
                markeredgewidth=0.8,     # Thickness of the edge
                color=kwargs.get('color', 'k'),
                lolims=lolims,
    )
        
        # # Optionally, add an arrow to highlight the lower limit
        # ax.annotate('', xy=(x, carbon_isotope_quantiles[1]-carbon_isotope_quantiles[0]), 
        #             xytext=(x, (carbon_isotope_quantiles[1]-carbon_isotope_quantiles[0]) - 0.1),
        #             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8)
        # )
    # if x<xlim[1]:
    if xytext is not None:
        # add text with target name next to point, offset text from point
        ax.annotate(label.replace('gl', 'Gl '), (x, carbon_isotope_quantiles[1]), textcoords="offset points", xytext=xytext, ha='left',
                    fontsize=8, color=kwargs.get('color', 'k'), alpha=0.9)
        
    

    if ax_new:
        ax.set_ylabel(r'$^{12}$C/$^{13}$C')
        
    return carbon_isotope_quantiles
        


df = read_spirou_sample_csv()
names = df['Star'].to_list()
teff =  dict(zip(names, [float(t.split('+-')[0]) for t in df['Teff (K)'].to_list()]))

# x_param = 'Teff (K)'
x_param = '[M/H]'
assert x_param in df.columns, f'Column {x_param} not found in {df.columns}'
x =  dict(zip(names, [float(t.split('+-')[0]) for t in df[x_param].to_list()]))
x_err = dict(zip(names, [float(t.split('+-')[1]) for t in df[x_param].to_list()]))

runs = dict(zip(spirou_sample.keys(), [spirou_sample[k][1] for k in spirou_sample.keys()]))

# create colormap with teff in K
norm = plt.Normalize(min(teff.values()), 4000.0)
cmap = plt.cm.plasma

# add Crossfield+2019 values for Gl 745 AB: isotope ratio, teff and metallicity, with errors
crossfield = {'Gl 745 A': [(296, 45), (3454, 31), (-0.43, 0.05)],
              'Gl 745 B': [(224, 26), (3440, 31), (-0.39, 0.05)],
}


plot_lower_limit_gl699 = False # TODO: check this...

fig, ax = plt.subplots(1,1, figsize=(5,5), tight_layout=True)

# zoom_in = False
# xlim = [2800.0, 4100.0]
# if zoom_in:
#     xlim[1] = 200.0
    
xytext = {'Gl 699' : (-28,5),
          'Gl 411' : (3,3),
          'Gl 382': (-20,-12),
          'Gl 1286': (2,-12),
          
}

sun = (93.5, 3.0)
# sun = {'Teff (K)': ()
ax.axhspan(sun[0]-sun[1], sun[0]+sun[1], color='deepskyblue', alpha=0.3, label='Solar',lw=0)


for name in names:
    target = name.replace('Gl ', 'gl')
    color = cmap(norm(teff[name]))

    C_ratio_t = main(target, x[name], xerr=x_err[name],
                     ax=ax, label=name, 
                     run=None,
                     color=color,
                     xytext=xytext.get(name, None))
    
    

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Only needed for color bar
cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.03, aspect=20, location='right')
cbar.set_label(r'T$_{\mathrm{eff}}$ (K)')

# solar = [89.0, 3.0] # review this value...
# ax.axhspan(solar[0]-solar[1], solar[0]+solar[1], color='deepskyblue', alpha=0.3, label='Solar',lw=0)

ism = [69.0, 15.0]
ax.axhspan(ism[0]-ism[1], ism[0]+ism[1], color='green', alpha=0.2,lw=0, zorder=-1)
ax.text(0.95, 0.15, 'ISM', color='darkgreen', fontsize=12, transform=ax.transAxes, ha='right', va='top')
ylim_min = 30.0
ylim_max = 350.0

# plot crossfield values
for k, v in crossfield.items():
    teff = v[1][0]
    color = cmap(norm(teff))
    x = v[1][0] if x_param == 'Teff (K)' else v[2][0]
    x_err = v[1][1] if x_param == 'Teff (K)' else v[2][1]
    ax.errorbar(x, v[0][0], xerr=x_err, yerr=v[0][1], fmt='s', label=k+' (C19)', color=color, markeredgecolor='black', markeredgewidth=0.8)

    # add thin arrow pointing to the marker with the name of the target
    ax.annotate(k, (x, v[0][0]), textcoords="offset points", 
                xytext=(60,5), ha='center', va='center',
                arrowprops=dict(facecolor='black', shrink=1, headwidth=1, width=0.5, headlength=0.1),
                fontsize=8,
                horizontalalignment='right', verticalalignment='top')
                
    
if x_param == '[M/H]':
    ax.axvline(0.0, color='k', lw=0.5, ls='--', zorder=-1)

ax.legend(ncol=4, frameon=False, fontsize=8, loc=(-0.03, 1.01))
# ax.set_xlabel(r'P$_{\mathrm{rot}}$ (days)')
ax.set_xlabel(x_param)
ax.set_ylabel(r'$^{12}$C/$^{13}$C')
# plt.show()

# zoom_in_label = '_zoom' if zoom_in else ''

loglog = False
loglog_label = '_loglog' if loglog else ''
if loglog:
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    yticks = [30, 60, 90, 120, 200, 300, 500]
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(t) for t in yticks])
    
    xticks = [10, 20, 50, 100, 200, 400,800]
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(t) for t in xticks])
else:
    ax.set_ylim(ylim_min, ylim_max)
    # ax.set_xlim(xlim)
# x_param_label = x_param.split('(')[0].strip()
x_param_label = {
    'Teff (K)': 'Teff',
    '[M/H]': 'metallicity',
}[x_param]
fig_name = base_path + f'paper/latex/figures/carbon_isotope_{x_param_label}{loglog_label}.pdf'
fig.savefig(fig_name)
print(f'Figure saved as {fig_name}')
plt.close(fig)