from retrieval_base.retrieval import Retrieval
import retrieval_base.figures as figs
from retrieval_base.config import Config
from retrieval_base.auxiliary_functions import spirou_sample, read_spirou_sample_csv, load_romano_models
# import config_freechem as conf
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import matplotlib.patheffects as pe

base_path = '/home/dario/phd/retrieval_base/'

water = False # take isotope ratio from H2O
main_label = 'H2O' if water else 'CO'
isotope = 'oxygen' 
assert isotope in ['carbon', 'oxygen'], f'Isotope {isotope} not recognized (choose from oxygen, carbon)'
y_labels = {'oxygen': r'$^{16}$O/$^{18}$O', 'carbon': r'$^{12}$C/$^{13}$C'}
y_lims = {'oxygen': (30, 4000), 'carbon': (20, 400)}

def main(target, isotope, x, xerr=None, label='', ax=None, run=None, xytext=None,**kwargs):
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
    
    # load sigma for C18O
    sigma = 10.0 # default, > 3 for plotting as errorbar
    if isotope == 'oxygen' and main_label == 'CO':
        sigma_file = test_output / f'B_sigma_C18O.dat' # contains two values: B, sigma
        if sigma_file.exists():
            print(f' {target}: Found {sigma_file}')
            B, sigma = np.loadtxt(sigma_file)
            print(f' {target}: B = {B:.2f}, sigma = {sigma:.2f}')
            # replace nan with 0.0
            sigma = 0.0 if np.isnan(sigma) else sigma

    
    isotope_posterior_file = base_path + target + '/retrieval_outputs/' + run + f'/{main_label}_{isotope}_isotope_posterior.npy'
    if not os.path.exists(isotope_posterior_file):

        config_file = 'config_freechem.txt'
        conf = Config(path=base_path, target=target, run=run)(config_file)

        ret = Retrieval(
                    conf=conf, 
                    evaluation=False,
                    )

        bestfit_params, posterior = ret.PMN_analyze()

        param_keys = list(ret.Param.param_keys)
        
        if isotope == 'oxygen':
            key  = 'log_H2O/H2O_181' if water else 'log_12CO/C18O'
        elif isotope == 'carbon':
            key = 'log_12CO/13CO'
            
        log_ratio_id = param_keys.index(key)

        # make scatter plot with one point corresponding to Teff vs log(12CO/13CO)
        # take uncertainties from posterior quantiles 

        isotope_posterior = 10.0**posterior[:, log_ratio_id]
        np.save(isotope_posterior_file, isotope_posterior)
        print(f' {target}: Saved isotope posterior to {isotope_posterior_file}')
    else:
        isotope_posterior = np.load(isotope_posterior_file)
        
    q=[0.16, 0.5, 0.84]
    isotope_quantiles = np.quantile(isotope_posterior, q)
        

    ax_new = ax is None
    ax = ax or plt.gca()
    print(f' {target}: log {main_label} isotope = {isotope_quantiles[1]:.2f} +{isotope_quantiles[2]-isotope_quantiles[1]:.2f} -{isotope_quantiles[1]-isotope_quantiles[0]:.2f}\n')
    # add black edge to points
    xerr = [x,x] if xerr is None else xerr
    if isinstance(xerr, (int, float)):
        xerr = [x-xerr, x+xerr]
    if isinstance(xerr, list):
        xerr = [[x-xerr[0]], [xerr[1]-x]]
        
    fmt = 'o'
    if sigma > 3.0:
        ax.errorbar(x, isotope_quantiles[1],
                    xerr=xerr,
                    yerr=[[isotope_quantiles[1]-isotope_quantiles[0]], [isotope_quantiles[2]-isotope_quantiles[1]]],
                    fmt=fmt, 
                    label=label.replace('gl', 'Gl '),
                    alpha=0.9,
                        # markerfacecolor='none',  # Make the inside of the marker transparent (optional)
                    markeredgecolor='black', # Black edge color
                    markeredgewidth=0.8,     # Thickness of the edge
                    color=kwargs.get('color', 'k'),
        )
    else:
        # plot lower limit
        fmt = '^'
        ax.errorbar(x, isotope_quantiles[0],
                    xerr=xerr,
                    yerr=[[0.0], [isotope_quantiles[2]-isotope_quantiles[0]]],
                    lolims=True,
                    fmt=fmt, 
                    label=label.replace('gl', 'Gl '),
                    alpha=0.9,
                        # markerfacecolor='none',  # Make the inside of the marker transparent (optional)
                    markeredgecolor='red', # Black edge color
                    markeredgewidth=0.8,     # Thickness of the edge
                    color=kwargs.get('color', 'k'),
        )
        
        
    if xytext is not None:
        # add text with target name next to point, offset text from point
        ax.annotate(label.replace('gl', 'Gl '), (x, isotope_quantiles[1]), textcoords="offset points", xytext=xytext, ha='left',
                    fontsize=8, color=kwargs.get('color', 'k'), alpha=0.9)
        
    return isotope_quantiles
        


df = read_spirou_sample_csv()
names = df['Star'].to_list()
teff =  dict(zip(names, [float(t.split('+-')[0]) for t in df['Teff (K)'].to_list()]))
valid = dict(zip(names, df['Valid'].to_list()))

ignore_targets = [name.replace('Gl ', 'gl') for name in names if valid[name] == 0]
ignore_more_targets = ['gl3622']
ignore_targets += ignore_more_targets

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
crossfield_dict = {'oxygen': {'Gl 745 A': [(1220, 260), (3454, 31), (-0.43, 0.05)],
                        'Gl 745 B': [(1550, 360), (3440, 31), (-0.39, 0.05)]},
              'carbon': {'Gl 745 A': [(296, 45), (3454, 31), (-0.43, 0.05)],
                        'Gl 745 B': [(224, 26), (3440, 31), (-0.39, 0.05)]}
}
sun_dict = {'oxygen': (529.7, 1.7),# solar wind McKeegan et al. 2011
            'carbon': (93.5, 3.0)}
ism_dict = {'oxygen': (557, 30), # ISM value from Wilson et al. 1999
            'carbon': (68.0, 14.0)}

plot_crossfield = True

top = 0.77
fig, axes = plt.subplots(2,1, figsize=(5,8), sharex=True, gridspec_kw={'hspace': 0.1, 
                                                                       'wspace': 0.1,
                                                                        'left': 0.14, 
                                                                        'right': 0.78, 
                                                                        'top': top, 
                                                                        'bottom': 0.06})
# ylim_min = 50.0
# ylim_max = 3000.0

xytext = {'Gl 699' : (-28,5),
        #   'Gl 411' : (3,3),
        #   'Gl 382': (-20,-12),
        #   'Gl 1286': (2,-12),
          
}
isotopes = ['carbon', 'oxygen']
for i, isotope in enumerate(isotopes):
    print(f' ** Isotope {isotope} **')
    ax = axes[i]
    ax.set_ylim(*y_lims[isotope])

    crossfield = crossfield_dict[isotope]
    ism = ism_dict[isotope]
    sun = sun_dict[isotope]

    # ax.axhspan(sun[0]-sun[1], sun[0]+sun[1], color='gold', alpha=0.3, label='Solar',lw=0)
    ax.plot(0.0, sun[0], color='gold', marker='*', ms=16, label='Sun', alpha=0.8, markeredgecolor='black', markeredgewidth=0.8, zorder=100)
    

    plot_teff_max = 4400.0
    for name in names:
        print(f'---> {name}')
        if teff[name] > plot_teff_max:
            continue
        target = name.replace('Gl ', 'gl')
        if target in ignore_targets:
            print(f'---> Skipping {target}...')
            continue
        color = cmap(norm(teff[name]))

        ratio_t = main(target, 
                       isotope,
                       x[name], 
                       xerr=x_err[name],
                        ax=ax, 
                        label=name, 
                        run=None,
                        color=color,
                        xytext=xytext.get(name, None))
        


    ax.axhspan(ism[0]-ism[1], ism[0]+ism[1], color='green', alpha=0.2,lw=0, zorder=-1, label='ISM')
    # ax.text(0.95, 0.15, 'ISM', color='darkgreen', fontsize=12, transform=ax.transAxes, ha='right', va='top')
   

    # plot crossfield values
    if plot_crossfield:
        for k, v in crossfield.items():
            teff_cf = v[1][0]
            color = cmap(norm(teff_cf))
            x_cf = v[1][0] if x_param == 'Teff (K)' else v[2][0]
            x_err_cf = v[1][1] if x_param == 'Teff (K)' else v[2][1]
            ax.errorbar(x_cf, v[0][0], xerr=x_err_cf, yerr=v[0][1], fmt='s', label=k+' (C19)', color=color, markeredgecolor='black', markeredgewidth=0.8)

            # add thin arrow pointing to the marker with the name of the target
            annotate = False
            if annotate:
                ax.annotate(k, (x_cf, v[0][0]), textcoords="offset points", 
                            xytext=(60,5), ha='center', va='center',
                            arrowprops=dict(facecolor='black', shrink=1, headwidth=1, width=0.5, headlength=0.1),
                            fontsize=8,
                            horizontalalignment='right', verticalalignment='top')
                    
        
    if x_param == '[M/H]':
        ax.axvline(0.0, color='k', lw=0.5, ls='--', zorder=-10)

    # if i == 0:
        
    if i == 1:
        ax.set_xlabel(x_param)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Only needed for color bar
        
        # define cbar_ax for colorbar
        cbar_ax = fig.add_axes([0.79, 0.06, 0.035, top-0.06])
        cbar = plt.colorbar(sm, cax=cbar_ax, orientation='vertical', aspect=20)
        cbar.set_label(r'T$_{\mathrm{eff}}$ (K)')

        
        
    ax.set_ylabel(y_labels[isotope])
    
# load Romano+2022 models
mass_ranges = ['1_8', '3_8']

gce_colors = ['black', 'royalblue']
# add white edge to line
path_effects = [pe.Stroke(linewidth=2.5, foreground='white'), pe.Normal()]

for i, mass_range in enumerate(mass_ranges):
    Z, c12c13, o16o18 = load_romano_models(Z_min=-0.7, mass_range=mass_range)
    mass_range_label = 'Romano22 (' + mass_range.replace('_', '-') + r' M$_\odot$)'
    axes[0].plot(Z, c12c13, color=gce_colors[i], lw=1.5, label=mass_range_label, alpha=0.8, path_effects=path_effects)
    axes[1].plot(Z, o16o18, color=gce_colors[i], lw=1.5, label=mass_range_label, alpha=0.8, path_effects=path_effects)
    

axes[0].legend(ncol=3, frameon=False, fontsize=8, loc=(-0.05, 1.01))
loglog = True
loglog_label = '_loglog' if loglog else ''
if loglog:
    
    ylims = {'oxygen': (100, 6000), 'carbon': (30, 600)}
    yticks = {'oxygen': [100, 200, 500, 1000, 2000, 6000], 'carbon': [30, 60, 100, 200, 600]}
    
    for ax, isotope in zip(axes, isotopes):
        ax.set_yscale('log')
        
        ylim = ylims[isotope]
        ax.set_ylim(*ylim)
        
        ax.set_yticks(yticks[isotope])
        ax.set_yticklabels([str(t) for t in yticks[isotope]])
        
        
        
        # yticks = [120, 240, 480, 960]
        # ax.set_yticks(yticks)
        # ax.set_yticklabels([str(t) for t in yticks])
        
        # xticks = [10, 20, 50, 100, 200, 400,800]
        # ax.set_xticks(xticks)
        # ax.set_xticklabels([str(t) for t in xticks])
else:
    # ax.set_ylim(ylim_min, ylim_max)
    pass
    # ax.set_xlim(xlim)
# x_param_label = x_param.split('(')[0].strip()
x_param_label = {
    'Teff (K)': 'Teff',
    '[M/H]': 'metallicity',
}[x_param]
fig_name = base_path + f'paper/latex/figures/{main_label}_isotopes_{x_param_label}{loglog_label}.pdf'
fig.savefig(fig_name)
print(f'Figure saved as {fig_name}')
plt.close(fig)