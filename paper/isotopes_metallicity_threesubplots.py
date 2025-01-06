from retrieval_base.retrieval import Retrieval
import retrieval_base.figures as figs
from retrieval_base.config import Config
from retrieval_base.auxiliary_functions import spirou_sample, read_spirou_sample_csv, find_run, load_romano_models
# import config_freechem as conf
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import matplotlib.patheffects as pe
import scienceplots

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
    ignore_fc5 = False
    if ignore_fc5:
        runs = [r for r in runs if r != 5]
    
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
    
    if main_label == 'CO':
        species_sigma = 'C18O' if isotope == 'oxygen' else '13CO'
        # sigma_file = test_output / f'B_sigma_{species_sigma}.dat' # contains two values: B, sigma
        sigma_file = test_output / f'lnB_sigma_{species_sigma}.dat'
        if sigma_file.exists():
            print(f' {target}: Found {sigma_file}')
            lnB, sigma = np.loadtxt(sigma_file)
            print(f' {target}: lnB = {lnB:.2f}, sigma = {sigma:.2f}')
            # replace nan with 0.0
            sigma = 0.0 if np.isnan(sigma) else sigma
            sigma = 100.0 if sigma > 100.0 else sigma

    
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
    # add errorbar style with capsize
    if sigma > 3.0:
        ax.errorbar(x, isotope_quantiles[1],
                    xerr=xerr,
                    yerr=[[isotope_quantiles[1]-isotope_quantiles[0]], [isotope_quantiles[2]-isotope_quantiles[1]]],
                    fmt=fmt, 
                    label=label.replace('gl', 'Gl '),
                    alpha=0.96,
                        # markerfacecolor='none',  # Make the inside of the marker transparent (optional)
                    markeredgecolor='black', # Black edge color
                    markeredgewidth=0.8,     # Thickness of the edge
                    capsize=2,               # Size of the cap on error bars
                    capthick=0.8,             # Thickness of the cap on error bars
                    ecolor='gray',          # Color of the error bars, set alpha of ecolor to make it transparent
                    elinewidth=0.8,           # Thickness of the error bars
                    
                    
                    color=kwargs.get('color', 'k'),
        )
    elif sigma > 2.0:
        ax.errorbar(x, isotope_quantiles[1],
                    xerr=xerr,
                    yerr=[[isotope_quantiles[1]-isotope_quantiles[0]], [isotope_quantiles[2]-isotope_quantiles[1]]],
                    fmt=fmt, 
                    label=label.replace('gl', 'Gl '),
                    alpha=0.96,
                        # markerfacecolor='none',  # Make the inside of the marker transparent (optional)
                    markeredgecolor='darkorange', # Black edge color
                    markeredgewidth=0.8,     # Thickness of the edge
                    capsize=2,               # Size of the cap on error bars
                    capthick=0.8,             # Thickness of the cap on error bars
                    ecolor='gray',          # Color of the error bars, set alpha of ecolor to make it transparent
                    elinewidth=0.8,           # Thickness of the error bars
                    
                    
                    color=kwargs.get('color', 'k'),
        )
    else:
        # plot lower limit
        fmt = '^'
        ax.errorbar(x, isotope_quantiles[0],
                    xerr=xerr,
                    yerr=[[0.0], [0.4*(isotope_quantiles[2]-isotope_quantiles[1])]],
                    lolims=True,
                    fmt=fmt, 
                    label=label.replace('gl', 'Gl '),
                    alpha=0.9,
                        # markerfacecolor='none',  # Make the inside of the marker transparent (optional)
                    markeredgecolor='k', # Black edge color
                    markeredgewidth=0.8,     # Thickness of the edge
                    color=kwargs.get('color', 'k'),
        )
        
    return np.append(isotope_quantiles, sigma)
        


df = read_spirou_sample_csv()
names = df['Star'].to_list()
teff =  dict(zip(names, [float(t.split('+-')[0]) for t in df['Teff (K)'].to_list()]))
dist = dict(zip(names, [float(t) for t in df['Distance (pc)'].to_list()]))
# valid = dict(zip(names, df['Valid'].to_list()))
ignore_targets = []

# x_param = '[C/H]'
x_param = '[M/H]'

sub10pc = False
sub10pc_label = '_sub10pc' if sub10pc else ''
# ignore targets with distance greater than 10 pc
if sub10pc:
    ignore_targets += [name for name in names if dist[name] > 10.0]

table_id_label = ''

metallicity_ref = 'C23'
assert metallicity_ref in ['M15', 'C23'], f'metallicity_ref must be M15 or C23, not {metallicity_ref}'

if x_param == '[M/H]':

    if metallicity_ref == 'M15':
        m15 = np.loadtxt(f'{base_path}paper/data/mann15_feh.txt', dtype=object)
        m15_names = ['Gl '+n[2:] for n in m15[:,0]]
        x = dict(zip(m15_names, m15[:,1].astype(float)))
        x_err   = dict(zip(m15_names, m15[:,2].astype(float)))
        
    if metallicity_ref == 'C23':
        # Load name, metallicity and error from Cristofari+2023
        table_id = 3 # 3 or 4 possible
        table_id_label = f'_table{table_id}'
        c23 = np.loadtxt(f'{base_path}paper/data/c23_table{table_id}_mh.txt', dtype=object)
        if table_id == 4:
            # manually add entry for GL 4063
            c23 = np.append(c23, [['Gl4063', '0.36', '0.1']], axis=0) # from table 3 C23

        c23_names = ['Gl '+n[2:] for n in c23[:,0]]
        x = dict(zip(c23_names, c23[:,1].astype(float)))
        x_err = dict(zip(c23_names, c23[:,2].astype(float)))

runs = dict(zip(spirou_sample.keys(), [spirou_sample[k][1] for k in spirou_sample.keys()]))

# create colormap with teff in K
norm = plt.Normalize(3000.0, 3900.0)
cmap = plt.cm.coolwarm_r
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

plot_crossfield = False

top = 0.92
fig, axes = plt.subplots(1,3, figsize=(10,3), sharex=False, gridspec_kw={'hspace': 0.1, 
                                                                       'wspace': 0.1,
                                                                        'left': 0.15, 
                                                                        'right': 0.78, 
                                                                        'top': top, 
                                                                        'bottom': 0.07})

isotopes = ['carbon', 'oxygen']
# create empty dictionary with keys matching targets
targets_isotopes = {name.replace('Gl ', 'gl') : {isotope: [] for isotope in isotopes} for name in names}
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
        if name in ignore_targets:
            print(f'---> Skipping {name}...')
            continue
        color = cmap(norm(teff[name]))
        
        if x_param == '[C/H]':
            run = find_run(target=target)
            posterior_file_CH = base_path + target + '/retrieval_outputs/' + run + f'/CH_posterior.npy'
            C_H_posterior = np.load(posterior_file_CH)
        
            q=[0.16, 0.5, 0.84]
            C_H_quantiles = np.quantile(C_H_posterior, q)
            x = {name: C_H_quantiles[1]}
            x_err = {name: [C_H_quantiles[0], C_H_quantiles[2]]}
            print(f'---> {name}: {C_H_quantiles}')
            print(f' xerr = {x_err[name]}')
            
        try:
            ratio_t = main(target, 
                        isotope,
                        x[name], 
                        xerr=x_err[name],
                            ax=ax, 
                            # label=name, 
                            label='',
                            run=None,
                            color=color,
                            xytext=None,)
            
            targets_isotopes[target][isotope] = ratio_t

        except Exception as e:
            print(e)
            print(f'---> Skipping {name}')
            continue
        


    ax.axhspan(ism[0]-ism[1], ism[0]+ism[1], color='green', alpha=0.2,lw=0, zorder=-1, label='ISM')
    # ax.text(0.95, 0.15, 'ISM', color='darkgreen', fontsize=12, transform=ax.transAxes, ha='right', va='top')
   

    # plot crossfield values
    if plot_crossfield:
        for cross_i, (k, v) in enumerate(crossfield.items()):
            teff_cf = v[1][0]
            color = cmap(norm(teff_cf))
            x_cf = v[1][0] if x_param == 'Teff (K)' else v[2][0]
            x_err_cf = v[1][1] if x_param == 'Teff (K)' else v[2][1]
            fmt = 's' if cross_i == 0 else 'D'
            ax.errorbar(x_cf, v[0][0], xerr=x_err_cf, yerr=v[0][1], fmt=fmt, label=k+' (C19)', color=color, markeredgecolor='black', markeredgewidth=0.8)

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
        cbar_ax = fig.add_axes([0.80, 0.06, 0.027, top-0.06])
        cbar = plt.colorbar(sm, cax=cbar_ax, orientation='vertical', aspect=1)
        cbar.set_label(r'T$_{\mathrm{eff}}$ (K)')

        
        
    ax.set_ylabel(y_labels[isotope])
    
# load Romano+2022 models
mass_ranges = ['1_8', '3_8']

gce_colors = ['black', 'purple']
# add white edge to line
path_effects = [pe.Stroke(linewidth=2.5, foreground='white'), pe.Normal()]

for i, mass_range in enumerate(mass_ranges):
    Z, c12c13, o16o18 = load_romano_models(Z_min=-0.7, mass_range=mass_range)
    mass_range_label = 'R22 (' + mass_range.replace('_', '-') + r' M$_\odot$)'
    axes[0].plot(Z, c12c13, color=gce_colors[i], lw=1.5, label=mass_range_label, alpha=0.8, path_effects=path_effects)
    
    axes[-1].plot(c12c13, o16o18, color=gce_colors[i], lw=1.5, label=mass_range_label, alpha=0.8, path_effects=path_effects)
    if i == 1:
        axes[1].plot(Z, o16o18, color=gce_colors[i], lw=1.5, label=mass_range_label, alpha=0.8, path_effects=path_effects)

# plot the 13C/18O ratio in the last subplot
carbon_oxygen_ratio = {}
for target in targets_isotopes.keys():
    name = target.replace('gl', 'Gl ')
    c = targets_isotopes[target]['carbon']
    o = targets_isotopes[target]['oxygen']
    
    # value and error via error propagation
    ratio = c[0]/o[0]
    ratio_err = ratio*np.sqrt((o[1]/o[0])**2 + (c[1]/c[0])**2)
    carbon_oxygen_ratio[target] = [ratio, ratio_err]
    
    c_err = [[c[1] - c[0]], [c[2] - c[1]]]
    o_err = [[o[1] - o[0]], [o[2] - o[1]]]
    sigma_o = o[-1]
    sigma_c = c[-1]
    # scatter plot
    if (sigma_o > 3.0) and (sigma_c > 3.0):
        print(f' {target}: sigma_c = {sigma_c}, sigma_o = {sigma_o}')
        print(f' {target}: c = {c}, o = {o}')
        print(f' {target}: c_err = {c_err}, o_err = {o_err}')
        axes[-1].errorbar(c[1], o[1], xerr=c_err, yerr=o_err, fmt='o', label=name, color=cmap(norm(teff[name])), markeredgecolor='black', markeredgewidth=0.8)
    if (sigma_c > 3.0) and (sigma_o < 3.0):
        # plot lower limit
        fmt = '^'
        axes[-1].errorbar(c[1],
                    o[0],
                    xerr=c_err,
                    yerr=[[0.0], [0.4*(o[2]-o[1])]],
                    lolims=True,
                    fmt=fmt, 
                    # label=label.replace('gl', 'Gl '),
                    alpha=0.9,
                        # markerfacecolor='none',  # Make the inside of the marker transparent (optional)
                    markeredgecolor='k', # Black edge color
                    markeredgewidth=0.8,     # Thickness of the edge
                    color=cmap(norm(teff[name]))
        )
    if (sigma_c < 3.0) and (sigma_o < 3.0):
        fmt = '^'
        axes[-1].errorbar(c[0],
                    o[0],
                    xerr=[[0.0], [0.4*(c[2]-c[1])]],
                    yerr=[[0.0], [0.4*(o[2]-o[1])]],
                    lolims=True,
                    xlolims=True,
                    fmt=fmt, 
                    # label=label.replace('gl', 'Gl '),
                    alpha=0.9,
                        # markerfacecolor='none',  # Make the inside of the marker transparent (optional)
                    markeredgecolor='k', # Black edge color
                    markeredgewidth=0.8,     # Thickness of the edge
                    color=cmap(norm(teff[name]))
        )
# axes[-1].set(xscale='log', yscale='log')
axes[-1].set_xlim(axes[0].get_ylim())
axes[-1].set_ylim(axes[1].get_ylim())

axes[0].legend(ncol=3, frameon=False, fontsize=8, loc=(-0.12, 1.01))
loglog = True
loglog_label = '_loglog' if loglog else ''
if loglog:
    
    ylims = {'oxygen': (200, 4000), 'carbon': (40, 400)}
    yticks = {'oxygen': [200, 500, 1000, 2000, 4000], 'carbon': [40, 60, 100, 200, 300, 400]}
    
    for ax, isotope in zip(axes, isotopes):
        ax.set_yscale('log')
        
        ylim = ylims[isotope]
        ax.set_ylim(*ylim)
        # remove all ytick labels
        ax.set_yticks([])
        ax.set_yticks(yticks[isotope])
        ax.set_yticklabels([str(t) for t in yticks[isotope]])
        
xlims = (-0.6, 0.6)
axes[1].set_xlim(*xlims)
# x_param_label = x_param.split('(')[0].strip()
x_param_label = {
    'Teff (K)': 'Teff',
    '[M/H]': 'metallicity',
    '[C/H]': 'carbon_metallicity'
}[x_param]
# fig_name = base_path + f'paper/latex/figures/{main_label}_isotopes_{x_param_label}{loglog_label}.pdf'
fig_name = nat_path + f'{main_label}_isotopes_metallicity_{metallicity_ref}{loglog_label}{table_id_label}{sub10pc_label}_isoratio.pdf'
fig.savefig(fig_name)
print(f'Figure saved as {fig_name}')
plt.close(fig)