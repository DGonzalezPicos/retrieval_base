""" copy of isotopes_metallicity_horizontal.py """
from retrieval_base.retrieval import Retrieval
import retrieval_base.figures as figs
from retrieval_base.config import Config
from retrieval_base.auxiliary_functions import spirou_sample, read_spirou_sample_csv, find_run, load_romano_models, axhspan_gradient
# import config_freechem as conf
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import matplotlib.patheffects as pe
import scienceplots
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch

# reset to default
plt.style.use('default')
# plt.style.use(['latex-sans'])
plt.style.use(['sans'])
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.linewidth'] = 1.0

font_size = 7
plt.rcParams['font.size'] = font_size

# Ensure RGB color mode for Nature requirements
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# ─── ensure text stays as text ────────────────────────────────────────────────
plt.rcParams['pdf.fonttype']   = 42
plt.rcParams['ps.fonttype']    = 42
plt.rcParams['svg.fonttype']   = 'none'
plt.rcParams['text.usetex']    = False

# enable latex
# plt.rcParams['text.usetex'] = True


# Ensure RGB color mode for Nature requirements
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

base_path = '/home/dario/phd/retrieval_base/'
nat_path = '/home/dario/phd/nat/'

df = read_spirou_sample_csv()
flip_rows = True
if flip_rows:
    df = df.iloc[::-1]
names = df['Star'].to_list()
teff =  dict(zip(names, [float(t.split('+-')[0]) for t in df['Teff (K)'].to_list()]))
spt = dict(zip(names, [t.split('+-')[0] for t in df['SpT'].to_list()]))
# dist = dict(zip(names, [float(t) for t in df['Distance (pc)'].to_list()]))
norm = plt.Normalize(3000.0, 3900.0)
cmap = plt.cm.coolwarm_r

water = False # take isotope ratio from H2O
main_label = 'H2O' if water else 'CO'
isotope = 'oxygen' 
assert isotope in ['carbon', 'oxygen'], f'Isotope {isotope} not recognized (choose from oxygen, carbon)'
y_labels = {'oxygen': r'$^{16}$O/$^{18}$O', 'carbon': r'$^{12}$C/$^{13}$C'}
y_lims = {'oxygen': (30, 4000), 'carbon': (20, 400)}

sigma_colors = {
                '3':'k',
                '2': '#0C823E',
                '1': '#ff6a90'
                }

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
                    markeredgecolor=sigma_colors['2'], # Black edge color
                    markeredgewidth=0.8,     # Thickness of the edge
                    capsize=2,               # Size of the cap on error bars
                    capthick=0.8,             # Thickness of the cap on error bars
                    ecolor='gray',          # Color of the error bars, set alpha of ecolor to make it transparent
                    elinewidth=0.8,           # Thickness of the error bars
                    
                    
                    color=kwargs.get('color', 'k'),
        )
    elif sigma > 1.0:
        ax.errorbar(x, isotope_quantiles[1],
                    xerr=xerr,
                    yerr=[[isotope_quantiles[1]-isotope_quantiles[0]], [isotope_quantiles[2]-isotope_quantiles[1]]],
                    fmt=fmt, 
                    label=label.replace('gl', 'Gl '),
                    alpha=0.96,
                        # markerfacecolor='none',  # Make the inside of the marker transparent (optional)
                    markeredgecolor=sigma_colors['1'], # Black edge color
                    markeredgewidth=0.8,     # Thickness of the edge
                    capsize=2,               # Size of the cap on error bars
                    capthick=0.8,             # Thickness of the cap on error bars
                    ecolor='gray',          # Color of the error bars, set alpha of ecolor to make it transparent
                    elinewidth=0.8,           # Thickness of the error bars
                    
                    
                    color=kwargs.get('color', 'k'),
        )
    else:
        # # plot lower limit
        # fmt = '^'
        # ax.errorbar(x, isotope_quantiles[0],
        #             xerr=xerr,
        #             yerr=[[0.0], [0.4*(isotope_quantiles[2]-isotope_quantiles[1])]],
        #             lolims=True,
        #             fmt=fmt, 
        #             label=label.replace('gl', 'Gl '),
        #             alpha=0.9,
        #                 # markerfacecolor='none',  # Make the inside of the marker transparent (optional)
        #             markeredgecolor='k', # Black edge color
        #             markeredgewidth=0.8,     # Thickness of the edge
        #             color=kwargs.get('color', 'k'),
        # )
        print(f' {target} not plotted... sigma = {sigma:.2f}')
        pass
        
        
    if xytext is not None:
        # add text with target name next to point, offset text from point
        ax.annotate(label.replace('gl', 'Gl '), (x, isotope_quantiles[1]), textcoords="offset points", xytext=xytext, ha='left',
                    fontsize=font_size, color=kwargs.get('color', 'k'), alpha=0.9)
        
    return isotope_quantiles
        



# valid = dict(zip(names, df['Valid'].to_list()))
ignore_targets = []

# x_param = '[C/H]'
x_param = '[M/H]'

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


# add Crossfield+2019 values for Gl 745 AB: isotope ratio, teff and metallicity, with errors
crossfield_dict = {'oxygen': {'Gl 745 A': [(1220, 260), (3454, 31), (-0.43, 0.05)],
                        'Gl 745 B': [(1550, 360), (3440, 31), (-0.39, 0.05)]},
              'carbon': {'Gl 745 A': [(296, 45), (3454, 31), (-0.43, 0.05)],
                        'Gl 745 B': [(224, 26), (3440, 31), (-0.39, 0.05)]}
}
sun_dict = {'oxygen': (511, 10),# solar wind McKeegan et al. 2011
            'carbon': (91.4, 1.3)}
ism_dict = {'oxygen': (557, 30), # ISM value from Wilson et al. 1999
            'carbon': (68.0, 14.0)}

plot_crossfield = True

top = 0.92

width_mm = 180.0
aspect_ratio = 4/4
height_mm = width_mm/aspect_ratio
inch_to_mm = 25.4
fig = plt.figure(figsize=(width_mm/inch_to_mm, height_mm/inch_to_mm))  # Adjust the figure size as needed
gs = fig.add_gridspec(10, 12, hspace=0.10, wspace=0.0)

ax_spectrum = fig.add_subplot(gs[0:3, :])
ax_residuals = fig.add_subplot(gs[3, :])



ax_carbon = fig.add_subplot(gs[5:, :5])  # Last 7 rows, half the width
ax_oxygen = fig.add_subplot(gs[5:, 7:])  # Last 7 rows, half the width
axes = [ax_carbon, ax_oxygen]

from bestfit_model_nat import plot
my_targets_id = ['338B', '205', '411', '436','699', '1286']
my_targets = ['gl'+t for t in my_targets_id]
# xlim = (2285, 2364)
# text_x = (xlim[0]+1., xlim[1]-3)
order = 0
# xlim = (2282, 2364)
xlim = (2342.01, 2359.99)
text_x = (
            # xlim[0]+0.4, 
            xlim[0]+0.2,
          xlim[1]+2.0,
          )
kwargs = {'show_lines': True, 'lw': 0.8}
plot(0, names,
     my_targets, 
     text_x=text_x, 
    text_y_offset=0.15,
     xlim=xlim, 
     add_cbar=False, 
     teff=teff,
     spt=spt,
     axes=[ax_spectrum, ax_residuals],
     include_path_effects=False,
      cmap=cmap,
      norm=norm, 
      **kwargs)
ax_spectrum.set_xticklabels([])

# add handles for subplots: a, b, c
thandles = ['a', 'b', 'c']
x_text_label = -0.08
y_offset_label =0.09
fig.text(x_text_label, 0.95 + y_offset_label, thandles[0], transform=ax_spectrum.transAxes, fontsize=font_size, ha='left', va='top', weight='bold')
# DGP 2025-07-10: change location of labels for consistency with a)
# fig.text(0.35, -0.85, thandles[1], transform=ax_spectrum.transAxes, fontsize=12, ha='left', va='top', weight='bold')
# fig.text(0.95, -0.85, thandles[2], transform=ax_spectrum.transAxes, fontsize=12, ha='left', va='top', weight='bold')
fig.text(x_text_label, -0.75 + y_offset_label, thandles[1], transform=ax_spectrum.transAxes, fontsize=font_size, ha='left', va='top', weight='bold')
fig.text(x_text_label+0.58, -0.75 + y_offset_label, thandles[2], transform=ax_spectrum.transAxes, fontsize=font_size, ha='left', va='top', weight='bold')

xytext = {'Gl 699' : (-28,5),
        #   'Gl 411' : (3,3),
        #   'Gl 382': (-20,-12),
        #   'Gl 1286': (2,-12),
          
}
isotopes = ['carbon', 'oxygen']
ism_label_strs = {'carbon': 'ISM', 'oxygen': 'ISM'}
sun_label_strs = {'carbon': 'Sun', 'oxygen': 'Sun'}
for i, isotope in enumerate(isotopes):
    print(f' ** Isotope {isotope} **')
    ax = axes[i]
    ax.set_ylim(*y_lims[isotope])

    crossfield = crossfield_dict[isotope]
    ism = ism_dict[isotope]
    sun = sun_dict[isotope]

    # ax.axhspan(sun[0]-sun[1], sun[0]+sun[1], color='gold', alpha=0.3, label='Solar',lw=0)
    decimal_places = 0 if isotope == 'oxygen' else 1
    sun_label = sun_label_strs[isotope] + f'\n({sun[0]:.{decimal_places}f} ± {sun[1]:.{decimal_places}f})'
    ax.plot(0.0, sun[0], color='gold', marker='*', ms=16, label=sun_label, alpha=0.8, markeredgecolor='black', markeredgewidth=0.8, zorder=100)
    

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
                            xytext=xytext.get(name, None))
        except Exception as e:
            print(e)
            print(f'---> Skipping {name}')
            continue


    # ax.axhspan(ism[0]-ism[1], ism[0]+ism[1], color='green', alpha=0.2,lw=0, zorder=-1, label='ISM')
    x_span = np.linspace(-0.4, 0.6, 100)
    rgb_color = np.array([10, 191, 134]) / 255.0 # light green
    rgb_color *= 0.7
    
    ism_label_strs[isotope] = 'ISM' + f'\n({ism[0]:.0f} ± {ism[1]:.0f})'
    poly, ism_label = axhspan_gradient(ax, x_span, y_range=(ism[0]-ism[1], ism[0]+ism[1]), rgb_color=rgb_color, gamma=3, n=120,
                            label=ism_label_strs[isotope])
    
    # ax.text(0.95, 0.15, 'ISM', color='darkgreen', fontsize=12, transform=ax.transAxes, ha='right', va='top')
   

    # plot crossfield values
    if plot_crossfield:
        for cross_i, (k, v) in enumerate(crossfield.items()):
            teff_cf = v[1][0]
            color = cmap(norm(teff_cf))
            x_cf = v[1][0] if x_param == 'Teff (K)' else v[2][0]
            x_err_cf = v[1][1] if x_param == 'Teff (K)' else v[2][1]
            fmt = 's' if cross_i == 0 else 'D'
            ax.errorbar(x_cf, v[0][0], xerr=x_err_cf, yerr=v[0][1], fmt=fmt, label=k.replace('Gl ', 'GJ '), color=color, markeredgecolor='black', markeredgewidth=0.8)

            # add thin arrow pointing to the marker with the name of the target
            annotate = False
            if annotate:
                ax.annotate(k, (x_cf, v[0][0]), textcoords="offset points", 
                            xytext=(60,5), ha='center', va='center',
                            arrowprops=dict(facecolor='black', shrink=1, headwidth=1, width=0.5, headlength=0.1),
                            fontsize=font_size,
                            horizontalalignment='right', verticalalignment='top')
                    
        
    # if x_param == '[M/H]':
    #     ax.axvline(0.0, color='k', lw=0.5, ls='--', zorder=-10)

    # if i == 0:
    ax.set_xlabel(x_param, fontsize=font_size)
    
    # Ensure tick labels follow journal requirements
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(labelsize=font_size)

    if i == 1:
        
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Only needed for color bar
        
        # define cbar_ax for colorbar
        # x, y = 0.912, 0.11
        x, y = 0.912, 0.654
        # w, h = 0.027, top-y*1.36
        w, h = 0.010,top-y*1.061
        cbar_ax = fig.add_axes([x, y, w, h])
        cbar = plt.colorbar(sm, cax=cbar_ax, orientation='vertical', aspect=1)
        cbar.set_label(r'T$_{\mathrm{eff}}$ (K)', fontsize=font_size)
        
        # Ensure colorbar tick labels follow requirements
        cbar.ax.tick_params(labelsize=font_size)
        
        
    ax.set_ylabel(y_labels[isotope], fontsize=font_size)
    
# load Romano+2022 models
mass_ranges = ['1_8', '3_8']

gce_colors = ['black', 'purple']
# Remove path_effects to ensure editable text as required by Nature
# path_effects = [pe.Stroke(linewidth=2.5, foreground='white'), pe.Normal()]

for i, mass_range in enumerate(mass_ranges):
    Z, c12c13, o16o18 = load_romano_models(Z_min=-0.7, mass_range=mass_range)
    # mass_range_label = 'R22 (' + mass_range.replace('_', '-') + r' M$_\odot$)'
    mass_range_label = mass_range.replace('_', '-') + r' M$_\odot$'
    axes[0].plot(Z, c12c13, color=gce_colors[i], lw=1.5, label=mass_range_label, alpha=0.8)
    axes[1].plot(Z, o16o18, color=gce_colors[i], lw=1.5, label=mass_range_label, alpha=0.8)
    

axes[0].legend(ncol=3)
axes[1].legend(ncol=3)

# add handle of ism_label to existing legend
handles, labels = axes[0].get_legend_handles_labels()
ism_label.set_alpha(0.65)
ism_label.set_linewidth(0.0)
handles.insert(1, ism_label)
labels.insert(1, ism_label_strs['carbon'])
# 
axes[0].legend(handles, labels, ncol=1, 
               frameon=True, 
               framealpha=0.4,
               fontsize=font_size, loc=(0.605, 0.57)) # shortcbar

handles, labels = axes[1].get_legend_handles_labels()
ism_label.set_alpha(0.65)
ism_label.set_linewidth(0.0)
handles.insert(1, ism_label)
labels.insert(1, ism_label_strs['oxygen'])
# remove handle and label 3 from axes[1]
handles.pop(2)
labels.pop(2)
axes[1].legend(handles, labels, ncol=1, frameon=False, fontsize=font_size, loc=(1.01, 0.64)) # shortcbar


# create another legend for the sigma values with circles
add_sigma_legend = True
if add_sigma_legend:
    sigma_handles = []
    sigma_labels = []
    from matplotlib.lines import Line2D
    for sigma in ['3', '2', '1']:
        # Create a circle patch for the legend
        sigma_handles.append(Line2D([0], [0], marker='o', color='w', markeredgecolor=sigma_colors[sigma], markersize=6, markeredgewidth=0.9))
        if sigma == '3':
            sigma_labels.append('>99.7%')
        elif sigma == '2':
            sigma_labels.append('95.4-99.7%')
        else:
            sigma_labels.append('68.3-95.4%')

    # Create a new axes just for the sigma legend
    sigma_legend_ax = fig.add_axes([0.60, 0.115, 0.1, 0.1])  # [left, bottom, width, height]
    # sigma_legend_ax.set_visible(False)  # Hide the axes but keep it for the legend
    # hide the spines and ticks
    sigma_legend_ax.spines['top'].set_visible(False)
    sigma_legend_ax.spines['right'].set_visible(False)
    sigma_legend_ax.spines['bottom'].set_visible(False)
    sigma_legend_ax.spines['left'].set_visible(False)
    sigma_legend_ax.set_xticks([])
    sigma_legend_ax.set_yticks([])
    
    # Create legend with custom handler map to ensure circles are drawn properly
    legend = sigma_legend_ax.legend(sigma_handles, sigma_labels, 
                                   framealpha=0.4,
                                   fontsize=font_size,
                                   title='Confidence interval',
                                   loc='center')

    # Make sure the circles appear round in the legend
    legend.get_frame().set_linewidth(0.5)

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
for ax in axes:
    ax.set_xlim(*xlims)
# x_param_label = x_param.split('(')[0].strip()
x_param_label = {
    'Teff (K)': 'Teff',
    '[M/H]': 'metallicity',
    '[C/H]': 'carbon_metallicity'
}[x_param]
# fig_name = base_path + f'paper/latex/figures/{main_label}_isotopes_{x_param_label}{loglog_label}.pdf'
# fig_name = nat_path + f'{main_label}_isotopes_metallicity_{metallicity_ref}{loglog_label}{table_id_label}_horizontal.pdf'
fig_name = nat_path + f'nat_fig_3_revised.pdf'
# Save in RGB mode with editable text as required by Nature
fig.savefig(fig_name, bbox_inches='tight', dpi=300, format='pdf', 
            facecolor='white', edgecolor='none', 
            metadata={'Creator': 'Dario Gonzalez Picos', 
                      'Producer': 'matplotlib'})
print(f'Figure saved as {fig_name}')
plt.close(fig)