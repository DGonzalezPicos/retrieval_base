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
# enable latex
# plt.rcParams['text.usetex'] = True
plt.rcParams.update({
    "font.size": 8,
})

base_path = '/home/dario/phd/retrieval_base/'
# nat_path = '/home/dario/phd/nat/figures/'
presentation_path = '/home/dario/phd/presentations/apr2_mdwarfs/'

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
                    fontsize=8, color=kwargs.get('color', 'k'), alpha=0.9)
        
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
sun_dict = {'oxygen': (529.7, 1.7),# solar wind McKeegan et al. 2011
            'carbon': (93.5, 3.0)}
ism_dict = {'oxygen': (557, 30), # ISM value from Wilson et al. 1999
            'carbon': (68.0, 14.0)}

plot_crossfield = True

top = 0.92

# Global variables
isotopes = ['carbon', 'oxygen']
plot_teff_max = 4400.0
xlims = (-0.6, 0.6)
xytext = {'Gl 699' : (-28,5)}
mass_ranges = ['1_8', '3_8']
gce_colors = ['black', 'purple']

def setup_figure():
    """Create and setup the base figure with subplots."""
    fig = plt.figure(figsize=(9, 3))
    gs = fig.add_gridspec(5, 5, hspace=0.00, wspace=0.0)
    ax_carbon = fig.add_subplot(gs[:, :2])
    ax_oxygen = fig.add_subplot(gs[:, 3:])
    return fig, [ax_carbon, ax_oxygen]

def plot_sun(ax, sun):
    """Plot the Sun marker."""
    ax.plot(0.0, sun[0], color='gold', marker='*', ms=16, label='Sun', 
            alpha=0.8, markeredgecolor='black', markeredgewidth=0.8, zorder=100)

def plot_ism(ax, ism):
    """Plot the ISM band."""
    x_span = np.linspace(-0.4, 0.6, 100)
    rgb_color = np.array([10, 191, 134]) / 255.0
    rgb_color *= 0.7
    return axhspan_gradient(ax, x_span, y_range=(ism[0]-ism[1], ism[0]+ism[1]), 
                          rgb_color=rgb_color, gamma=3, n=120, label='ISM')

def plot_crossfield(ax, crossfield, x_param):
    """Plot Crossfield measurements."""
    for cross_i, (k, v) in enumerate(crossfield.items()):
        teff_cf = v[1][0]
        color = cmap(norm(teff_cf))
        x_cf = v[1][0] if x_param == 'Teff (K)' else v[2][0]
        x_err_cf = v[1][1] if x_param == 'Teff (K)' else v[2][1]
        fmt = 's' if cross_i == 0 else 'D'
        ax.errorbar(x_cf, v[0][0], xerr=x_err_cf, yerr=v[0][1], fmt=fmt, 
                   label=k.replace('Gl ', 'GJ '), color=color, 
                   markeredgecolor='black', markeredgewidth=0.8)

def plot_romano_models(axes, mass_ranges, gce_colors):
    """Plot Romano models on both axes."""
    path_effects = [pe.Stroke(linewidth=2.5, foreground='white'), pe.Normal()]
    for j, mass_range in enumerate(mass_ranges):
        Z, c12c13, o16o18 = load_romano_models(Z_min=-0.7, mass_range=mass_range)
        mass_range_label = mass_range.replace('_', '-') + r' M$_\odot$'
        axes[0].plot(Z, c12c13, color=gce_colors[j], lw=1.5, 
                    label=mass_range_label, alpha=0.8, path_effects=path_effects)
        axes[1].plot(Z, o16o18, color=gce_colors[j], lw=1.5, 
                    label=mass_range_label, alpha=0.8, path_effects=path_effects)

def setup_axes(axes, isotopes, y_labels, x_param, loglog=True):
    """Setup axes properties."""
    if loglog:
        ylims = {'oxygen': (200, 4000), 'carbon': (40, 400)}
        yticks = {'oxygen': [200, 500, 1000, 2000, 4000], 
                 'carbon': [40, 60, 100, 200, 300, 400]}
        
        for ax, isotope in zip(axes, isotopes):
            ax.set_yscale('log')
            ylim = ylims[isotope]
            ax.set_ylim(*ylim)
            ax.set_yticks([])
            ax.set_yticks(yticks[isotope])
            ax.set_yticklabels([str(t) for t in yticks[isotope]])
            ax.set_xlabel(x_param)
            ax.set_ylabel(y_labels[isotope])
            ax.set_xlim(*xlims)

def add_colorbar(fig, axes, norm, cmap):
    """Add colorbar to the figure."""
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes, orientation='vertical')
    cbar.set_label(r'T$_{\mathrm{eff}}$ (K)')

def save_frame(fig, frame_dir, counter, dpi=300):
    """Save frame as PNG with specified DPI."""
    frame_path = os.path.join(frame_dir, f'frame_{counter:03d}.png')
    fig.savefig(frame_path, bbox_inches='tight', dpi=dpi)
    return frame_path

def create_gif(frame_dir, output_path, duration=1000):
    """Create GIF from PNG frames using imageio."""
    import imageio.v2 as imageio
    import glob
    
    # Get list of frames in order
    frames = sorted(glob.glob(os.path.join(frame_dir, 'frame_*.png')))
    
    # Read all frames
    images = [imageio.imread(frame) for frame in frames]
    
    # Save as GIF
    imageio.mimsave(output_path, images, duration=duration)
    print(f'Created GIF at {output_path}')

def setup_legends(axes, ism_label):
    """Setup legends for both plots including sigma confidence intervals."""
    from matplotlib.lines import Line2D

    # Create sigma legend handles
    sigma_handles = []
    sigma_labels = []
    for sigma in ['3', '2', '1']:
        sigma_handles.append(Line2D([0], [0], marker='o', color='w', 
                                  markeredgecolor=sigma_colors[sigma], 
                                  markersize=6, markeredgewidth=0.9))
        if sigma == '3':
            sigma_labels.append(f'$\geq${int(sigma)}$\sigma$')
        else:
            sigma_labels.append(f'{int(sigma)}$\sigma$ - {int(sigma)+1}$\sigma$')

    # Setup main legend for left plot (carbon)
    handles, labels = axes[0].get_legend_handles_labels()
    
    # Modify ISM label appearance
    ism_label.set_alpha(0.65)
    ism_label.set_linewidth(0.0)
    
    # Add ISM to legend
    handles.insert(1, ism_label)
    labels.insert(1, 'ISM')
    
    # Add sigma legend to both plots
   
    # Add sigma confidence intervals legend
    sigma_legend = axes[1].legend(sigma_handles, sigma_labels,
                            framealpha=0.4,
                            fontsize=7,
                            loc=(0.0,1.01),
                            ncol=3)
    sigma_legend.get_frame().set_linewidth(0.5)
    
        
    # Add main legend
    axes[0].legend(handles, labels, 
                ncol=2, frameon=False, fontsize=8, 
                loc=(0.15, 1.01))
    
    # Add both legends
    # axes[1].add_artist(sigma_legend)

def generate_frames():
    """Generate all frames for the animation."""
    # Create frames directory
    frames_dir = os.path.join(presentation_path, 'frames')
    os.makedirs(frames_dir, exist_ok=True)
    frame_counter = 0

    # First frame: Sun, ISM, and Crossfield only
    fig, axes = setup_figure()
    for i, isotope in enumerate(isotopes):
        ax = axes[i]
        ax.set_ylim(*y_lims[isotope])
        
        # Plot baseline elements
        plot_sun(ax, sun_dict[isotope])
        _, ism_label = plot_ism(ax, ism_dict[isotope])
        if plot_crossfield:
            plot_crossfield(ax, crossfield_dict[isotope], x_param)
    
    setup_axes(axes, isotopes, y_labels, x_param, loglog=True)
    setup_legends(axes, ism_label)  # Add legends
    add_colorbar(fig, axes, norm, cmap)
    save_frame(fig, frames_dir, frame_counter)
    frame_counter += 1
    plt.close(fig)

    # Add one target at a time
    for name in names:
        if teff[name] > plot_teff_max:
            continue
        target = name.replace('Gl ', 'gl')
        if name in ignore_targets:
            continue

        fig, axes = setup_figure()
        
        # Plot baseline for both panels
        for i, isotope in enumerate(isotopes):
            ax = axes[i]
            ax.set_ylim(*y_lims[isotope])
            
            # Plot baseline elements
            plot_sun(ax, sun_dict[isotope])
            _, ism_label = plot_ism(ax, ism_dict[isotope])
            if plot_crossfield:
                plot_crossfield(ax, crossfield_dict[isotope], x_param)
        
        # Plot all previous targets
        for prev_name in names[:names.index(name)]:
            if prev_name not in ignore_targets and teff[prev_name] <= plot_teff_max:
                prev_target = prev_name.replace('Gl ', 'gl')
                color = cmap(norm(teff[prev_name]))
                try:
                    for i, isotope in enumerate(isotopes):
                        main(prev_target, isotope, x[prev_name], 
                             xerr=x_err[prev_name], ax=axes[i], 
                             label='', run=None, color=color,
                             xytext=xytext.get(prev_name, None))
                except Exception as e:
                    print(f'Error plotting {prev_name}: {e}')
                    continue
        
        # Plot current target
        color = cmap(norm(teff[name]))
        try:
            for i, isotope in enumerate(isotopes):
                main(target, isotope, x[name], 
                     xerr=x_err[name], ax=axes[i], 
                     label='', run=None, color=color,
                     xytext=xytext.get(name, None))
        except Exception as e:
            print(f'Error plotting {name}: {e}')
            continue
            
        setup_axes(axes, isotopes, y_labels, x_param, loglog=True)
        setup_legends(axes, ism_label)  # Add legends
        add_colorbar(fig, axes, norm, cmap)
        save_frame(fig, frames_dir, frame_counter)
        frame_counter += 1
        plt.close(fig)

    # Final frame: add Romano models
    fig, axes = setup_figure()
    
    # Plot baseline and all targets
    for i, isotope in enumerate(isotopes):
        ax = axes[i]
        ax.set_ylim(*y_lims[isotope])
        
        # Plot baseline elements
        plot_sun(ax, sun_dict[isotope])
        _, ism_label = plot_ism(ax, ism_dict[isotope])
        if plot_crossfield:
            plot_crossfield(ax, crossfield_dict[isotope], x_param)
    
    # Plot all targets
    for name in names:
        if teff[name] > plot_teff_max or name in ignore_targets:
            continue
        target = name.replace('Gl ', 'gl')
        color = cmap(norm(teff[name]))
        try:
            for i, isotope in enumerate(isotopes):
                main(target, isotope, x[name], 
                     xerr=x_err[name], ax=axes[i], 
                     label='', run=None, color=color,
                     xytext=xytext.get(name, None))
        except Exception as e:
            print(f'Error plotting {name}: {e}')
            continue
    
    # Add Romano models
    plot_romano_models(axes, mass_ranges, gce_colors)
    
    setup_axes(axes, isotopes, y_labels, x_param, loglog=True)
    setup_legends(axes, ism_label)  # Add legends
    add_colorbar(fig, axes, norm, cmap)
    
    save_frame(fig, frames_dir, frame_counter)
    frame_counter += 1
    plt.close(fig)
    
    print(f'Generated {frame_counter} frames in {frames_dir}')
    
    # Create GIF from frames
    gif_path = os.path.join(presentation_path, f'{main_label}_isotopes_animation.gif')
    create_gif(frames_dir, gif_path)

if __name__ == '__main__':
    generate_frames()