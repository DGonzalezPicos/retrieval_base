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

y_labels = '[F/H]'
y_lims = (-0.5, 0.5)


def main(target, x, xerr=None, label='', ax=None, run=None, xytext=None,
         cache=True, **kwargs):
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
    # if HF == 'oxygen' and main_label == 'CO':
    #     sigma_file = test_output / f'B_sigma_C18O.dat' # contains two values: B, sigma
    #     if sigma_file.exists():
    #         print(f' {target}: Found {sigma_file}')
    #         B, sigma = np.loadtxt(sigma_file)
    #         print(f' {target}: B = {B:.2f}, sigma = {sigma:.2f}')
    #         # replace nan with 0.0
    #         sigma = 0.0 if np.isnan(sigma) else sigma

    
    posterior_file = base_path + target + '/retrieval_outputs/' + run + f'/FH_posterior.npy'
    if not os.path.exists(posterior_file) or not cache:

        config_file = 'config_freechem.txt'
        conf = Config(path=base_path, target=target, run=run)(config_file)

        ret = Retrieval(
                    conf=conf, 
                    evaluation=False,
                    )

        bestfit_params, posterior = ret.PMN_analyze()
        ret.get_PT_mf_envelopes(posterior=posterior)
        # ret.Chem.get_VMRs_posterior()
        MMW = ret.Chem.mass_fractions_posterior['MMW'].mean() if hasattr(ret.Chem, 'mass_fractions_posterior') else ret.Chem.mass_fractions['MMW']
        print(f' {target}: MMW = {MMW}')
        line_species_i = 'HF_high'
        key_i = ret.Chem.pRT_name_dict.get(line_species_i, None)
        mu = ret.Chem.read_species_info(key_i, 'mass')
        HF = ret.Chem.mass_fractions_posterior[line_species_i] * (MMW/ mu)
        
        # select altitude
        HF = HF[:, HF.shape[1]//2]
        param_keys = list(ret.Param.param_keys)
        # key = 'log_HF'
        # id = param_keys.index(key)

        # make scatter plot with one point corresponding to Teff vs log(12CO/13CO)
        # take uncertainties from posterior quantiles 
        # log_HF = posterior[:, id]
        # HF = ret.Chem.VMRs_posterior['HF']
        assert np.sum(np.isnan(HF)) == 0, f'Found {np.sum(np.isnan(HF))} nan values in log_HF'
        
        A_F_sun = 4.40 # +- 0.25, Maiorca+2014, 
        A_F_sun_err = 0.25
    
        # solar scaled abundance
        H = ret.Chem.mass_fractions['H']
        print(f' {target}: H = {H.mean()}')
        F_H_posterior = np.log10(1e12 * HF / H.mean()) - A_F_sun
        assert np.sum(np.isnan(F_H_posterior)) == 0, f'Found {np.sum(np.isnan(F_H_posterior))} nan values in F_H_posterior'
        
        np.save(posterior_file, F_H_posterior)
        print(f' {target}: Saved [F/H] posterior to {posterior_file}')
    else:
        F_H_posterior = np.load(posterior_file)
        
    q=[0.16, 0.5, 0.84]
    F_H_quantiles = np.quantile(F_H_posterior, q)
        

    ax_new = ax is None
    ax = ax or plt.gca()
    print(f' {target}: log HF = {F_H_quantiles[1]:.2f} +{F_H_quantiles[2]-F_H_quantiles[1]:.2f} -{F_H_quantiles[1]-F_H_quantiles[0]:.2f}\n')
    # add black edge to points
    xerr = [x,x] if xerr is None else xerr
    if isinstance(xerr, (int, float)):
        xerr = [x-xerr, x+xerr]
    if isinstance(xerr, list):
        xerr = [[x-xerr[0]], [xerr[1]-x]]
        
    fmt = 'o'
    # add errorbar style with capsize
    if sigma > 3.0:
        ax.errorbar(x, F_H_quantiles[1],
                    xerr=xerr,
                    yerr=[[F_H_quantiles[1]-F_H_quantiles[0]], [F_H_quantiles[2]-F_H_quantiles[1]]],
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
    else:
        # plot lower limit
        fmt = '^'
        ax.errorbar(x, F_H_quantiles[0],
                    xerr=xerr,
                    yerr=[[0.0], [F_H_quantiles[2]-F_H_quantiles[0]]],
                    lolims=True,
                    fmt=fmt, 
                    label=label.replace('gl', 'Gl '),
                    alpha=0.9,
                        # markerfacecolor='none',  # Make the inside of the marker transparent (optional)
                    markeredgecolor='k', # Black edge color
                    markeredgewidth=0.8,     # Thickness of the edge
                    color=kwargs.get('color', 'k'),
        )
        
        
    if xytext is not None:
        # add text with target name next to point, offset text from point
        ax.annotate(label.replace('gl', 'Gl '), (x, F_H_quantiles[1]), textcoords="offset points", xytext=xytext, ha='left',
                    fontsize=8, color=kwargs.get('color', 'k'), alpha=0.9)
        
    return F_H_quantiles
        


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
# norm = plt.Normalize(min(teff.values()), 4000.0)
# cmap = plt.cm.plasma
norm = plt.Normalize(2900, 3900.0)
cmap = plt.cm.coolwarm_r
# add Crossfield+2019 values for Gl 745 AB: HF ratio, teff and metallicity, with errors

top = 0.92
fig, ax = plt.subplots(1,1, figsize=(4,3), sharex=True, gridspec_kw={'hspace': 0.1, 
                                                                       'wspace': 0.1,
                                                                        'left': 0.15, 
                                                                        'right': 0.78, 
                                                                        'top': top, 
                                                                        'bottom': 0.07})
# ylim_min = 50.0
# ylim_max = 3000.0

xytext = {'Gl 699' : (-28,5),
        #   'Gl 411' : (3,3),
        #   'Gl 382': (-20,-12),
        #   'Gl 1286': (2,-12),
          
}

sun = (0.0, 0.0)
# ax.axhspan(sun[0]-sun[1], sun[0]+sun[1], color='gold', alpha=0.3, label='Solar',lw=0)
ax.plot(0.0, sun[0], color='gold', marker='*', ms=16, label='Sun', alpha=0.8, markeredgecolor='black', markeredgewidth=0.8, zorder=100)


plot_teff_max = 4400.0

testing = False
cache = True
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
                    x[name], 
                    xerr=x_err[name],
                    ax=ax, 
                    # label=name, 
                    label='',
                    run=None,
                    color=color,
                    xytext=xytext.get(name, None),
                    cache=cache,
    )
    
    if testing:
        break
    

ism = (0.0, 0.10)
ax.axhspan(ism[0]-ism[1], ism[0]+ism[1], color='green', alpha=0.2,lw=0, zorder=-1, label='ISM')
# ax.text(0.95, 0.15, 'ISM', color='darkgreen', fontsize=12, transform=ax.transAxes, ha='right', va='top')
                
    
if x_param == '[M/H]':
    ax.axvline(0.0, color='k', lw=0.5, ls='--', zorder=-10)

# if i == 0:

ax.set_xlabel(x_param)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Only needed for color bar

# define cbar_ax for colorbar
cbar_ax = fig.add_axes([0.80, 0.06, 0.027, top-0.06])
cbar = plt.colorbar(sm, cax=cbar_ax, orientation='vertical', aspect=1)
cbar.set_label(r'T$_{\mathrm{eff}}$ (K)')

    
    
ax.set_ylabel('[F/H]')
    

ax.legend(ncol=3, frameon=False, fontsize=8, loc=(-0.12, 1.01))
loglog = False
loglog_label = '_loglog' if loglog else ''
if loglog:
    
    ylims = (-0.5, 0.5)
    yticks = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    ax.set_yscale('log')
    
    ax.set_ylim(ylims)
    
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(t) for t in yticks])
else:
    # ax.set_ylim(ylim_min, ylim_max)
    pass
    # ax.set_xlim(xlim)
# x_param_label = x_param.split('(')[0].strip()
x_param_label = {
    'Teff (K)': 'Teff',
    '[M/H]': 'metallicity',
}[x_param]
# fig_name = base_path + f'paper/latex/figures/{main_label}_HFs_{x_param_label}{loglog_label}.pdf'
fig_name = nat_path + f'HF_{x_param_label}{loglog_label}.pdf'
fig.savefig(fig_name)
print(f'Figure saved as {fig_name}')
plt.close(fig)