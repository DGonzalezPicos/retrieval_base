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

    posterior_file = base_path + target + '/retrieval_outputs/' + run + f'/FH_posterior.npy'
    posterior_file_CH = base_path + target + '/retrieval_outputs/' + run + f'/CH_posterior.npy'
    # if not os.path.exists(posterior_file) or not cache:
    if any([not os.path.exists(posterior_file), not os.path.exists(posterior_file_CH), not cache]):

        config_file = 'config_freechem.txt'
        conf = Config(path=base_path, target=target, run=run)(config_file)

        ret = Retrieval(
                    conf=conf, 
                    evaluation=False,
                    )

        bestfit_params, posterior = ret.PMN_analyze()
        ret.get_PT_mf_envelopes(posterior=posterior)
        # get_VMRs_posterior
        # ret.Chem.get_VMRs_posterior(save_to=base_path + target + '/retrieval_outputs/' + run)
        MMW = ret.Chem.mass_fractions_posterior['MMW'].mean() if hasattr(ret.Chem, 'mass_fractions_posterior') else ret.Chem.mass_fractions['MMW']
        print(f' {target}: MMW = {MMW}')
        line_species_i = 'HF_high'
        key_i = ret.Chem.pRT_name_dict.get(line_species_i, None)
        mu = ret.Chem.read_species_info(key_i, 'mass')
        HF = ret.Chem.mass_fractions_posterior[line_species_i] * (MMW/ mu)
        
        C = ret.Chem.mass_fractions_posterior['CO_high_Sam'] * (MMW/ 28.01)
        # C += ret.Chem.mass_fractions_posterior['CO_36_high_Sam'] * (MMW/ 29.01)
        H = ret.Chem.mass_fractions['H'] * (MMW/ 1.01)
        H += ret.Chem.mass_fractions['H2'] * (MMW/ 2.02)
        
        # select altitude
        vid = HF.shape[1]//2 # vertical id
        HF = HF[:, vid]
        
        print(f' C.shape before selection = {C.shape}')
        C = C[:, vid]
        
        H_species = ['H2O', 'OH']
        # H_species =[]
        if len(H_species) > 0:
            for species in H_species:
                line_species_i = ret.Chem.pRT_name_dict_r.get(species, None)
                mu = ret.Chem.read_species_info(species, 'mass')
                print(f' {target}: mu({species}) = {mu}, {line_species_i}')
                H += np.quantile(ret.Chem.mass_fractions_posterior[line_species_i], 0.5, axis=0)[vid] * (MMW/ mu)

    
        
        param_keys = list(ret.Param.param_keys)
        assert np.sum(np.isnan(HF)) == 0, f'Found {np.sum(np.isnan(HF))} nan values in log_HF'
        
        A_F_sun = 4.40 # +- 0.25, Maiorca+2014, 
        A_F_sun_err = 0.25
        
        A_C_sun = 8.46 # +- 0.05, Asplund+2009
        A_C_sun_err = 0.04
    
        # solar scaled abundance
        # H = ret.Chem.mass_fractions['H']
        print(f' {target}: H = {H.mean()}')
        F_H_posterior = np.log10(1e12 * HF / H.mean()) - A_F_sun
        assert np.sum(np.isnan(F_H_posterior)) == 0, f'Found {np.sum(np.isnan(F_H_posterior))} nan values in F_H_posterior'
        
        np.save(posterior_file, F_H_posterior)
        print(f' {target}: Saved [F/H] posterior to {posterior_file}')
        C_H_posterior = np.log10(1e12 * C / H.mean()) - A_C_sun
        assert np.sum(np.isnan(C_H_posterior)) == 0, f'Found {np.sum(np.isnan(C_H_posterior))} nan values in C_H_posterior'
        
        np.save(posterior_file_CH, C_H_posterior)
        print(f' {target}: Saved [C/H] posterior to {posterior_file_CH}')
        
    else:
        F_H_posterior = np.load(posterior_file)
        C_H_posterior = np.load(posterior_file_CH)
        
    q=[0.16, 0.5, 0.84]
    F_H_quantiles = np.quantile(F_H_posterior, q)
    C_H_quantiles = np.quantile(C_H_posterior, q)
        

    ax_new = ax is None
    ax = ax or plt.gca()
    print(f' {target}: log HF = {F_H_quantiles[1]:.2f} +{F_H_quantiles[2]-F_H_quantiles[1]:.2f} -{F_H_quantiles[1]-F_H_quantiles[0]:.2f}\n')
    print(f' {target}: log CH = {C_H_quantiles[1]:.2f} +{C_H_quantiles[2]-C_H_quantiles[1]:.2f} -{C_H_quantiles[1]-C_H_quantiles[0]:.2f}\n')
    # add black edge to points
    xerr_i = [x,x] if xerr is None else xerr
    if isinstance(xerr, (int, float)):
        xerr_i = [x-xerr, x+xerr]
    if isinstance(xerr, list):
        xerr_i = [[x-xerr[0]], [xerr[1]-x]]
        
    fmt = 'o'
    if not load_c23:
        x = C_H_quantiles[1] # NEW: use CH instead of [M/H]
        xerr = [[C_H_quantiles[1]-C_H_quantiles[0]], [C_H_quantiles[2]-C_H_quantiles[1]]]
    
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
# ignore_more_targets = ['gl3622']
ignore_more_targets = []
ignore_targets += ignore_more_targets



# x_param = 'Teff (K)'
x_param = '[M/H]'
load_c23 = True
if load_c23:
    c23 = np.loadtxt(f'{base_path}paper/data/c23_mh.txt', dtype=object)
    c23_names = ['Gl '+n[2:] for n in c23[:,0]]
    x = dict(zip(c23_names, c23[:,1].astype(float)))
    x_err = dict(zip(c23_names, c23[:,2].astype(float)))
    
else:
    assert x_param in df.columns, f'Column {x_param} not found in {df.columns}'
    x =  dict(zip(names, [float(t.split('+-')[0]) for t in df[x_param].to_list()]))
    x_err = dict(zip(names, [float(t.split('+-')[1]) for t in df[x_param].to_list()]))


runs = dict(zip(spirou_sample.keys(), [spirou_sample[k][1] for k in spirou_sample.keys()]))

# create colormap with teff in K
# norm = plt.Normalize(min(teff.values()), 4000.0)
# cmap = plt.cm.plasma
norm = plt.Normalize(3000.0, 3900.0)
cmap = plt.cm.coolwarm_r
# add Crossfield+2019 values for Gl 745 AB: HF ratio, teff and metallicity, with errors

top = 0.96
fig, ax = plt.subplots(1,1, figsize=(4,3), sharex=True, gridspec_kw={'hspace': 0.1, 
                                                                       'wspace': 0.1,
                                                                        'left': 0.15, 
                                                                        'right': 0.76, 
                                                                        'top': top, 
                                                                        'bottom': 0.13})
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
    

# ism = (0.0, 0.10)
# ax.axhspan(ism[0]-ism[1], ism[0]+ism[1], color='green', alpha=0.2,lw=0, zorder=-1, label='ISM')
# ax.text(0.95, 0.15, 'ISM', color='darkgreen', fontsize=12, transform=ax.transAxes, ha='right', va='top')
                

# draw diagonal line
ax.plot([-1.0, 1.0], [-1.0, 1.0], color='gray', lw=0.5, ls='--', zorder=-10)
# add text


ax.set_xlabel(x_param)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Only needed for color bar

# define cbar_ax for colorbar
cbar_ax = fig.add_axes([0.80, 0.06, 0.027, top-0.06])
cbar = plt.colorbar(sm, cax=cbar_ax, orientation='vertical', aspect=1)
cbar.set_label(r'T$_{\mathrm{eff}}$ (K)')

    
    
ax.set_ylabel('[F/H]')
xlabel = '[M/H]' if load_c23 else '[C/H]'
ax.set_xlabel(xlabel)

ax.legend(ncol=1, frameon=True, fontsize=8, loc='upper left', bbox_to_anchor=(0.0, 1.0))
loglog = False
loglog_label = '_loglog' if loglog else ''

xlim = (-0.6, 0.6)
ylim = (-0.9, 0.6)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
# x_param_label = x_param.split('(')[0].strip()
# fig_name = base_path + f'paper/latex/figures/{main_label}_HFs_{x_param_label}{loglog_label}.pdf'
c23_label = '_C23' if load_c23 else ''
fig_name = nat_path + f'HF_metallicity{loglog_label}{c23_label}.pdf'
fig.savefig(fig_name)
print(f'Figure saved as {fig_name}')
plt.close(fig)