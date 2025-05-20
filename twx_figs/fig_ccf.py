""" 
Generate a model for G235+G395 with the best-fit parameters from G235 alone 
Inspect the residuals, disk emission?

date: 2024-09-17
"""
import pathlib
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
# Set global font size
plt.rcParams.update({'font.size': 10})
# pdf pages
from matplotlib.backends.backend_pdf import PdfPages
import copy

from retrieval_base.retrieval import Retrieval
import retrieval_base.auxiliary_functions as af
from retrieval_base.config import Config
import seaborn as sns

path = pathlib.Path(af.get_path())
# path_figures = pathlib.Path('/home/dario/phd/retrieval_base/twx_figs')
path_figures = pathlib.Path('/home/dario/phd/twa2x_paper/figures')

config_file = 'config_jwst.txt'
# target = 'TWA28'
w_set='NIRSpec'


def check_dir(target):
    cwd = os.getcwd()
    if target not in cwd:
        os.chdir(f'{path}/{target}')
        print(f'Changed directory to {target}')

def get_ccf_files(target,run):
    
    check_dir(target)
    
    conf = Config(path=path, target=target, run=run)(config_file)        
    ccf_path = conf.data_path.parent / 'test_plots/CCF'
    assert ccf_path.exists(), f'CCF path {ccf_path} does not exist'
    # get all .txt files in the CCF directory
    files = list(ccf_path.glob('*.txt'))
    assert len(files) > 0, f'No CCF files found in {ccf_path}'
    opacity_labels = {k[4:] : v[0][-1] for k,v in conf.opacity_params.items()}
    opacity_labels = {k:v.replace("\\log\\ ", "") for k,v in opacity_labels.items()} 
    return files, opacity_labels
runs = dict(
    TWA27A='freeslab_lbl10_G2G3_0',
    TWA28='freeslab_lbl10_G2G3_0',
            )
colors = dict(
    TWA27A='#0a74da',
    TWA28='orange',
    )

# for target, run in runs.items():
   


species_to_plot = ['13CO', 'SiO','C18O','OH', 'H2O_181','CO2', 'C17O']
n_ccf = len(species_to_plot)
fig, ax = plt.subplots(n_ccf*3, 1,
                       figsize=(4.0, n_ccf*1.65), 
                       gridspec_kw={'height_ratios': [3, 1, 0.5]*n_ccf})
lw = 1.4
# every three plots we have: ccf, residuals, empty (spacing)
# create path stroke for white outline around line
pe_line = [path_effects.withStroke(linewidth=2.0, foreground='w')]

for t, target in enumerate(runs.keys()):
    files, opacity_labels = get_ccf_files(target, runs[target])
    print(f'{len(files)} CCF files found for {target} {runs[target]}')
    
    ax_ccfs = ax[:n_ccf*3:3]
    ax_resids = ax[1:n_ccf*3:3]
    ax_spacers = ax[2:n_ccf*3:3]

    file_root = files[0].parent
    rv_max = 1000.0
    for i, s_i in enumerate(species_to_plot):
        file_s = file_root / f'RV_CCF_ACF_{s_i}.txt'
        if not file_s.exists():
            print(f'File {file_s} does not exist')
            continue
    
        rv, ccf, acf = np.loadtxt(file_s).T
        ax_ccfs[i].plot(rv, ccf, color=colors[target], alpha=0.9, lw=lw)
        ax_ccfs[i].plot(rv, acf, ls='--', color=colors[target], alpha=0.9, lw=lw)
        
        ax_resids[i].plot(rv, ccf - acf, color=colors[target], alpha=0.9, lw=lw)
        ax_spacers[i].set_visible(False)
        
        ax_ccfs[i].set_xlim(-rv_max, rv_max)
        ax_ccfs[i].set_xlabel('')
        ax_ccfs[i].set_xticks([])
        ax_resids[i].set_xlim(-rv_max, rv_max)
        ylim_res = 5.0
        ax_resids[i].set_ylim(-ylim_res, ylim_res)
        ax_resids[i].set_yticks([-ylim_res, 0, ylim_res])
        
        
        if t==0:
            label = opacity_labels[s_i]
            pe = [path_effects.withStroke(linewidth=2.5, foreground='w')]
            ax_ccfs[i].text(0.05, 0.65, label, color='k', fontsize=13, weight='bold', transform=ax_ccfs[i].transAxes,
                        path_effects=pe)
            for axi in [ax_resids[i], ax_ccfs[i]]:
                axi.axhline(0, color='k', ls='-', lw=0.5*lw, alpha=0.8, zorder=-100)
                axi.axvline(0, color='k', ls='-', lw=0.5*lw, alpha=0.8, zorder=-100)
            

# create custom manual legend with the colors of each target
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color=colors[target], lw=lw*1.5, label=target.replace('TWA', 'TWA ')) for target in runs.keys()]
ax_ccfs[0].legend(ncol=2,handles=legend_elements, loc=(0.11, 1.01), frameon=False, fontsize=10,
                  handlelength=1.4, handleheight=1.0)

ax_resids[-1].set_xlabel('RV / km s$^{-1}$')
# set common ylabel
fig.text(0.01, 0.5, 'S/N', va='center', rotation='vertical', fontsize=10)
# tight layout
# plt.show()
fig_name = f'{path_figures}/fig_ccf.pdf'
plt.savefig(fig_name, bbox_inches='tight')
print(f'Saved figure to {fig_name}')
plt.close()