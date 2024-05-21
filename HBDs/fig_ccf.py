import numpy as np
import matplotlib.pyplot as plt
# set fontsize to 16
plt.style.use('/home/dario/phd/retrieval_base/HBDs/my_science.mplstyle')
plt.rcParams.update({'font.size': 16})
import pathlib

import os
cwd = os.getcwd()
path = pathlib.Path('/home/dario/phd/retrieval_base/')
    
from retrieval_base.retrieval import pre_processing, Retrieval
from retrieval_base.config import Config
import retrieval_base.figures as figs

out_path = pathlib.Path('/home/dario/phd/Hot_Brown_Dwarfs_Retrievals/figures/')
targets = dict(
                J1200='final_full',
                TWA28='final_full',
                J0856='final_full',
                )
# reverse dictionary
targets = dict(reversed(list(targets.items())))

colors = dict(J1200='royalblue', TWA28='seagreen', J0856='indianred')


species = {
            '13CO': 'CO_36_high',
           'H2O_181': 'H2O_181_HotWat78',
           }
labels = {
            # '13CO': '\\textsuperscript{{13}}CO',
            # 'H2O_181': 'H\\textsubscript{{2}}\\textsuperscript{{18}}O',
            '13CO': r'$^{13}$CO',
            'H2O_181': r'H$_2$$^{18}$O',
            }
fig, ax = plt.subplots(2*len(species), 
                       1,
                       figsize=(6, len(species)*3), 
                       sharex=True,
                       gridspec_kw={'height_ratios': [3,1]*len(species)},
                       tight_layout=True,
                       )

lw = 1.2
alpha =0.9
rv_max = 400.
for i, (target) in enumerate(targets.keys()):
    data_path =  path / f'{target}'
    retrieval_path = data_path / f'retrieval_outputs/{targets[target]}'

    for h, (k, v) in enumerate(species.items()):
        ccf_file = retrieval_path / f'test_data/CCF_{k}.npy'
        assert ccf_file.exists(), f'CCF file {ccf_file} does not exist.'
        rv, ccf, acf = np.load(ccf_file).T # ccf, acf in SNR, RV in km/s
        rv_mask = np.abs(rv) <= rv_max
        rv = rv[rv_mask]
        ccf = ccf[rv_mask]
        acf = acf[rv_mask]
        print(f' ccf.shape = {ccf.shape}')
        
        ax[2*h].plot(rv, ccf, label=target if h ==0 else None,
                     color=colors[target], alpha=alpha, lw=lw)
        # ax[2*h].fill_between(rv, ccf, color=colors[target], alpha=0.3)
        # ax[2*h].plot(rv, acf, ls='--', lw=1.0, color=colors[target])
        ax[2*h + 1].plot(rv, ccf-acf, color=colors[target],
                            alpha=alpha, lw=lw)
        
        snr_peak = np.max(ccf)
        if h==0:
            ax[2*h].text(0.67, 0.86 - 0.12*i, target, transform=ax[2*h].transAxes,
                            fontsize=14, weight='bold', color=colors[target])
        ax[2*h].text(0.85, 0.86 - 0.12*i, f'{snr_peak:.1f}', transform=ax[2*h].transAxes,
                        fontsize=14, weight='bold', color=colors[target])
        if i == 0:
            ax[2*h].text(0.05, 0.80, labels[k], transform=ax[2*h].transAxes,
                            fontsize=16, weight='bold')
            ax[2*h].set_ylabel('SNR')
            ax[2*h+1].set_ylabel('CCF - ACF')
            ax[2*h].axhline(0, color='k', ls='-', lw=0.5, zorder=0)
            ax[2*h+1].axhline(0, color='k', ls='-', lw=0.5, zorder=0)
    # ax[h].set_ylabel('CCF SNR')
    # ax[h].legend()
ax[-1].set(
            # xlim=(rv.min(), rv.max())
           xlabel='$\\Delta v_{\\text{rad}}$ / km s$^{-1}$',
           )
ax[-1].set_xlim(-rv_max, rv_max)
# merge ylims of ax[0] and ax[2] and ax[1] and ax[3]
ax_ccf = [ax[0], ax[2]]
ax_res = [ax[1], ax[3]]

for z, ax_list in enumerate([ax_ccf, ax_res]):
    ylims = [axi.get_ylim() for axi in ax_list]
    min_ = min([ylim[0] for ylim in ylims])
    max_ = max([ylim[1] for ylim in ylims])
    if z==0:
        max_ *= 1.15
    for axi in ax_list:
        axi.set(ylim=(min_, max_))
fig.align_ylabels()



fig.savefig(out_path / f'CCF_13CO_H218O.pdf')
print(f'Figure saved at {out_path / f"CCF_13CO_H218O.pdf"}')
# plt.show()
plt.close(fig)