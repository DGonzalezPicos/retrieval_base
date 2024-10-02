from retrieval_base.retrieval import Retrieval
import retrieval_base.figures as figs
from retrieval_base.config import Config
# import config_freechem as conf
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib

base_path = '/home/dario/phd/retrieval_base/'

target = 'gl880'

def main(target, ax, orders=[0], offset=0.0, run=None, **kwargs):
    
    # assert len(ax) ==3 if ax is not None else True, f'Lenght of ax must be 3, not {len(ax)}'
    ax = np.atleast_1d(ax)
    assert len(ax) == len(orders), f'Lenght of ax must be {len(orders)}, not {len(ax)}'

    if target not in os.getcwd():
        os.chdir(base_path + target)

    outputs = pathlib.Path(base_path) / target / 'retrieval_outputs'
    # find dirs in outputs
    print(f' outputs = {outputs}')
    dirs = [d for d in outputs.iterdir() if d.is_dir() and 'sphinx' in d.name and '_' not in d.name]
    runs = [int(d.name.split('sphinx')[-1]) for d in dirs]
    # run = 'sphinx'+str(max(runs))
    if run is None:
        run = 'sphinx'+str(max(runs))
    else:
        run = 'sphinx'+str(run)
        assert run in [d.name for d in dirs], f'Run {run} not found in {dirs}'
    print('Run with largest number:', run)

    config_file = 'config_freechem.txt'
    conf = Config(path=base_path, target=target, run=run)(config_file)

    ret = Retrieval(
                conf=conf, 
                evaluation=False,
                )

    bestfit_params, posterior = ret.PMN_analyze()
    ret.evaluate_model(bestfit_params)
    ret.PMN_lnL_func()
    
    rv = bestfit_params[list(ret.Param.param_keys).index('rv')]
    
    wave = np.squeeze(ret.d_spec['spirou'].wave) * (1 - rv/299792.458)
    flux = np.squeeze(ret.d_spec['spirou'].flux) + offset
    
    s = ret.LogLike['spirou'].s
    
    # err  = [ret.Cov['spirou'][i][0].err * s[i] for i in range(3)]
    
    m = np.squeeze(ret.LogLike['spirou'].m) + offset
    # residuals, save as npy with wave, residuals, err
    # np.save(ret.conf_output + 'residuals.npy', np.array([wave, flux-m, ret.Cov['spirou'][0].err * s[0]]))
    
    
    lw = kwargs.get('lw', 1.0)
    color = kwargs.get('color', 'orange')
    for i, order in enumerate(orders):
        err_i = np.ones_like(wave[order]) * np.nan
        
        mask_i = ret.d_spec['spirou'].mask_isfinite[order,0]
        err_i[mask_i] = ret.Cov['spirou'][order][0].err * s[order]
        
        residuals_i = flux[order] - m[order]
        file_name = ret.conf_output + f'residuals_{order}.npy'
        np.save(file_name, np.array([wave[order], residuals_i, err_i]))
        print(f'Residuals saved as {file_name}')
        
        
        ax[i].plot(wave[order], flux[order], color='k',lw=lw)
        ax[i].fill_between(wave[order], flux[order]-err_i, flux[order]+err_i, alpha=0.2, color='k', lw=0)
        ax[i].plot(wave[order], m[order], label=target,lw=lw, color=color)
        
        # add text above spectra in units of data

        text_pos = (np.nanmin(wave[order, mask_i])-4.5, np.nanquantile(flux[order, :len(flux[order]//6)], 0.95))
        # add white box around text
        ax[i].text(*text_pos, target.replace('gl','Gl'), color='k', fontsize=12, weight='bold', transform=ax[i].transData)
        
    
    return ret

spirou_sample = {'880': [(3720, 4.72, 0.21, 6.868), '17'],
                 '15A': [(3603, 4.86, -0.30, 3.563), None],
                # '411': (3563, 4.84, 0.12), # TODO: double check this target
                '832': [(3590, 4.70, 0.06, 4.670),None],  # Tilipman+2021
                '752A': [(3558, 4.76, 0.10, 3.522),None], # Cristofari+2022
                '849':  [(3530, 4.78, 0.37, 8.803),None], # Cristofari+2022
                '725A': [(3441, 4.87, -0.23, 3.522),None],# Cristofari+2022
                '687': [(3413, 4.80, 0.10, 4.550),None], # Cristofari+2022
                '876' : [(3366, 4.80, 0.10, 4.672),None], # Moutou+2023, no measurement for logg, Z

                '725B': [(3345, 4.96, -0.30, 3.523),None],
                '699': [(3228.0, 5.09, -0.40, 1.827),None],
                '15B': [(3218, 5.07, -0.30, 3.561),None],
                '1151': [(3178, 4.71, -0.04, 8.043),None], # Lehmann+2024, I call it `gl` but it's `gj`
                '905': [(2930, 5.04, 0.23, 3.155),None],
}
targets = ['gl'+t for t in spirou_sample.keys()]
temperature_dict = {t: spirou_sample[t[2:]][0][0] for t in targets}
# norm = plt.Normalize(min(temperature_dict.values()), max(temperature_dict.values()))
norm = plt.Normalize(min(temperature_dict.values()), 4000.0)
cmap = plt.cm.plasma

def plot(orders):
    fig, ax = plt.subplots(1,1, figsize=(14,8), tight_layout=True)

    # orders = [0]
    orders_str = [str(o) for o in orders]
    # colors = plt.cm.
   
    for t, target in enumerate(targets):
        temperature = temperature_dict[target]
        color = cmap(norm(temperature))
        
        ret = main(target, ax=ax, offset=0.42*(len(targets)-t), orders=orders,
                run=spirou_sample[target[2:]][1], lw=1.0, color=color)

    # ax.legend(ncols=len(targets), loc='upper right')
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Only needed for color bar
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.03, aspect=20, location='right')
    cbar.set_label('Temperature (K)')

    xlim = ax.get_xlim()
    ax.set_xlim((xlim[0]-3, xlim[1]-3))

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Flux + offset')
    fig_name = base_path + 'paper/figures/best_fit_model' + "-".join(orders_str) + ".pdf"
    fig.savefig(fig_name)
    print(f'Figure saved as {fig_name}')

    show = False
    if show:
        plt.show()
    else:
        plt.close(fig)
    
for order in range(3):
    plot([order])