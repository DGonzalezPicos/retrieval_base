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
    
    lw = kwargs.get('lw', 1.0)
    
    for i, order in enumerate(orders):
        err_i = np.ones_like(wave[order]) * np.nan
        
        mask_i = ret.d_spec['spirou'].mask_isfinite[order,0]
        err_i[mask_i] = ret.Cov['spirou'][order][0].err * s[order]
        
        ax[i].plot(wave[order], flux[order], color='k',lw=lw)
        ax[i].fill_between(wave[order], flux[order]-err_i, flux[order]+err_i, alpha=0.2, color='k', lw=0)
        ax[i].plot(wave[order], m[order], label=target,lw=lw)
    
    return ret

spirou_sample = {'880': [(3720, 4.72, 0.21), '17'],
                 '15A': [(3603, 4.86, -0.30), None],
                # '411': (3563, 4.84, 0.12), # TODO: double check this target
                # '752A': [(3558, 4.76, 0.10),None],
                '725A': [(3441, 4.87, -0.23),None],
                '725B': [(3345, 4.96, -0.30),None],
                '15B': [(3218, 5.07, -0.30),None],
                '905': [(2930, 5.04, 0.23),None],
}
targets = ['gl'+t for t in spirou_sample.keys()]

def plot(orders):
    fig, ax = plt.subplots(1,1, figsize=(14,8), tight_layout=True)

    # orders = [0]
    orders_str = [str(o) for o in orders]
    for t, target in enumerate(targets):
        ret = main(target, ax=ax, offset=0.42*(len(targets)-t), orders=orders,
                run=spirou_sample[target[2:]][1])

    ax.legend(ncols=len(targets), loc='upper right')

    xlim = ax.get_xlim()
    ax.set_xlim((xlim[0]+3, xlim[1]-3))

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