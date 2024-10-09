from retrieval_base.retrieval import Retrieval
import retrieval_base.figures as figs
from retrieval_base.config import Config
from retrieval_base.auxiliary_functions import spirou_sample
# import config_freechem as conf
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
plt.style.use('/home/dario/phd/retrieval_base/HBDs/my_science.mplstyle')

base_path = '/home/dario/phd/retrieval_base/'

def main(target, ax, orders=[0], offset=0.0, run=None, text_x=None, **kwargs):

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
    
    m = np.squeeze(ret.LogLike['spirou'].m) #+ offset
    m_flux_flat = ret.m_spec['spirou'].flux[0,:,0,:]
    spline_cont = m_flux_flat / m
    print(f' m.shape = {m.shape}')
    print(f' spline_cont.shape = {spline_cont.shape}')
    print(f' flux.shape = {flux.shape}')
    
    divide_spline = kwargs.get('divide_spline', False)
    if divide_spline:
        m /= spline_cont
        flux *= spline_cont
        
    m += offset
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
        
        
        ax.plot(wave[order], flux[order], color='k',lw=lw)
        ax.fill_between(wave[order], flux[order]-err_i, flux[order]+err_i, alpha=0.2, color='k', lw=0)
        ax.plot(wave[order], m[order], label=target,lw=lw, color=color)
        
        # add text above spectra in units of data
        if order == 1:
            text_pos = [np.nanmin(wave[order, mask_i])-4.5, np.nanquantile(flux[order, :len(flux[order]//6)], 0.90)]
            if text_x is not None:
                text_pos[0] = text_x
            # add white box around text
            ax.text(*text_pos, target.replace('gl','Gl'), color='k', fontsize=12, weight='bold', transform=ax.transData)
        
    
    return ret

targets = ['gl'+t for t in spirou_sample.keys()]
temperature_dict = {t: spirou_sample[t[2:]][0][0] for t in targets}
# norm = plt.Normalize(min(temperature_dict.values()), max(temperature_dict.values()))
norm = plt.Normalize(min(temperature_dict.values()), 4000.0)
cmap = plt.cm.plasma

def plot(text_x=None):
    fig, ax = plt.subplots(1,1, figsize=(12,6), tight_layout=True)

    # orders = [0]
    orders_str = [str(o) for o in orders]
    # colors = plt.cm.
   
    for t, target in enumerate(targets):
        temperature = temperature_dict[target]
        color = cmap(norm(temperature))
        
        ret = main(target, ax=ax, offset=0.46*(len(targets)-1-t), orders=orders,
                run=spirou_sample[target[2:]][1], lw=0.4, color=color,
                text_x=text_x, divide_spline=False)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Only needed for color bar
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.03, aspect=20, location='right')
    cbar.set_label('Temperature (K)')

    xlim = ax.get_xlim()
    ax.set_xlim((xlim[0]-12, xlim[1]-6))

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Flux + offset')
    fig_name = base_path + 'paper/latex/figures/best_fit_model' + "-".join(orders_str) + ".pdf"
    fig.savefig(fig_name)
    print(f'Figure saved as {fig_name}')

    show = False
    if show:
        plt.show()
    else:
        plt.close(fig)
    
text_x = 2273.5
orders = [0,1,2]
plot(text_x=text_x)