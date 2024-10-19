from retrieval_base.retrieval import Retrieval
import retrieval_base.figures as figs
from retrieval_base.config import Config
from retrieval_base.auxiliary_functions import spirou_sample, read_spirou_sample_csv
# import config_freechem as conf
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
plt.style.use('/home/dario/phd/retrieval_base/HBDs/my_science.mplstyle')

base_path = '/home/dario/phd/retrieval_base/'

def main(target, ax, orders=[0], offset=0.0, run=None, text_x=None, **kwargs):
    
    # assert len(ax) ==3 if ax is not None else True, f'Lenght of ax must be 3, not {len(ax)}'
    ax = np.atleast_1d(ax)
    assert len(ax) == len(orders), f'Lenght of ax must be {len(orders)}, not {len(ax)}'

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
        
        
        ax[i].plot(wave[order], flux[order], color='k',lw=lw)
        ax[i].fill_between(wave[order], flux[order]-err_i, flux[order]+err_i, alpha=0.2, color='k', lw=0)
        ax[i].plot(wave[order], m[order], label=target,lw=lw, color=color)
        
        # add text above spectra in units of data

        text_pos = [np.nanmin(wave[order, mask_i])-4.5, np.nanquantile(flux[order, :len(flux[order]//6)], 0.95)]
        if text_x is not None:
            text_pos[0] = text_x[0]
        # add white box around text
        ax[i].text(*text_pos, target.replace('gl','Gl'), color='k', fontsize=12, weight='bold', transform=ax[i].transData)
        
    
    return ret


df = read_spirou_sample_csv()
names = df['Star'].to_list()
teff =  dict(zip(names, [float(t.split('+-')[0]) for t in df['Teff (K)'].to_list()]))
# prot = dict(zip(names, [float(t.split('+-')[0]) for t in df['Period (days)'].to_list()]))
# prot_err = dict(zip(names, [float(t.split('+-')[1]) for t in df['Period (days)'].to_list()]))
runs = dict(zip(spirou_sample.keys(), [spirou_sample[k][1] for k in spirou_sample.keys()]))

# norm = plt.Normalize(min(temperature_dict.values()), max(temperature_dict.values()))
norm = plt.Normalize(min(teff.values()), 4000.0)
cmap = plt.cm.plasma

def plot(orders, text_x=None):
    fig, ax = plt.subplots(1,1, figsize=(12,6), tight_layout=True)

    # orders = [0]
    orders_str = [str(o) for o in orders]
    # colors = plt.cm.
   
    for t, name in enumerate(names):
        target = name.replace('Gl ', 'gl')

        temperature = teff[name]
        color = cmap(norm(temperature))
        
        ret = main(target, ax=ax, offset=0.42*(len(names)-t), orders=orders,
                run=None, 
                lw=1.0, color=color,
                text_x=text_x, divide_spline=False)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Only needed for color bar
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.03, aspect=20, location='right')
    cbar.set_label('Temperature (K)')

    xlim = ax.get_xlim()
    ax.set_xlim((xlim[0]-5, xlim[1]-3))

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
    
text_x = [(2285.5, 2364.),
          (2358.0, 2438.),
          (2435.0, 2510.0),
]
  
for order in range(3):
    plot([order], text_x=text_x[order])