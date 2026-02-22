""" 2024-12-11 Fig. 1 of nat paper showing best fit spectra of one target of each SpT """
from retrieval_base.retrieval import Retrieval
import retrieval_base.figures as figs
from retrieval_base.config import Config
from retrieval_base.auxiliary_functions import spirou_sample, read_spirou_sample_csv
# import config_freechem as conf
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
# plt.style.use('/home/dario/phd/retrieval_base/HBDs/my_science.mplstyle')
import scienceplots

import matplotlib.patches as patches

import matplotlib.patheffects as path_effects

base_path = '/home/dario/phd/retrieval_base/'
nat_path = '/home/dario/phd/nat/figures/'

def main(target, ax, order=0, offset=0.0, run=None, text_x=None, offset_y=0.0, offset_x=0.0, **kwargs):
    
    
    assert len(ax) == 2, f'Lenght of ax must be 2, not {len(ax)}'
    # assert len(ax) ==3 if ax is not None else True, f'Lenght of ax must be 3, not {len(ax)}'
    # ax = np.atleast_1d(ax)
    # assert len(ax) == len(orders), f'Lenght of ax must be {len(orders)}, not {len(ax)}'
    # if len(ax) < len(orders):
    #     ax = np.append(ax, ax[-1])
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
    bestfit_spec_file = test_output / 'bestfit_spec.npy'
    
    cache = kwargs.get('cache', True)
    if bestfit_spec_file.exists() and cache:
        
        print(f' Bestfit model found in {bestfit_spec_file}')
        wave, flux, err, mask, m, spline_cont = np.load(bestfit_spec_file)
        print(f' Bestfit model loaded from {bestfit_spec_file}')
        mask = mask.astype(bool)
        
    else:
        

        ret = Retrieval(
                    conf=conf, 
                    evaluation=False,
                    )

        bestfit_params, posterior = ret.PMN_analyze()
        ret.evaluate_model(bestfit_params)
        ret.PMN_lnL_func()
        
        rv = bestfit_params[list(ret.Param.param_keys).index('rv')]
        
        wave = np.squeeze(ret.d_spec['spirou'].wave) * (1 - rv/299792.458)
        flux = np.squeeze(ret.d_spec['spirou'].flux)
        
        s = ret.LogLike['spirou'].s
        
        # err  = [ret.Cov['spirou'][i][0].err * s[i] for i in range(3)]
        
        m = np.squeeze(ret.LogLike['spirou'].m) #+ offset
        m_flux_flat = ret.m_spec['spirou'].flux[0,:,0,:]
        spline_cont = m / m_flux_flat
        print(f' m.shape = {m.shape}')
        print(f' spline_cont.shape = {spline_cont.shape}')
        print(f' flux.shape = {flux.shape}')
        
        err = np.ones_like(wave) * np.nan
        mask = ret.d_spec['spirou'].mask_isfinite[:,0]
        for i in range(3):
            # err_i = np.ones_like(wave[order]) * np.nan
        
            err[order, mask[order]] = ret.Cov['spirou'][order][0].err * s[order]
        
        # save file
        np.save(bestfit_spec_file, np.array([wave, flux, err, mask, m, spline_cont]))
        print(f'Bestfit model saved as {bestfit_spec_file}')

    divide_spline = kwargs.get('divide_spline', False)
    if divide_spline:
        m /= spline_cont
        flux /= spline_cont
            
    m += offset
    flux += offset
    # residuals, save as npy with wave, residuals, err
    # np.save(ret.conf_output + 'residuals.npy', np.array([wave, flux-m, ret.Cov['spirou'][0].err * s[0]]))
    
    
    lw = kwargs.pop('lw', 0.4)
    color = kwargs.pop('color', 'orange')
    dark_theme = kwargs.get('dark_theme', False)
    black = 'k' if not dark_theme else 'w'
    # for i, order in enumerate(orders):

    
    residuals_i = flux[order] - m[order]
    file_name = test_output / f'residuals_{order}.npy'
    np.save(file_name, np.array([wave[order], residuals_i, err[order]]))
    print(f'Residuals saved as {file_name}')
    
    
    ax[0].plot(wave[order], flux[order], color=black,lw=lw)
    ax[0].fill_between(wave[order], flux[order]-err[order], flux[order]+err[order], alpha=0.2, color=black, lw=0)
    ax[0].plot(wave[order], m[order], label=target,lw=lw, color=color)
    
    ax[1].plot(wave[order], residuals_i, color=color, lw=lw, alpha=0.8)
    # ax[i].plot(wave[order],spline_cont[order]+offset, color='red', lw=lw)
    
    # add text above spectra in units of data

    text_pos = [np.nanmin(wave[order, mask[order]]), np.nanquantile(flux[order, :len(flux[order]//2)], 0.90)-0.15]
    if text_x is not None:
        text_pos[0] = text_x[0]
    
    text_pos[1] += offset_y
    # add white box around text
    if kwargs.get('include_path_effects', True):
        pe = [path_effects.withStroke(linewidth=2, foreground='w')]
    else:
        pe = None
    s = target.replace('gl','')
    ax[0].text(*text_pos, s, color='k', fontsize=9, weight='bold', transform=ax[0].transData,
                path_effects=pe)
    
    # show the mean error in the residual plot as errorbar
    # err_q = [0.50, 0.90, 0.95]
    # 
    # for q, err_i in enumerate(np.nanquantile(residuals_i, err_q)):
    #     ax[1].errorbar(2290-offset_x, 0, yerr=err_i, color=color, zorder=1, alpha=alpha[q])
    
    sigmas = [1,2,3]
    mean_err = np.nanmean(err[order])
    alpha = [1.0, 0.7, 0.4]
    for s, sigma in enumerate(sigmas):
        ax[1].errorbar(2286.4-offset_x, 0, yerr=mean_err*sigma, color=color, zorder=1, lw=1.5, capsize=0, capthick=1.5, alpha=alpha[s])
        
        
    
    return None


def plot(order, names, my_targets, 
         teff, spt,
         cmap, norm,
         text_x=None, 
         text_y_offset=0.0, 
         xlim=None, 
         axes=None, 
         add_cbar=True,
         show_lines=False,
         include_path_effects=True, **kwargs):
    
    fig = None
    if axes is None:
        fig, ax = plt.subplots(2,1, figsize=(3.35,3.35/2), sharex=False, gridspec_kw={'height_ratios': [4, 1],
                                                                            'hspace': 0.08,
                                                                            'top': 0.97,
                                                                            'bottom': 0.13,
                                                                            'left': 0.10,
                                                                            'right': 0.99})
    else:
        assert len(axes) == 2, f'Axes must have length 2, not {len(axes)}'
        ax = axes
    
    dark_theme = kwargs.get('dark_theme', False)
    # orders = [0]
    orders_str = [str(order)]
    # colors = plt.cm.
    count = 0
    for name in names:
        target = name.replace('Gl ', 'gl')
        if target not in my_targets:
            continue
        count += 1
        temperature = teff[name]
        color = cmap(norm(temperature))
        
        # offset = 0.42*(len(names)-t)
        offset = 0.54*(len(my_targets)-my_targets.index(target)-1)
        ret = main(target, ax=ax, offset=offset, order=order,
                run=None, 
                color=color,
                text_x=text_x, 
                offset_y=text_y_offset,
                divide_spline=True,
                offset_x=-0.7*count,
                include_path_effects=include_path_effects,
                **kwargs)
        
        if include_path_effects:
            pe = [path_effects.withStroke(linewidth=2, foreground='w')]
        else:
            print(f' No path effects for {name}')
            pe = None
        
        ax[0].text(s=spt[name].split('.')[0].replace('V',''), x=text_x[1]-3, y=1.0+offset, transform=ax[0].transData,
                    color=color, fontsize=7, weight='bold',                   
                    path_effects=pe)
        
    black = 'k' if not dark_theme else 'w'
    ax[-1].axhline(0.0, color=black, lw=0.5, zorder=-1)
    lw = kwargs.get('lw', 0.7)
    # print(f' lw = {lw}')
    if show_lines:
        # load from npy file
        lines = {}
        lines['13CO'] = np.load("/home/dario/phd/retrieval_base/gl205/retrieval_outputs/fc5/test_output/13CO_lines.npy")[0]
        lines['C18O'] = np.load("/home/dario/phd/retrieval_base/gl205/retrieval_outputs/fc5/test_output/C18O_lines.npy")[0]
        print(f' lines[13CO].shape = {lines["13CO"].shape}')
        colors_lines = ['orange', 'green']
        for s, species in enumerate(lines.keys()):
            for line in lines[species]:
                print(f' line = {line}')
                for axx in ax:
                    axx.axvline(line, color=colors_lines[s], lw=lw*1.6,
                                ymin=0.95 if axx == ax[0] else 0.84, 
                                ymax=0.98 if axx == ax[0] else 0.92)
                    
        # create patches and labels for custom legend
        handles = [patches.Patch(color=colors_lines[0], label=r'$^{13}$' + 'CO'),
                   patches.Patch(color=colors_lines[1], label='C' + r'$^{18}$' + 'O')]
        ax[-1].legend(handles=handles, loc='upper left', bbox_to_anchor=(1.0, 1.02),
                      fontsize=lw*10,
                      frameon=False,
                      handlelength=0.3,
                      handletextpad=0.5)
        # change linelength and linewidth of legend
        # for handle in handles:
        #     handle.set_linewidth(1.5)
        #     handle.set_linestyle('--')
    
    if add_cbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Only needed for color bar
        # cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.01, aspect=40, location='right')
        cbar_ax = fig.add_axes([1.002, 0.325, 0.015, 0.648])
        cbar = plt.colorbar(sm, cax=cbar_ax, orientation='vertical', location='right')
        cbar.set_label('Temperature (K)')

        # set ticks of colorbar
        cbar_ticks = np.linspace(3000, 3900, 7)
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels([str(int(t)) for t in cbar_ticks])
    if xlim is not None:
        ax[0].set_xlim(xlim)
        ax[1].set_xlim(xlim)
    else:
        xlim = ax.get_xlim()
        ax[0].set_xlim((xlim[0]-5, xlim[1]-3))
        ax[1].set_xlim((xlim[0]-5, xlim[1]-3))

    ax[0].set_ylim(0.45, 4.0)
    ax[-1].set_xlabel('Wavelength (nm)')
    ylim_res = ax[1].get_ylim()
    # make symmetric
    ylim_res_sym = max(abs(ylim_res[0]), abs(ylim_res[1]))
    ax[1].set_ylim(-ylim_res_sym, ylim_res_sym)
    ax[0].set_ylabel('Flux + offset')
    ax[1].set_ylabel('Residuals', labelpad=0)
    # fig_name = base_path + 'paper/latex/figures/best_fit_model' + "-".join(orders_str) + ".pdf"
    if fig is not None:
        fig_name = nat_path + 'best_fit_model' + "-".join(orders_str) + ".pdf"
        fig.savefig(fig_name, bbox_inches='tight')
        # save as png too
        fig_name = nat_path + 'png/best_fit_model' + "-".join(orders_str) + ".png"
        fig.savefig(fig_name, dpi=300, bbox_inches='tight')
        print(f'Figure saved as {fig_name}')    
        plt.close(fig)
        
    return fig, ax


if __name__ == '__main__':
    # reset to default
    plt.style.use('default')
    # plt.style.use(['latex-sans'])
    plt.style.use(['sans'])
    # enable latex
    # plt.rcParams['text.usetex'] = True
    plt.rcParams.update({
        "font.size": 8,
    })
    
    df = read_spirou_sample_csv()
    # flip order of all columns
    flip_rows = True
    if flip_rows:
        df = df.iloc[::-1]

    names = df['Star'].to_list()
    teff =  dict(zip(names, [float(t.split('+-')[0]) for t in df['Teff (K)'].to_list()]))
    spt = dict(zip(names, [t.split('+-')[0] for t in df['SpT'].to_list()]))
    # prot = dict(zip(names, [float(t.split('+-')[0]) for t in df['Period (days)'].to_list()]))
    # prot_err = dict(zip(names, [float(t.split('+-')[1]) for t in df['Period (days)'].to_list()]))
    runs = dict(zip(spirou_sample.keys(), [spirou_sample[k][1] for k in spirou_sample.keys()]))


    norm = plt.Normalize(3000, 3900.0)
    cmap = plt.cm.coolwarm_r

    my_targets_id = ['338B', '205', '411', '436','699', '1286']
    my_targets = ['gl'+t for t in my_targets_id]

    order = 0
    # xlim = (2282, 2364)
    xlim = (2341, 2364)
    text_x = (xlim[0]+0.4, xlim[1]+1.2)
    plot(order, names, my_targets, 
         teff=teff, spt=spt,
         cmap=cmap, norm=norm,
         text_x=text_x, 
         xlim=xlim,
         add_cbar=True,
         show_lines=True)
