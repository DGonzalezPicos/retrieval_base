from retrieval_base.retrieval import Retrieval
import retrieval_base.figures as figs
from retrieval_base.config import Config
from retrieval_base.auxiliary_functions import spirou_sample, read_spirou_sample_csv
# import config_freechem as conf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import pathlib
# plt.style.use('/home/dario/phd/retrieval_base/HBDs/my_science.mplstyle')
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

# change font to sans-serif
# plt.rcParams['font.family'] = 'sans-serif'
# patheffects
import matplotlib.patheffects as path_effects

base_path = '/home/dario/phd/retrieval_base/'
nat_path = '/home/dario/phd/nat/figures/'
def main(target, ax, order=0, offset=0.0, run=None, text_x=None, offset_x=0.0, **kwargs):
    
    
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
    ccf_path = pathlib.Path(conf.prefix + 'plots/CCF/')

    bestfit_spec_file = test_output / 'bestfit_spec.npy'
    bestfit_spec_file_no13CO = test_output / 'bestfit_spec_no13CO.npy'
    bestfit_spec_file_noC18O = test_output / 'bestfit_spec_noC18O.npy'
    
    cache = kwargs.get('cache', True)
    if all([bestfit_spec_file.exists(), bestfit_spec_file_no13CO.exists(), bestfit_spec_file_noC18O.exists()]) and cache:
        
        print(f' Bestfit model found in {bestfit_spec_file}')
        wave, flux, err, mask, m, spline_cont = np.load(bestfit_spec_file)
        print(f' Bestfit model loaded from {bestfit_spec_file}')
        mask = mask.astype(bool)
        
        _, _, _, _, m_no13CO, spline_cont_no13CO = np.load(bestfit_spec_file_no13CO)    
        print(f' Bestfit model loaded from {bestfit_spec_file_no13CO}')
        _, _, _, _, m_noC18O, spline_cont_noC18O = np.load(bestfit_spec_file_noC18O)  
        
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
        
        
        # generate model without 13CO
        bestfit_params_no13CO = bestfit_params.copy()
        bestfit_params_no13CO[list(ret.Param.param_keys).index('log_12CO/13CO')] = 5.0
        
        ret.evaluate_model(bestfit_params_no13CO)
        ret.PMN_lnL_func()
        m_no13CO = np.squeeze(ret.LogLike['spirou'].m)
        m_flux_flat = ret.m_spec['spirou'].flux[0,:,0,:]
        spline_cont_no13CO = m_no13CO / m_flux_flat
        np.save(bestfit_spec_file_no13CO, np.array([wave, flux, err, mask, m_no13CO, spline_cont_no13CO]))
        print(f'Bestfit model saved as {bestfit_spec_file_no13CO}')
        
        # generate model without C18O
        bestfit_params_noC18O = bestfit_params.copy()
        # print(f'bestfit_params_noC18O (before)= {bestfit_params_noC18O}')
        bestfit_params_noC18O[list(ret.Param.param_keys).index('log_12CO/C18O')] = 5.0
        # print(f'bestfit_params_noC18O (after)= {bestfit_params_noC18O}')
        
        ret.evaluate_model(bestfit_params_noC18O)
        ret.PMN_lnL_func()
        m_noC18O = np.squeeze(ret.LogLike['spirou'].m)
        m_flux_flat = ret.m_spec['spirou'].flux[0,:,0,:]
        spline_cont_noC18O = m_noC18O / m_flux_flat
        np.save(bestfit_spec_file_noC18O, np.array([wave, flux, err, mask, m_noC18O, spline_cont_noC18O]))
        print(f'Bestfit model saved as {bestfit_spec_file_noC18O}')
        
        
        
        

    divide_spline = kwargs.get('divide_spline', False)
    if divide_spline:
        m /= spline_cont
        flux /= spline_cont
        
        m_no13CO /= spline_cont_no13CO
        m_noC18O /= spline_cont_noC18O
            
    m += offset
    flux += offset
    m_no13CO += offset
    m_noC18O += offset
    # residuals, save as npy with wave, residuals, err
    # np.save(ret.conf_output + 'residuals.npy', np.array([wave, flux-m, ret.Cov['spirou'][0].err * s[0]]))
    
    
    lw = kwargs.get('lw', 1.0)
    color = kwargs.get('color', 'orange')
    # for i, order in enumerate(orders):

    
    residuals_i = flux[order] - m[order]
    residuals_i_no13CO = flux[order] - m_no13CO[order]
    residuals_i_noC18O = flux[order] - m_noC18O[order]
    file_name = test_output / f'residuals_{order}.npy'
    np.save(file_name, np.array([wave[order], residuals_i, err[order]]))
    print(f'Residuals saved as {file_name}')
    
    
    ax[0].plot(wave[order], flux[order], color='k',lw=lw)
    ax[0].fill_between(wave[order], flux[order]-err[order], flux[order]+err[order], alpha=0.2, color='k', lw=0)
    ax[0].plot(wave[order], m[order], label=target,lw=lw, color=colors[0])
    # fill between m and m_no13CO
    # add path_effects to line with white edge
    pe = [path_effects.withStroke(linewidth=2, foreground='w')]
    ax[0].plot(wave[order], m_no13CO[order], lw=lw/1.5, color=colors[1], ls='-', zorder=-1)
    ax[0].fill_between(wave[order], m[order], m_no13CO[order], alpha=0.7, color=colors[1], lw=0)
    # ax[0].plot(wave[order], m_no13CO[order], label=target,lw=lw, color=colors[1], zorder=-1)
    
    ax[1].plot(wave[order], residuals_i, color=color, lw=lw, alpha=0.8)
    ax[1].plot(wave[order], residuals_i_no13CO, color=colors[1], lw=lw, alpha=0.8, zorder=-1)
    ax[1].fill_between(wave[order], residuals_i, residuals_i_no13CO, alpha=0.4, color=colors[1], lw=0, zorder=-2)
    
    ax[0].plot(wave[order], m_noC18O[order], lw=lw/1.5, color=colors[2], ls='-', zorder=-1)
    ax[0].fill_between(wave[order], m[order], m_noC18O[order], alpha=0.8, color=colors[2], lw=0, zorder=-2)
    ax[1].plot(wave[order], residuals_i_noC18O, color=colors[2], lw=lw, alpha=0.8, zorder=-2)
    ax[1].fill_between(wave[order], residuals_i, residuals_i_noC18O, alpha=0.4, color=colors[2], lw=0, zorder=-3)
    
    # add text above spectra in units of data
    show_name = kwargs.get('show_name', False)
    if show_name:
        text_pos = [np.nanmin(wave[order, mask[order]]), np.nanquantile(flux[order, :len(flux[order]//2)], 0.90)-0.15]
        if text_x is not None:
            text_pos[0] = text_x[0]
        if kwargs.get('text_y', None) is not None:
            text_pos[1] = kwargs.get('text_y')
        # add white box around text
        
        pe = [path_effects.withStroke(linewidth=2, foreground='w')]
        s = target.replace('gl','')
        ax[0].text(*text_pos, s, color='k', fontsize=9, weight='bold', transform=ax[0].transData,
                    path_effects=pe)
    
    # sigmas = [1,2,3]
    # mean_err = np.nanmean(err[order])
    # alpha = [1.0, 0.7, 0.4]
    # for s, sigma in enumerate(sigmas):
    #     ax[1].errorbar(2286.4-offset_x, 0, yerr=mean_err*sigma, color=color, zorder=1, lw=1.5, capsize=0, capthick=1.5, alpha=alpha[s])
        
        
    
    return ccf_path


df = read_spirou_sample_csv()
# flip order of all columns
flip_rows = True
if flip_rows:
    df = df.iloc[::-1]

names = df['Star'].to_list()
teff =  dict(zip(names, [float(t.split('+-')[0]) for t in df['Teff (K)'].to_list()]))
spt = dict(zip(names, [t.split('+-')[0] for t in df['SpT'].to_list()]))
runs = dict(zip(spirou_sample.keys(), [spirou_sample[k][1] for k in spirou_sample.keys()]))

norm = plt.Normalize(2800, 4000.0)
cmap = plt.cm.coolwarm_r

my_targets_id = [
                # '338B', 
                 '205',
                #  '411', 
                #  '436',
                #  '699',
                #  '1286',
                 ]
my_targets = ['gl'+t for t in my_targets_id]
def plot(orders, name, text_x=None, xlim=None, **kwargs):
    
    fig = plt.figure(figsize=(5, 3.5))  # Increase the height to accommodate spacing
    gs = gridspec.GridSpec(11, 10, wspace=0.2, hspace=0.05, 
                            top=0.97, bottom=0.12, left=0.12, right=0.89)  # Set spacing
    fig.text(0.36, 0.03, r'Wavelength [nm]', ha='center', va='center', fontsize=8)
    fig.text(0.03, 0.5, r'Normalized flux', ha='center', va='center', rotation='vertical', fontsize=8)
    
    ax_label_y = 0.01
    ax_labels = dict(a=(0.126, 0.82), b=(0.126, 0.74), 
                     c=(0.126, 0.51), d=(0.126, 0.425),
                     e=(0.126, 0.202), f=(0.126, 0.115),
                     g=(0.60, 0.61), h=(0.60, 0.225),
                     )
    for k, v in ax_labels.items():
        fig.text(v[0], v[1] + ax_label_y, k, fontsize=9, fontweight='bold', zorder=0, alpha=0.90)

    maxes = []
    raxes = []
    saxes = []  # Axes for spacing, remove later if not needed

    for i in range(3):
        # Define starting and ending rows for each main/residual plot pair
        i_start = 4 * i
        i_end = i_start + 2
        print(i_start, i_end)
        # Main axes (larger, for spectra)
        maxi = plt.subplot(gs[i_start:i_end, :6])
        # remove x-axis labels for all but the bottom plot
        plt.setp(maxi.get_xticklabels(), visible=False)
        maxes.append(maxi)
        
        # Residual axes (smaller, for residuals) directly below the main axes
        raxes.append(plt.subplot(gs[i_end:i_end + 1, :6], sharex=maxes[-1]))
        
        # Spacing axes (empty) between each maxes/raxes pair
        if i < 2:  # Avoid adding a spacer after the last pair
            saxes.append(plt.subplot(gs[i_end + 1:i_end + 2, :6]))
            saxes[-1].axis('off')  # Hide the spacing axes

    # Custom axis on the right side
    ccf_ax = plt.subplot(gs[4:9, 6:])
    plt.setp(ccf_ax.get_xticklabels(), visible=False)
    ccf_res_ax = plt.subplot(gs[9:11, 6:], sharex=ccf_ax)
    # move ylabels and ticks to the right side
    ccf_ax.yaxis.tick_right()
    ccf_ax.yaxis.set_label_position("right")
    ccf_res_ax.yaxis.tick_right()
    ccf_res_ax.yaxis.set_label_position("right")
    
    fig.tight_layout()

    # orders = [0]
    orders_str = [str(o) for o in orders]
    # colors = plt.cm.
    count = 1    
    # for t, name in enumerate(names):
    target = name.replace('Gl ', 'gl')
    
    temperature = teff[name]
    color = cmap(norm(temperature))
    
    # offset = 0.42*(len(names)-t)
    offset = 0.54*(len(my_targets)-my_targets.index(target)-1)
    
    for order in orders:
        ax_s, ax_r = maxes[order], raxes[order]
        ccf_path = main(target, ax=[maxes[order], raxes[order]], offset=offset, order=order,
                run=None, 
                lw=0.4, 
                color=color,
                text_x=text_x[order] if text_x is not None else None,
                divide_spline=True,
                offset_x=-0.7*count,
                show_name=(order==0),
                **kwargs)
        
        
        
        ax_s.set_xlim(xlim[order])
        ax_s.set_ylim(0.55, 1.14)
        
        ax_r.axhline(0, color='k', lw=0.5)
        ax_r.set_ylim(-0.10, 0.05)
        
    species_list = ['13CO', 'C18O']
    lw_ccf = 0.6
    for s, species in enumerate(species_list):
        ccf_file = ccf_path / f'RV_CCF_ACF_{species}.txt'
        rv, CCF_SNR, ACF_SNR = np.loadtxt(ccf_file).T
        print(f' Loaded {ccf_file}')
        ccf_ax.plot(rv, CCF_SNR, color=colors[s+1], lw=lw_ccf)
        ccf_ax.fill_between(rv, CCF_SNR, alpha=0.1, color=colors[s+1])
        ccf_ax.plot(rv, ACF_SNR, color=colors[s+1], ls='--', alpha=0.9,
                    lw=lw_ccf)
        ccf_res_ax.plot(rv, CCF_SNR - ACF_SNR, color=colors[s+1],
                        lw=lw_ccf)
    ccf_res_ax.axhline(0, color='k', lw=0.5)
    ccf_res_ax.set(xlabel=r'RV [km s$^{-1}$]', ylabel='CCF - ACF')
    ccf_ax.set_xlim(-150, 150)
    ccf_ax.set_title('Cross-correlation function', fontsize=8)
    ccf_ax.set_ylabel('SNR')
    ccf_ax.set_ylim(-9.99, None)
    
    ccf_res_yticks = [-4.0, 0.0, 4.0]
    ccf_res_ax.set_yticks(ccf_res_yticks)
    ccf_res_ax.set_yticklabels([f'{t:.0f}' for t in ccf_res_yticks])
    
    
    # create custom legend with handles and labels
    from matplotlib.lines import Line2D
    handles = []
    labels = []
    handles.append(Line2D([0], [0], color='k', lw=2.0, alpha=0.85, ls='-'))
    labels.append('Observation')
    handles.append(Line2D([0], [0], color=colors[0], lw=2.0, alpha=0.85))
    labels.append('Best-fit model')
    for s, species in enumerate(species_list):
        handles.append(Line2D([0], [0], color=colors[s+1], lw=2.0, alpha=0.85))
        labels.append(species_labels[species])
    ccf_ax.legend(handles, labels, loc=(0.05, 1.25), frameon=False)

    
    
    fig_name = nat_path + 'best_fit_model' + "-".join(orders_str) + "_no13CO.pdf"
    fig.savefig(fig_name)
    print(f'Figure saved as {fig_name}')

    show = False
    if show:
        plt.show()
    else:
        plt.close(fig)
  
order = 0
orders = [0,1,2]
cenwaves = [2347.4, 2376.5, 2467.5]
dwave = 3.02
xlim_list = [(cenwave-dwave, cenwave+dwave) for cenwave in cenwaves]
# xlim = (2343, 2358.0)

# xlim = xlim_list[order]
# text_x = (xlim[0]+0.1, xlim[1]-2)
text_x = [(xlim[0]+0.3, xlim[1]-2) for xlim in xlim_list]
species_labels = {
    '13CO': r'$^{13}$CO',
    'C18O': r'C$^{18}$O',
}
name = 'Gl 205'
temperature = teff[name]
colors = [cmap(norm(temperature)), 'orange', 'seagreen']
plot(orders, name=name, text_x=text_x, xlim=xlim_list, text_y=1.07)