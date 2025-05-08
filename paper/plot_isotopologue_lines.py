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
import scipy.signal as signal
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
    
    model_13CO = m - m_no13CO
    model_C18O = m - m_noC18O
    
    lines_13CO = find_lines(wave[order], model_13CO[order], threshold=0.0001)
    lines_C18O = find_lines(wave[order], model_C18O[order], threshold=0.0001)
    
    # save as npy file called: CO_lines.npy with columns, wave_13CO, wave_C18O
    np.save(test_output / '13CO_lines.npy', np.array([lines_13CO]))
    np.save(test_output / 'C18O_lines.npy', np.array([lines_C18O]))
    print(f'{test_output}/13CO_lines.npy saved')
    print(f'{test_output}/C18O_lines.npy saved')
    
    assert len(lines_13CO) > 0, f'No lines found in 13CO model'
    assert len(lines_C18O) > 0, f'No lines found in C18O model'
    # print(f'lines_13CO = {lines_13CO}')
    # print(f'lines_C18O = {lines_C18O}')
    
    if divide_spline:
        m /= spline_cont
        flux /= spline_cont
        
        m_no13CO /= spline_cont_no13CO
        m_noC18O /= spline_cont_noC18O
        
            
    m += offset
    flux += offset
    m_no13CO += offset
    m_noC18O += offset
    
    residuals_i = flux[order] - m[order]
    residuals_i_no13CO = flux[order] - m_no13CO[order]
    residuals_i_noC18O = flux[order] - m_noC18O[order]
    
    
    # ax[0].plot(wave[order], m[order], color='k', lw=0.5)
    ax[0].plot(wave[order], residuals_i, color='k', lw=0.5)
    ax[0].plot(wave[order], residuals_i_no13CO, color='r', lw=0.5)
    ax[0].plot(wave[order], residuals_i_noC18O, color='b', lw=0.5)
    
    ax[1].plot(wave[order], model_13CO[order], color='r', lw=0.5)
    ax[1].plot(wave[order], model_C18O[order], color='b', lw=0.5)
    
    for line in lines_13CO:
        ax[1].axvline(line, color='r', lw=1, ymin=0.9, ymax=0.98)
    for line in lines_C18O:
        ax[1].axvline(line, color='b', lw=1, ymin=0.9, ymax=0.98)
    
    return m, flux, m_no13CO, m_noC18O

def find_lines(x, y, distance=10, threshold=0.05):
    """ identify absorption lines in the model and plot them,
    return wavelengths of the lines """
    # find lines using scipy.signal.find_peaks
    peaks, _ = signal.find_peaks(-y, distance=distance, threshold=threshold)
    return x[peaks]

fig, ax = plt.subplots(2, 1, figsize=(10, 8))

order = 0
m, flux, m_no13CO, m_noC18O = main('gl205', ax=ax, order=order, divide_spline=True)

plt.show()