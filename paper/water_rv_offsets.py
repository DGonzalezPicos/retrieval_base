""" 2025-01-20 Measure RV offsets of residual H2O lines """
from retrieval_base.retrieval import Retrieval
import retrieval_base.figures as figs
from retrieval_base.config import Config
from retrieval_base.auxiliary_functions import spirou_sample, read_spirou_sample_csv
# import config_freechem as conf
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
from scipy.optimize import curve_fit

base_path = '/home/dario/phd/retrieval_base/'
nat_path = '/home/dario/phd/nat/figures/'

def main(target, ax, run=None, **kwargs):
    
    
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
        
    print(f' Bestfit model found in {bestfit_spec_file}')
    wave, flux, err, mask, m, spline_cont = np.load(bestfit_spec_file)
    print(f' Bestfit model loaded from {bestfit_spec_file}')
    mask = mask.astype(bool)
    
    divide_spline = kwargs.get('divide_spline', False)
    if divide_spline:
        m /= spline_cont
        flux /= spline_cont
        

    
    residuals_i = flux - m
    # file_name = test_output / f'residuals_{order}.npy'
    
    return wave, flux, err, mask, m, spline_cont, residuals_i

def gaussian(x, amp, mean, std, offset=0.0):
    return amp * np.exp(-(x - mean)**2 / (2 * std**2)) + offset
    
def double_gaussian(x, amp1, mean1, std1, amp2, mean2, std2):
    return (gaussian(x, amp1, mean1, std1) +
            gaussian(x, amp2, mean2, std2))
        
def fit_double_gaussian(wavelength, flux, residuals):
    # First fit line center of flux
    # line_center = np.nanmedian(wavelength)
    # line_width = 0.1
    flux = np.nan_to_num(flux, nan=0.0)
    residuals = np.nan_to_num(residuals, nan=0.0)
    try:
        popt_flux, pcov_flux = curve_fit(gaussian, wavelength, flux, p0=[-1, line_center, line_width, 1.0])
    except RuntimeError:
        print('Failed to fit flux gaussian')
        popt_flux = [1, np.nanmedian(wavelength), 0.1, 0.0]
    # line_center = popt_flux[1]
    # Initial guesses for the parameters
    p0 = [1, line_center, line_width, -1, line_center, line_width]
    
    try:
        # Fit the double Gaussian model to the data
        popt_residuals, pcov_residuals = curve_fit(double_gaussian, wavelength, residuals, p0=p0)
    except RuntimeError:
        print('Failed to fit residuals gaussian')
        popt_residuals = [1, np.nanmedian(wavelength), 0.1, -1, np.nanmedian(wavelength), 0.1]
    
    return popt_flux, popt_residuals

df = read_spirou_sample_csv()
# flip order of all columns
flip_rows = True
if flip_rows:
    df = df.iloc[::-1]

names = df['Star'].to_list()
targets = [name.replace('Gl ', 'gl') for name in names]

# fig, ax = plt.subplots(2,1, figsize=(5,4), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)



offset = 0.0
text_x = 0.0
kwargs = {'divide_spline': True}

# target = targets[0]
# target = 'gl4333'
target = 'gl1286'

    
def get_line_offset(target, order, line_center, line_width):
    
    fig = plt.figure(figsize=(12,4))
    lw = 1.0
    gs = fig.add_gridspec(9,9, hspace=0.2, wspace=0.4)
    ax_spec = fig.add_subplot(gs[:6, :6])
    ax_res = fig.add_subplot(gs[6:, :6])
    ax = [ax_spec, ax_res]

    ax_rv = fig.add_subplot(gs[:, 6:])
    wave, flux, err, mask, m, spline_cont, residuals_i = main(target, ax=ax, run=None, **kwargs)

    wave, flux, err, mask, m, spline_cont, residuals_i = wave[order], flux[order], err[order], mask[order], m[order], spline_cont[order], residuals_i[order]

    region = (wave > line_center - line_width) & (wave < line_center + line_width)


    # ax_rv.plot(rv[:, 1], rv[:, 0], color='black', lw=lw)
    ax[0].plot(wave[region], flux[region], color='black', lw=lw, label='Data')
    ax[0].plot(wave[region], m[region], color='darkorange', lw=lw, label='Model')
    ax[1].plot(wave[region], residuals_i[region], color='black', lw=lw, label='Residuals')
    ax[1].axhline(0, color='k', lw=lw)
    ax[1].fill_between(wave[region], residuals_i[region], color='black', alpha=0.2)



    popt_flux, popt_residuals = fit_double_gaussian(wave[region], flux[region], residuals_i[region])
    print(popt_flux)
    print(popt_residuals)

    x_double_gaussian = np.linspace(min(wave[region]), max(wave[region]), 1000)
    model_double_gaussian = double_gaussian(x_double_gaussian, *popt_residuals)
    ax[1].plot(x_double_gaussian, model_double_gaussian, color='magenta', lw=1.0, alpha=0.9)
    separation_double_gaussian = popt_residuals[1] - popt_residuals[4]
    half_separation_double_gaussian_rv = 0.5 * separation_double_gaussian * 299792.458 / line_center
    print(f'Half separation between the two gaussians: {half_separation_double_gaussian_rv:.2f} km/s')

    model_gaussian = gaussian(x_double_gaussian, *popt_flux)
    ax[0].plot(x_double_gaussian, model_gaussian, color='green', lw=0.8, label='Gaussian fit (data)')


    # fit gaussian to model spectrum
    try:
        popt_m, pcov_m = curve_fit(gaussian, wave[region], np.nan_to_num(m[region], nan=1.0), p0=[-1, line_center, line_width, 1.0])
    except RuntimeError:
        print('Failed to fit model gaussian')
        popt_m = [1, np.nanmedian(wave[region]), 0.1, 0.0]
    print(popt_m)
    m_gaussian = gaussian(x_double_gaussian, *popt_m)
    ax[0].plot(x_double_gaussian, m_gaussian, color='navy', lw=0.8, label='Gaussian fit (model)')
    line_center_m = popt_m[1]

    # measure shifts of residuals with respect to line center measured with gaussian fit
    line_center_gaussian = popt_flux[1]
    for axi in ax:
        axi.axvline(line_center_gaussian, color='green', lw=0.8, ls='--')
        axi.axvline(line_center_m, color='navy', lw=0.8, ls='--')
        
        
    line_offset_rv = -(line_center_gaussian - line_center_m) * 299792.458 / line_center_gaussian
    print(f'Line offset: {line_offset_rv:.2f} km/s')


    flux_g = gaussian(x_double_gaussian, *popt_flux)
    model_g = gaussian(x_double_gaussian, *popt_m)
    rv_region = (x_double_gaussian - line_center_gaussian) * 299792.458 / line_center_gaussian

    ax_rv.plot(rv_region, flux_g, color='green', lw=1.2)
    ax_rv.plot(rv_region, model_g, color='navy', lw=1.2)
    ax_rv.axvline(line_offset_rv, color='navy', lw=1.2, ls='--', label=f'RV = {line_offset_rv:.2f} km/s')
    ax_rv.legend()
    # add labels to all plots
    ax[0].set_ylabel('Flux')
    ax[0].set_xlabel('Wavelength (nm)')
    ax[0].legend()
    ax[1].set_ylabel('Residuals')
    ax[1].set_ylim(-0.20, 0.20)
    ax_rv.set_xlabel('RV (km/s)')

    ax[0].set_title(f'{target.replace("gl", "Gl ")}')

    folder = pathlib.Path(base_path) / 'paper' / f'water_line_offset_{line_center:.1f}'
    folder.mkdir(parents=True, exist_ok=True)
    fig_name = folder / f'{target}.pdf'
    plt.savefig(fig_name, bbox_inches='tight')
    # plt.show()
    print(f'Figure saved in {fig_name}')
    plt.close()


# line_center = 2322.25
line_center = 2470.6
line_width = 0.25
order = 2

for target in targets:
    get_line_offset(target, order, line_center=line_center, line_width=line_width)