""" 2024-12-11 Generate same fig as Fig. 1 for appendix with all orders and all targets """
from retrieval_base.retrieval import Retrieval
import retrieval_base.figures as figs
from retrieval_base.config import Config
from retrieval_base.auxiliary_functions import spirou_sample, read_spirou_sample_csv
# import config_freechem as conf
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import pandas as pd
from datetime import datetime
# plt.style.use('/home/dario/phd/retrieval_base/HBDs/my_science.mplstyle')
import scienceplots

# reset to default
plt.style.use('default')
# plt.style.use(['latex-sans'])
plt.style.use(['sans'])
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
font_size = 6  # Keep within 5-7pt range as required by Nature
plt.rcParams['font.size'] = font_size

# Ensure RGB color mode for Nature requirements
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

plt.rcParams['pdf.fonttype']   = 42
plt.rcParams['ps.fonttype']    = 42
plt.rcParams['svg.fonttype']   = 'none'
plt.rcParams['text.usetex']    = False

# Remove path_effects import as we won't use outlined text
# import matplotlib.patheffects as path_effects

base_path = '/home/dario/phd/retrieval_base/'
nat_path = '/home/dario/phd/nat/'
out_path = '/home/dario/phd/red_dwarf_isotopes/data' # store for reproducibility

def save_all_orders_to_csv(target, run, wave, flux, err, model, spline_cont, 
                          mask, rv=None, divide_spline=False, teff=None, spt=None, offset=0.0):
    """
    Save all spectral orders data to a single CSV file for reproducibility.
    
    Parameters:
    -----------
    target : str
        Target name
    run : str
        Retrieval run identifier
    wave : array
        Wavelength array for all orders (RV corrected)
    flux : array
        Observed flux for all orders
    err : array
        Error on flux for all orders
    model : array
        Best fit model for all orders
    spline_cont : array
        Spline continuum correction for all orders
    mask : array
        Finite mask for valid data points
    rv : float, optional
        Radial velocity correction in km/s
    divide_spline : bool
        Whether spline continuum correction was applied
    teff : float, optional
        Effective temperature in K
    spt : str, optional
        Spectral type
    offset : float, optional
        Plotting offset applied to data (stored in header only)
    """
    
    # Create output directory for target
    target_dir = pathlib.Path(out_path) / target
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Flatten all arrays and create order column
    wave_flat = []
    flux_flat = []
    err_flat = []
    model_flat = []
    spline_flat = []
    order_flat = []
    valid_flat = []
    
    n_orders = len(wave)
    
    for order in range(n_orders):
        # Get data for this order (remove plotting offset from flux and model)
        wave_order = wave[order]
        flux_order = flux[order] - offset  # Remove plotting offset
        err_order = err[order]
        model_order = model[order] - offset  # Remove plotting offset
        spline_order = spline_cont[order]
        mask_order = mask[order]
        
        # Flatten and append
        n_pixels = len(wave_order)
        wave_flat.extend(wave_order)
        flux_flat.extend(flux_order)
        err_flat.extend(err_order)
        model_flat.extend(model_order)
        spline_flat.extend(spline_order)
        order_flat.extend([order] * n_pixels)
        valid_flat.extend(mask_order)
    
    # Convert to numpy arrays
    wave_flat = np.array(wave_flat)
    flux_flat = np.array(flux_flat)
    err_flat = np.array(err_flat)
    model_flat = np.array(model_flat)
    spline_flat = np.array(spline_flat)
    order_flat = np.array(order_flat)
    valid_flat = np.array(valid_flat)
    
    # Create data dictionary
    data = {
        'spectral_order': order_flat,
        'wavelength_nm': wave_flat,
        'observed_flux': flux_flat,
        'flux_error': err_flat,
        'petitradtrans_model': model_flat,
        'spline_continuum': spline_flat,
        'valid_pixel': valid_flat.astype(int)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create comprehensive header
    header_lines = [
        "# Supplementary data for best-fit atmospheric models - all spectral orders",
        f"# Target: {target}",
        f"# Retrieval run: {run}",
        f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "# Authors: Darío González Picos, Ignas Snellen and Sam de Regt",
        "# Contact: picos@strw.leidenuniv.nl",
        "#",
        "# Data description:",
        "# This file contains high-resolution K-band spectroscopic data and",
        "# best-fit atmospheric models from Bayesian retrieval analysis of M dwarf",
        "# stellar atmospheres. All spectral orders are included in a single file.",
        "# Data obtained with SPIRou at CFHT.",
        "#",
        "# Target properties:",
        f"# - Effective temperature: {teff:.0f} K" if teff else "# - Effective temperature: Not available",
        f"# - Spectral type: {spt}" if spt else "# - Spectral type: Not available",
        f"# - Number of spectral orders: {n_orders}",
        f"# - Total data points: {len(wave_flat)}",
        "#",
        "# Observational details:",
        f"# - Radial velocity correction: {rv:.3f} km/s" if rv else "# - Radial velocity correction: Not available",
        f"# - RV correction applied: {'Yes' if rv else 'Unknown'} (wavelengths are barycentric)",
        f"# - Spline continuum correction applied: {'Yes' if divide_spline else 'No'}",
        f"# - Plotting offset: {offset:.3f} (removed from saved data)",
        f"# - Spectral resolution: R ~ 75,000",
        f"# - Wavelength coverage: {np.nanmin(wave_flat):.1f} - {np.nanmax(wave_flat):.1f} nm",
        "#",
        "# Column descriptions:",
        "# spectral_order: SPIRou spectral order number (0, 1, 2 for K-band)",
        "# wavelength_nm: Wavelength in nanometers (barycentric, RV corrected)",
        "# observed_flux: Observed normalized flux",
        "# flux_error: 1-sigma uncertainty on observed flux",
        "# petitradtrans_model: Best-fit atmospheric model flux from petitRADTRANS",
        "# spline_continuum: Spline continuum normalization function",
        "# valid_pixel: 1 for valid data points, 0 for masked/invalid pixels",
        "#",
        "# Data processing notes:",
        "# - Wavelengths corrected for radial velocity using:",
        "#   λ_corrected = λ_observed × (1 - RV/c)",
        "# - Flux normalized to continuum level",
        f"# - Spline continuum {'divided out from model and applied to data' if divide_spline else 'not applied'}",
        f"# - Plotting offset of {offset:.3f} has been removed from flux and model data",
        "# - Invalid pixels (telluric contamination, cosmic rays) marked in valid_pixel column",
        "#",
        "# Units:",
        "# - Wavelength: nanometers (nm)",
        "# - Flux: normalized (dimensionless)",
        "# - Error: normalized flux units",
        "# - Temperature: Kelvin (K)",
        "# - Radial velocity: km/s",
        "#",
        "# Quality metrics:",
        f"# - Mean S/N ratio: {np.nanmean(flux_flat[valid_flat.astype(bool)]/err_flat[valid_flat.astype(bool)]):.1f}",
        f"# - Valid pixels: {np.sum(valid_flat)} / {len(valid_flat)} ({100*np.sum(valid_flat)/len(valid_flat):.1f}%)",
        f"# - RMS residuals: {np.nanstd((flux_flat - model_flat)[valid_flat.astype(bool)]):.4f}",
        "#",
        "# Notes:",
        "# - Each row represents one wavelength point",
        "# - NaN values are preserved as NaN (not converted to other values)",
        "# - Residuals can be calculated as: observed_flux - petitradtrans_model",
        "# - Spline model contains the fitted continuum with 25 equally spaced knots",
        "# - Data and model values have plotting offset removed for scientific analysis",
        "#"
    ]
    
    # Save to CSV file
    csv_file = target_dir / f'supplementary_best_fit_model_all_orders.csv'
    
    with open(csv_file, 'w') as f:
        # Write header
        for line in header_lines:
            f.write(line + '\n')
        
        # Write data (preserve NaN values)
        df.to_csv(f, index=False, float_format='%.6f', na_rep='nan')
    
    print(f'Saved all orders spectroscopic data to {csv_file}')
    return csv_file

def main(target, ax, order=0, offset=0.0, run=None, text_x=None, offset_x=0.0, fl=1.0, cache=True, save_csv=False, **kwargs):
    
    
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
    rv_value = None  # Initialize RV value
    
    if bestfit_spec_file.exists() and cache:
        
        print(f' Bestfit model found in {bestfit_spec_file}')
        bestfit_data = np.load(bestfit_spec_file, allow_pickle=True)
        if len(bestfit_data) == 7:  # New format with RV
            wave, flux, err, mask, m, spline_cont, rv_value = bestfit_data
        else:  # Old format without RV
            wave, flux, err, mask, m, spline_cont = bestfit_data
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
        
        rv_value = bestfit_params[list(ret.Param.param_keys).index('rv')]
        
        wave = np.squeeze(ret.d_spec['spirou'].wave) * (1 - rv_value/299792.458)
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
        mask = np.squeeze(ret.d_spec['spirou'].mask_isfinite)
        
        # print(f' s.shape = {s.shape}')
        # print(f'err.shape = {err.shape}')
        # print(f'mask.shape = {mask.shape}')
        # if debug:
        #     breakpoint()
        # for i in range(3):
            # err_i = np.ones_like(wave[order]) * np.nan
        for ii in range(ret.d_spec['spirou'].n_orders):
            # this is necessary to properly store the err for each order
            err_order = err[ii]
            err_order[mask[ii]] = ret.Cov['spirou'][ii][0].err * s[ii,0]
            assert np.sum(np.isnan(err_order)) < np.size(err_order), f'WARNING: {target}: All err are NaN for order {ii}'
            print(f'[DEBUG] sum(nans(err_order)) = {np.sum(np.isnan(err_order))}')
            err[ii] = err_order
        
        # save file with RV value
        np.save(bestfit_spec_file, np.array([wave, flux, err, mask, m, spline_cont, rv_value], dtype=object))
        print(f'Bestfit model saved as {bestfit_spec_file}')

    divide_spline = kwargs.get('divide_spline', False)
    if divide_spline:
        m /= spline_cont
        flux /= spline_cont
            
    m += offset
    flux += offset
    # residuals, save as npy with wave, residuals, err
    # np.save(ret.conf_output + 'residuals.npy', np.array([wave, flux-m, ret.Cov['spirou'][0].err * s[0]]))
    # select 200 pixels without nans
    nans = np.isnan(flux[order])
    flux_nonans = flux[order, ~nans]
    
    if fl != 1.0:
        scale = np.median(flux_nonans[:100]) / fl
        
        flux /= scale
        m /= scale
        print(f' Flux scaled by {scale}')
    
    lw = kwargs.get('lw', 0.7)
    color = kwargs.get('color', 'orange')
    # for i, order in enumerate(orders):

    
    residuals_i = flux[order] - m[order]
    file_name = test_output / f'residuals_{order}.npy'
    np.save(file_name, np.array([wave[order], residuals_i, err[order]]))
    print(f'Residuals saved as {file_name}')
    
    
    ax[0].plot(wave[order], flux[order], color='k',lw=lw)
    ax[0].fill_between(wave[order], flux[order]-err[order], flux[order]+err[order], alpha=0.2, color='k', lw=0)
    ax[0].plot(wave[order], m[order], label=target,lw=lw, color=color)
    
    ax[1].plot(wave[order], residuals_i, color=color, lw=lw, alpha=0.6)
    # ax[i].plot(wave[order],spline_cont[order]+offset, color='red', lw=lw)
    
    # add text above spectra in units of data
    if order==0:
        text_pos = [np.nanmin(wave[order, mask[order]]), np.nanquantile(flux[order, :len(flux[order]//2)], 0.90)-0.25]
        if text_x is not None:
            text_pos[0] = text_x[0]
        # add white box around text
        
        # pe = [path_effects.withStroke(linewidth=2, foreground='w')]
        s = target.replace('gl','')
        ax[0].text(*text_pos, s, color='k', fontsize=font_size, weight='bold', transform=ax[0].transData)
    
   
    
    # if order ==0:
    if order == 99: # ignore for now... too many errorbars
        # take the mean error of ALL ORDERS here
        sigmas = [1,2,3]
        # mean_err = np.nanmean(err[order])
        mean_err = np.nanmedian(err) # NEW 2024-12-11
        # only plot once (order=0)
        alpha = [1.0, 0.7, 0.4]
        for s, sigma in enumerate(sigmas):
            ax[1].errorbar(2250-offset_x, 0, yerr=mean_err*sigma, color=color, zorder=1, lw=1.5, capsize=0, capthick=1.5, alpha=alpha[s])
        
        
    
    
    # Return data for CSV export if needed
    if save_csv and order == 0:  # Only save CSV once per target (when processing first order)
        return {
            'wave': wave,
            'flux': flux,
            'err': err,
            'model': m,
            'spline_cont': spline_cont,
            'mask': mask,
            'rv': rv_value,
            'divide_spline': divide_spline,
            'median_flux': np.median(flux_nonans[-100:])
        }
    
    return np.median(flux_nonans[-100:])


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
ignore_names = ['Gl 3622']
# norm = plt.Normalize(min(temperature_dict.values()), max(temperature_dict.values()))
# norm = plt.Normalize(min(teff.values()), 4000.0)
norm = plt.Normalize(3000, 3900.0)
cmap = plt.cm.coolwarm_r

my_targets_id = ['338B', '205', '411', '436','699', '1286']
# my_targets = ['gl'+t for t in my_targets_id]
my_targets = [s.replace('Gl ', 'gl') for s in names if s not in ignore_names][::-1]


def plot(orders, text_x=None, xlim=None, save_csv=True, **kwargs):
    width_mm = 88.0
    aspect_ratio = 5/8
    height_mm = width_mm/aspect_ratio
    mm_to_inch = 0.0393701
    fig, ax = plt.subplots(2,1, figsize=(width_mm*mm_to_inch, height_mm*mm_to_inch),
                           sharex=True, gridspec_kw={'height_ratios': [15, 1],
                                                                        'hspace': 0.03,
                                                                        'top': 0.97,
                                                                        'bottom': 0.13,
                                                                        'left': 0.10,
                                                                        'right': 0.99})
    

    # orders = [0]
    orders_str = [str(o) for o in orders]
    # colors = plt.cm.
    count = 0
    for t, name in enumerate(names):
        target = name.replace('Gl ', 'gl')
        if target not in my_targets:
            continue
        count += 1
        temperature = teff[name]
        color = cmap(norm(temperature))
        
        # offset = 0.42*(len(names)-t)
        offset = 0.54*(len(my_targets)-my_targets.index(target)-1)
        fl  = 1.0
        cache = kwargs.pop('cache', True)
        csv_data = None
        
        for order in orders:
            result = main(target, ax=ax, offset=offset, order=order,
                    # run=None, 
                    run='5', # DGP 2025-06-10: fix run to fc5
                    lw=0.25, color=color,
                    text_x=text_x, 
                    divide_spline=True,
                    offset_x=-2*count,
                    fl=fl,
                    cache=cache if order == 0 else True,
                    save_csv=save_csv,
                    **kwargs)
            
            # Handle CSV data export (only saved once per target)
            if isinstance(result, dict):
                csv_data = result
                fl = result['median_flux']
            else:
                fl = result
                
            if debug:
                print(f'Checkpoint {target} {order}')
                break
        
        # Save CSV data after processing all orders for this target
        if save_csv and csv_data is not None:
            save_all_orders_to_csv(
                target=target,
                run='fc5',
                wave=csv_data['wave'],
                flux=csv_data['flux'],
                err=csv_data['err'],
                model=csv_data['model'],
                spline_cont=csv_data['spline_cont'],
                mask=csv_data['mask'],
                rv=csv_data['rv'],
                divide_spline=csv_data['divide_spline'],
                teff=temperature,
                spt=spt[name],
                offset=offset
            )
        
        ax[0].text(s=spt[name].split('.')[0].replace('V',''), x=text_x[1]-4, y=0.75+offset, transform=ax[0].transData,
                    color=color, fontsize=font_size, weight='bold')
        
        
    ax[-1].axhline(0.0, color='k', lw=0.5, zorder=-1)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Only needed for color bar
    
    # create ax for colorbar
    cbar_ax = fig.add_axes([1.005, 0.1948, 0.018, 0.775])
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation='vertical', pad=0.01, aspect=40, location='right')
    # cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.01, aspect=80, location='right')
    cbar.set_label('Temperature (K)', fontsize=font_size)

    if xlim is not None:
        print(f'Setting xlim to {xlim}')
        ax[0].set_xlim(xlim)
    else:
        print(f'No xlim provided, using default')
        xlim = ax.get_xlim()
        ax[0].set_xlim((xlim[0]-5, xlim[1]-3))


    ax[0].set_ylim(0.25, 18.0)  
    ax[-1].set_xlabel('Wavelength (nm)', fontsize=font_size)
    ylim_res = ax[1].get_ylim()
    # make symmetric
    ylim_res_sym = max(abs(ylim_res[0]), abs(ylim_res[1]))
    ax[1].set_ylim(-ylim_res_sym, ylim_res_sym)
    ax[0].set_ylabel('Flux + offset', fontsize=font_size)
    ax[1].set_ylabel('Residuals', labelpad=0, fontsize=font_size)
    
    # Ensure tick labels also follow journal requirements
    for axis in [ax[0].xaxis, ax[0].yaxis, ax[1].xaxis, ax[1].yaxis]:
        axis.set_tick_params(labelsize=font_size)
    
    # Ensure colorbar tick labels follow requirements
    cbar.ax.tick_params(labelsize=font_size)

    # fig_name = base_path + 'paper/latex/figures/best_fit_model' + "-".join(orders_str) + ".pdf"
    if debug:
        return
    # fig name to nat path, data to out_path
    fig_name = nat_path + 'nat_fig_1_revised.pdf'
    # Save in RGB mode with editable text as required by Nature
    fig.savefig(fig_name, bbox_inches='tight', dpi=300, format='pdf', 
                facecolor='white', edgecolor='none', 
                metadata={'Creator': 'Dario Gonzalez Picos', 
                          'Producer': 'matplotlib'})
    print(f'Figure saved as {fig_name}')

    show = False
    if show:
        plt.show()
    else:
        plt.close(fig)
    
# text_x = [(2285.5, 2364.),
#           (2358.0, 2438.),
#           (2435.0, 2510.0),
# ]
debug = False
order = 0
# xlim = (2282, 2364) # for order 0
# xlim = (2270, 2500) # for all orders
xlim = (2274, 2495)
text_x = (xlim[0]+1.5, xlim[1]-9)
plot([0,1,2], text_x=text_x, xlim=xlim, cache=False, save_csv=True)