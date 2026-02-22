from retrieval_base.retrieval import Retrieval
import retrieval_base.figures as figs
from retrieval_base.config import Config
from retrieval_base.auxiliary_functions import spirou_sample, read_spirou_sample_csv, find_run, load_romano_models

# import config_freechem as conf
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import pandas as pd
from datetime import datetime
plt.style.use('/home/dario/phd/retrieval_base/HBDs/my_science.mplstyle')

base_path = '/home/dario/phd/retrieval_base/'
out_path = '/home/dario/phd/red_dwarf_isotopes/data' # store for reproducibility

def save_spectroscopic_data_to_csv(target, run, wave, flux, err, model, rv, 
                                  spline_cont, offset, divide_spline=False, 
                                  orders=None):
    """
    Save spectroscopic data to CSV file for reproducibility.
    
    Parameters:
    -----------
    target : str
        Target name
    run : str
        Retrieval run identifier
    wave : array
        Wavelength array (RV corrected)
    flux : array
        Observed flux
    err : array
        Error on flux
    model : array
        Best fit model
    rv : float
        Radial velocity correction in km/s
    spline_cont : array
        Spline continuum correction
    offset : float
        Plotting offset applied to data
    divide_spline : bool
        Whether spline continuum correction was applied
    orders : list
        List of spectral orders included
    """
    
    # Create output directory for target
    target_dir = pathlib.Path(out_path) / target
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for each order
    for order in (orders or range(len(wave))):
        # Get data for this order
        wave_order = wave[order]
        flux_order = flux[order] 
        model_order = model[order]
        spline_order = spline_cont[order] if spline_cont.ndim > 1 else spline_cont
        err_order = err[order] if err.ndim > 1 else err
        
        # Create data dictionary
        data = {
            'wavelength_nm': wave_order,
            'observed_flux': flux_order,
            'flux_error': err_order,
            'best_fit_model': model_order,
            'spline_continuum': spline_order,
            'residuals': flux_order - model_order
        }
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Create comprehensive header
        header_lines = [
            "# Supplementary data for best-fit atmospheric model comparison",
            f"# Target: {target}",
            f"# Spectral order: {order}",
            f"# Retrieval run: {run}",
            f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "# Authors: Darío González Picos, Ignas Snellen and Sam de Regt",
            "# Contact: picos@strw.leidenuniv.nl",
            "#",
            "# Data description:",
            "# This file contains high-resolution K-band spectroscopic data and",
            "# best-fit atmospheric model from Bayesian retrieval analysis of M dwarf",
            "# stellar atmospheres. Data obtained with SPIRou at CFHT.",
            "#",
            "# Observational details:",
            f"# - Radial velocity correction: {rv:.3f} km/s",
            f"# - RV correction applied: Yes (wavelengths are barycentric)",
            f"# - Spline continuum correction applied: {'Yes' if divide_spline else 'No'}",
            f"# - Plotting offset applied: {offset:.3f}",
            f"# - Spectral resolution: R ~ 75,000",
            "#",
            "# Retrieval methodology:",
            "# - Atmospheric modeling with petitRADTRANS radiative transfer code",
            "# - Bayesian parameter estimation using PyMultiNest nested sampling",
            "# - Free chemistry retrieval of molecular abundances",
            "# - Temperature-pressure profile retrieval",
            "# - Instrumental effects modeling (LSF, telluric contamination)",
            "#",
            "# Column descriptions:",
            "# wavelength_nm: Wavelength in nanometers (barycentric, RV corrected)",
            "# observed_flux: Observed normalized flux",
            "# flux_error: 1-sigma uncertainty on observed flux",
            "# best_fit_model: Best-fit atmospheric model flux",
            "# spline_continuum: Spline continuum normalization function",
            "# residuals: Observed minus model residuals (flux - model)",
            "#",
            "# Data processing notes:",
            "# - Wavelengths corrected for radial velocity using:",
            "#   λ_corrected = λ_observed × (1 - RV/c)",
            "# - Flux normalized to continuum level",
            f"# - Spline continuum {'divided out from model and multiplied to data' if divide_spline else 'not applied'}",
            f"# - Plotting offset of {offset:.3f} added for visualization purposes",
            "#",
            "# Units:",
            "# - Wavelength: nanometers (nm)",
            "# - Flux: normalized (dimensionless)",
            "# - Error: normalized flux units",
            "# - Radial velocity: km/s",
            "#",
            "# Quality metrics:",
            f"# - Data points in this order: {len(wave_order)}",
            f"# - Wavelength range: {np.nanmin(wave_order):.2f} - {np.nanmax(wave_order):.2f} nm",
            f"# - Mean S/N ratio: {np.nanmean(flux_order/err_order):.1f}",
            "#",
            "# References:",
            "# - SPIRou instrument: Donati et al. (2020), MNRAS, 498, 5684",
            "# - petitRADTRANS: Mollière et al. (2019), A&A, 627, A67",
            "# - PyMultiNest: Buchner et al. (2014), A&A, 564, A125",
            "# - Retrieval methodology: Brogi & Line (2019), AJ, 157, 114",
            "#",
            "# Notes:",
            "# - Each row represents one wavelength point",
            "# - NaN values indicate masked or invalid data points",
            "# - Residuals can be used to assess model quality",
            "# - Spline continuum function shows instrumental/stellar continuum shape",
            "#"
        ]
        
        # Save to CSV file
        csv_file = target_dir / f'supplementary_best_fit_model_order_{order}.csv'
        
        with open(csv_file, 'w') as f:
            # Write header
            for line in header_lines:
                f.write(line + '\n')
            
            # Write data
            df.to_csv(f, index=False, float_format='%.6f')
        
        print(f'Saved spectroscopic data for order {order} to {csv_file}')
    
    return target_dir

def main(target, ax, orders=[0], offset=0.0, run=None, text_x=None, save_csv=True, **kwargs):

    if target not in os.getcwd():
        os.chdir(base_path + target)

    outputs = pathlib.Path(base_path) / target / 'retrieval_outputs'
    # find dirs in outputs
    print(f' outputs = {outputs}')
    dirs = [d for d in outputs.iterdir() if d.is_dir() and 'sphinx' in d.name and '_' not in d.name]
    runs = [int(d.name.split('sphinx')[-1]) for d in dirs]
    if run is None:
        # run = 'fc5'  # default
        run = find_run(base_path, target)


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
    
    # Prepare error array for all orders
    err_all = np.ones_like(wave) * np.nan
    for order in range(len(wave)):
        mask_i = ret.d_spec['spirou'].mask_isfinite[order,0]
        err_all[order][mask_i] = ret.Cov['spirou'][order][0].err * s[order]
    
    # Save spectroscopic data to CSV for reproducibility
    if save_csv:
        save_spectroscopic_data_to_csv(
            target=target,
            run=run,
            wave=wave,
            flux=flux,
            err=err_all,
            model=m,
            rv=rv,
            spline_cont=spline_cont,
            offset=offset,
            divide_spline=divide_spline,
            orders=orders
        )
    
    lw = kwargs.get('lw', 1.0)
    color = kwargs.get('color', 'orange')
    for i, order in enumerate(orders):
        err_i = err_all[order]
        
        residuals_i = flux[order] - m[order]
        file_name = ret.conf_output + f'residuals_{order}.npy'
        np.save(file_name, np.array([wave[order], residuals_i, err_i]))
        print(f'Residuals saved as {file_name}')
        
        ax.plot(wave[order], flux[order], color='k',lw=lw)
        ax.fill_between(wave[order], flux[order]-err_i, flux[order]+err_i, alpha=0.2, color='k', lw=0)
        ax.plot(wave[order], m[order], label=target,lw=lw, color=color)
        
        # add text above spectra in units of data
        if order == 1:
            mask_i = ret.d_spec['spirou'].mask_isfinite[order,0]
            text_pos = [np.nanmin(wave[order, mask_i])-4.5, np.nanquantile(flux[order, :len(flux[order]//6)], 0.90)]
            if text_x is not None:
                text_pos[0] = text_x
            # add white box around text
            ax.text(*text_pos, target.replace('gl','Gl'), color='k', fontsize=12, weight='bold', transform=ax.transData)
        
    
    return ret


df = read_spirou_sample_csv()
names = df['Star'].to_list()
teff =  dict(zip(names, [float(t.split('+-')[0]) for t in df['Teff (K)'].to_list()]))

targets = [s.replace('Gl ','gl') for s in names]
temperature_dict = {t:v for t,v in zip(targets, teff.values())}
# norm = plt.Normalize(min(temperature_dict.values()), max(temperature_dict.values()))
norm = plt.Normalize(min(temperature_dict.values()), 4000.0)
cmap = plt.cm.plasma
dark_background = True
def plot(text_x=None):
    fig, ax = plt.subplots(1,1, figsize=(12,6), tight_layout=True,
                           facecolor='black' if dark_background else 'white')

    # orders = [0]
    orders_str = [str(o) for o in orders]
    # colors = plt.cm.
   
    for t, target in enumerate(targets):
        temperature = temperature_dict[target]
        color = cmap(norm(temperature))
        
        ret = main(target, ax=ax, offset=0.46*(len(targets)-1-t), orders=orders,
                # run=spirou_sample[target[2:]][1],
                run='fc5', # DGP 2025-06-10: fix run to fc5
                lw=0.4, color=color,
                text_x=text_x, divide_spline=False, save_csv=True)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Only needed for color bar
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.03, aspect=20, location='right')
    cbar.set_label('Temperature (K)')

    xlim = ax.get_xlim()
    ax.set_xlim((xlim[0]-12, xlim[1]-6))

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Flux + offset')
    # figure still saved in base_path
    
    fig_name = base_path + 'paper/latex/figures/new_best_fit_model' + "-".join(orders_str) + ".pdf"
    if dark_background:
        fig_name = fig_name.replace('.pdf', '_dark.pdf')
    fig.savefig(fig_name, dpi=300, facecolor='black' if dark_background else 'white')
    print(f'Figure saved as {fig_name}')

    show = False
    if show:
        plt.show()
    else:
        plt.close(fig)
    
text_x = 2273.5
orders = [0,1,2]
plot(text_x=text_x)