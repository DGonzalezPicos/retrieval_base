import pathlib
import numpy as np
import os
import matplotlib.pyplot as plt
# pdf pages
from matplotlib.backends.backend_pdf import PdfPages
import copy

from retrieval_base.retrieval import Retrieval
from retrieval_base.spectrum import ModelSpectrum
import retrieval_base.auxiliary_functions as af
from retrieval_base.config import Config
# import config_jwst as conf


def get_species(ret,
                 wave, 
                 line_species, 
                 bestfit_params_dict, 
                 disk_species=False,
                  **kwargs):

    n_orders = len(wave)
    
    bestfit_params_dict_copy = copy.deepcopy(bestfit_params_dict)
    
    if disk_species:
        line_species = line_species.replace('_disk','')
        bestfit_params_dict_copy[f'log_A_au_{line_species}'] = -5.0
        bestfit_params_dict_copy[f'log_N_mol_{line_species}'] = 15.0
    
    else:
            
        if conf.chem_mode == 'fastchem':
            bestfit_params_dict_copy[f'alpha_{line_species}'] = -14.0
            if line_species in conf.isotopologues_dict.keys():
                print(f'Found isotopologue {line_species} with ratio {conf.isotopologues_dict[line_species][0]}')
                log_ratio = conf.isotopologues_dict[line_species][0]
                bestfit_params_dict_copy[log_ratio] = 4.0
        else:
            bestfit_params_dict_copy[f'log_{line_species}'] = -14.0

    ret.evaluate_model(np.array(list(bestfit_params_dict_copy.values())))
    ret.PMN_lnL_func()
    chi2 = ret.LogLike[w_set].chi_squared_red
    
    m_flux_wo = ret.LogLike[w_set].m_flux
    m_wave_pRT_grid = ret.pRT_atm_broad['NIRSpec'].wave_pRT_grid
    m_flux_pRT_grid = ret.pRT_atm_broad['NIRSpec'].flux_pRT_grid
    
    
    # m_flux_wo_species = m_flux - m_flux_wo
    m_flux_wo_species = m_flux_wo
    # m_flux_wo_species_pRT_grid = [m_flux_pRT_grid_full[i] - m_flux_pRT_grid[i] for i in range(n_orders)]
    m_flux_wo_species_pRT_grid = m_flux_pRT_grid
    
    m_spec_wo_species = ModelSpectrum(wave=wave, flux=m_flux_wo_species)
        
    return m_flux_wo, m_spec_wo_species, m_flux_wo_species_pRT_grid


def get_ccf(
    ret: Retrieval,
    m_spec: ModelSpectrum,
    m_wave_pRT_grid: np.ndarray,
    m_flux_pRT_grid_full: np.ndarray,
    species: str,
    rv_max: float = 2000.0,
    rv_step: float = 5.0,
    rv_noise: float = 400.0,
    hpf_sigma: float = 10.0
) -> None:
    """
    Compute the cross-correlation function (CCF) of the observed spectrum
    with a model spectrum, and plot the result.

    Parameters
    ----------
    ret : Retrieval
        The retrieval object
    m_spec : ModelSpectrum
        The model spectrum
    m_wave_pRT_grid : np.ndarray
        The wavelength grid of the model spectrum
    m_flux_pRT_grid_full : np.ndarray
        The flux grid of the full model spectrum
    species : str
        The species to compute the CCF for
    rv_max : float, optional
        The maximum radial velocity to consider, by default 2000.0
    rv_step : float, optional
        The step size of the radial velocity grid, by default 5.0
    rv_noise : float, optional
        The radial velocity noise to exclude from the CCF, by default 400.0
    hpf_sigma : float, optional
        The standard deviation of the high-pass filter to apply to the CCF, by default 10.0
    """
        
    disk_species = False
    if 'disk' in species:
        disk_species = True
        label_species = species.replace('_disk',' (disk)')
    else:
        label_species = conf.opacity_params[f'log_{species}'][0][1].replace('\\log\\','')
        
    m_flux_wo, m_spec_wo_species, m_flux_wo_species_pRT_grid = get_species(ret, m_wave, species, bestfit_params_dict,
                                                                           disk_species=disk_species)
    
    # Compute the cross-correlation function
    rv, CCF, d_ACF, m_ACF = af.CCF(d_spec=d_spec,
                                    m_spec=m_spec,
                                    m_wave_pRT_grid=m_wave_pRT_grid,
                                    m_flux_pRT_grid=m_flux_pRT_grid_full,
                                    m_spec_wo_species=m_spec_wo_species,
                                    m_flux_wo_species_pRT_grid=m_flux_wo_species_pRT_grid,
                                    # LogLike=ret.LogLike[w_set],
                                    Cov=ret.Cov[w_set],
                                    rv=np.arange(-rv_max,rv_max+1e-6, rv_step), 
                                    apply_high_pass_filter=(hpf_sigma > 0),
                                    high_pass_filter_sigma=hpf_sigma,
    )
        
    CCF_SNR, ACF_SNR, peak_SNR = af.CCF_to_SNR(rv, 
                                               np.sum(CCF, axis=(0,1)),
                                               ACF=np.sum(m_ACF,axis=(0,1)),
                                               rv_to_exclude=(-rv_noise, rv_noise))
    
    ccf_file = ccf_path / f'RV_CCF_ACF_{species}.txt'
    np.savetxt(ccf_file, np.array([rv, CCF_SNR, ACF_SNR]).T, header='rv CCF ACF')
    print(f' Saved {ccf_file}')
    
    # plot CCF and ACF
    fig, (ax, ax_res) = plt.subplots(2, 1, figsize=(6, 6), tight_layout=True, sharex=True,
                           gridspec_kw={'height_ratios': [3, 1]})
    ax.plot(rv, CCF_SNR, color='k')
    ax.plot(rv, ACF_SNR, color='darkorange', ls='--', alpha=0.9)
    # ax.plot(rv, d_ACF_SNR, color='k', ls=':', alpha=0.4)

    ax.set(ylabel='SNR', title=label_species)
    ax.set_xlim(rv.min(), rv.max())
    ax.axhline(0, color='k', lw=0.5, alpha=0.3)
    
    CCF_RES = CCF_SNR - ACF_SNR
    ax_res.plot(rv, CCF_RES, color='k')
    ax_res.axhline(0, color='darkorange', lw=0.5)
    ax_res.set(xlabel=r'RV / km s$^{-1}$', ylabel='CCF - ACF')
    ylim_res = ax_res.get_ylim()
    # make symmetric ylims for ax_res
    max_res = np.abs(ylim_res).max()
    ax_res.set_ylim(-max_res, max_res)
    
    rv_peak = rv[np.argmax(CCF_SNR)]
    snr_peak = np.max(CCF_SNR)
    print(f' Peak SNR at {rv_peak} km/s: {snr_peak}')
    [axi.axvspan(-rv_noise, rv_noise, color='gray', alpha=0.1) for axi in [ax, ax_res]]
    ax.axvline(rv_peak, lw=1.0, label=f'SNR = {snr_peak:.1f}', color='#CC9900', alpha=0.9) # color='darkgold'
    ax.legend(frameon=False, fontsize=14, loc='upper right', bbox_to_anchor=(1, 1))
    
    fig_name = ccf_path / f'CCF_SNR_{species}.pdf'
    fig.savefig(fig_name)
    print(f' Saved {fig_name}')
    plt.close(fig)  
    
    
if __name__=='__main__':
    import argparse

    # define arguments
    parser = argparse.ArgumentParser(description='Run retrieval')
    parser.add_argument('-t', '--target', type=str, help='Target name')
    parser.add_argument('-r', '--run', type=str, help='Run name')
    args = parser.parse_args()

    target = args.target
    run = args.run

    path = af.get_path()
    config_file = 'config_jwst.txt'
    # target = 'TWA28'
    # run = None
    # run = 'lbl12_G2G3_6'
    w_set='NIRSpec'

    cwd = os.getcwd()
    if target not in cwd:
        nwd = os.path.join(cwd, target)
        print(f'Changing directory to {nwd}')
        os.chdir(nwd)


    conf = Config(path=path, target=target, run=run)(config_file)        
    ccf_path = pathlib.Path(conf.prefix + 'plots/CCF/')
    ccf_path.mkdir(parents=True, exist_ok=True)

    ret = Retrieval(
        conf=conf, 
        evaluation=True,
        plot_ccf=True
        )
    ret.calc_flux_fast = True # backwards compatibility

    bestfit_params, posterior = ret.PMN_analyze()
    bestfit_params_dict = dict(zip(ret.Param.param_keys, bestfit_params))

    ret.evaluate_model(bestfit_params)
    ret.PMN_lnL_func()

    d_spec = ret.d_spec[w_set]
    d_flux = ret.d_spec[w_set].flux

    m_spec = ret.m_spec[w_set]

    m_wave_pRT_grid = ret.pRT_atm_broad['NIRSpec'].wave_pRT_grid
    m_flux_pRT_grid_full = ret.pRT_atm_broad['NIRSpec'].flux_pRT_grid

    m_wave = ret.d_spec[w_set].wave
    m_flux = ret.LogLike[w_set].m_flux

    chi2_full = ret.LogLike[w_set].chi_squared_red

      

    species_list = [k[4:] for k in ret.conf.opacity_params.keys() if 'log_' in k]
    disk_species = getattr(ret.conf, 'disk_species', [])
    species_list += [f'{k}_disk' for k in disk_species]

    # species_list = ['12CO', '13CO']
    # species_list = ['SiO']
    # print(stop)
    rv_max = 2000.0
    rv_step = 5.0
    rv_noise = 400.0
    for species in species_list:
        
        get_ccf(ret, m_spec, m_wave_pRT_grid, m_flux_pRT_grid_full, species, rv_max=rv_max, rv_step=rv_step, rv_noise=rv_noise)