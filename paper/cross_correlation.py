from retrieval_base.retrieval import Retrieval
import retrieval_base.figures as figs
from retrieval_base.config import Config
import retrieval_base.auxiliary_functions as af
from retrieval_base.spectrum import ModelSpectrum
# import config_freechem as conf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import pathlib
import copy


def get_species(ret,
                 wave, 
                 line_species, 
                 bestfit_params_dict, 
                  **kwargs):

    n_orders = len(wave)
    
    bestfit_params_dict_copy = copy.deepcopy(bestfit_params_dict)

            
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
    
    m_flux_wo = ret.LogLike[w_set].m
    m_wave_pRT_grid = ret.pRT_atm_broad['spirou'].wave_pRT_grid
    m_flux_pRT_grid = ret.pRT_atm_broad['spirou'].flux_pRT_grid
    
    
    # m_flux_wo_species = m_flux - m_flux_wo
    m_flux_wo_species = m_flux_wo
    # m_flux_wo_species_pRT_grid = [m_flux_pRT_grid_full[i] - m_flux_pRT_grid[i] for i in range(n_orders)]
    m_flux_wo_species_pRT_grid = m_flux_pRT_grid
    
    m_spec_wo_species = ModelSpectrum(wave=wave, flux=m_flux_wo_species)
        
    return m_flux_wo, m_spec_wo_species, m_flux_wo_species_pRT_grid


def plot_ccf(
    rv: np.ndarray,
    CCF_SNR: np.ndarray,
    ACF_SNR: np.ndarray,
    label_species: str,
    rv_noise: float,
    max_rv_plot: float,
    ccf_path: pathlib.Path,
) -> None:
    """
    Plot the cross-correlation function (CCF) and auto-correlation function (ACF)
    of a spectrum."""
        
    # plot CCF and ACF
    fig, (ax, ax_res) = plt.subplots(2, 1, figsize=(6, 6), tight_layout=True, sharex=True,
                           gridspec_kw={'height_ratios': [3, 1]})
    ax.plot(rv, CCF_SNR, color='k')
    ax.plot(rv, ACF_SNR, color='darkorange', ls='--', alpha=0.9)

    ax.set(ylabel='SNR', title=label_species)
    # ax.set_xlim(rv.min(), rv.max())
    ax.set_xlim(-max_rv_plot, max_rv_plot)
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
    
    return None

def get_ccf(
    ret: Retrieval,
    m_spec: ModelSpectrum,
    m_wave_pRT_grid: np.ndarray,
    m_flux_pRT_grid_full: np.ndarray,
    species: str,
    rv_max: float = 2000.0,
    rv_step: float = 5.0,
    rv_noise: float = 400.0,
    hpf_sigma: float = 10.0,
    ccf_file: pathlib.Path = None,
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
    
    # ccf_file = ccf_path / f'RV_CCF_ACF_{species}.txt'
    if ccf_file is not None:
        np.savetxt(ccf_file, np.array([rv, CCF_SNR, ACF_SNR]).T, header='rv CCF ACF')
        print(f' Saved {ccf_file}')
    
    
    return rv, CCF_SNR, ACF_SNR


base_path = af.get_path(return_pathlib=True)
target = 'gl205'
run = None

if target not in os.getcwd():
    os.chdir(os.path.join(base_path, target))

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


config_file = 'config_freechem.txt'
conf = Config(path=base_path, target=target, run=run)(config_file)
ccf_path = pathlib.Path(conf.prefix + 'plots/CCF/')
ccf_path.mkdir(parents=True, exist_ok=True)

ret = Retrieval(
                    conf=conf, 
                    evaluation=True,
                    plot_ccf=True,

                    )

bestfit_params, posterior = ret.PMN_analyze()
bestfit_params_dict = dict(zip(ret.Param.param_keys, bestfit_params))

ret.evaluate_model(bestfit_params)
ret.PMN_lnL_func()


w_set = 'spirou'
d_spec = ret.d_spec[w_set]
d_flux = ret.d_spec[w_set].flux

m_spec = ret.m_spec[w_set]

m_wave_pRT_grid = ret.pRT_atm_broad['spirou'].wave_pRT_grid
m_flux_pRT_grid_full = ret.pRT_atm_broad['spirou'].flux_pRT_grid

m_wave = ret.d_spec[w_set].wave
m_flux = ret.LogLike[w_set].m

chi2_full = ret.LogLike[w_set].chi_squared_red
    
m_flux_wo, m_spec_wo_species, m_flux_wo_species_pRT_grid = get_species(
                    ret=ret,
                    wave=m_wave, 
                    line_species='CO_36_high_Sam', 
                    bestfit_params_dict=bestfit_params_dict, 
                    )

#  species_list = [k[4:] for k in ret.conf.opacity_params.keys() if 'log_' in k]
species_list = ['12CO', '13CO','C18O', 'OH', 'H2O', 'HF', 'Ca','Na', 'CN',
                'Ti','Mg','Fe']
# TODO: calculate for more species

# species_list = ['SiO']
# print(stop)
rv_max = 1000.0
rv_step = 1.0
rv_noise = 100.0
max_rv_plot = 200.0
cache = True
for species in species_list:
    
    
    ccf_file = ccf_path / f'RV_CCF_ACF_{species}.txt'
    if ccf_file.exists() and cache:
        rv, CCF_SNR, ACF_SNR = np.loadtxt(ccf_file).T
        print(f' Loaded {ccf_file}')
        
    else:
        rv, CCF_SNR, ACF_SNR = get_ccf(ret, m_spec, m_wave_pRT_grid, 
                m_flux_pRT_grid_full, 
                species, 
                rv_max=rv_max, 
                rv_step=rv_step, 
                rv_noise=rv_noise,
                ccf_file=ccf_file,
                )
    
    plot_ccf(rv, 
             CCF_SNR, 
             ACF_SNR, 
             conf.opacity_params[f'log_{species}'][0][1].replace('\\log\\',''),
             rv_noise, 
             max_rv_plot, 
             ccf_path,
    )
