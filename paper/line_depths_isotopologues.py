""" Compare line depths of CO isotopologues """
from retrieval_base.retrieval import Retrieval
import retrieval_base.figures as figs
from retrieval_base.config import Config
# import config_freechem as conf
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
from retrieval_base.auxiliary_functions import spirou_sample, read_spirou_sample_csv, get_path

def find_run(base_path, target, run=None):
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
    
    return run

def get_model(ret, bestfit_params, order, return_data=False):
    ret.evaluate_model(bestfit_params)
    ret.PMN_lnL_func()
    
    m = np.squeeze(ret.LogLike['spirou'].m)[order] #+ offset
    chi2 = ret.LogLike['spirou'].chi_squared_red
    
    if return_data:
        wave = np.squeeze(ret.d_spec['spirou'].wave)[order]
        flux = np.squeeze(ret.d_spec['spirou'].flux)[order]
        return wave, flux, m, chi2
    return m, chi2

base_path = '/home/dario/phd/retrieval_base/'
# target = 'gl725A'
target = 'gl205'
# run = 'fc3'
run = find_run(base_path, target, run=None)
order = 0
if target not in os.getcwd():
    os.chdir(base_path + target)
config_file = 'config_freechem.txt'
conf = Config(path=base_path, target=target, run=run)(config_file)

ret = Retrieval(
            conf=conf, 
            evaluation=False,
            )
path_data = pathlib.Path(base_path) / target / 'retrieval_outputs' / run  / 'test_data'
path_plots = pathlib.Path(base_path) / target / 'retrieval_outputs' / run  / 'test_plots'


bestfit_params, posterior = ret.PMN_analyze()
bestfit_params_dict = dict(zip(ret.Param.param_keys, bestfit_params))
wave, flux, m, chi2 = get_model(ret, bestfit_params, order, return_data=True)

# save model with filename

# generate model with different isotope ratios
# key = 'log_12CO/C18O'
key = 'log_12CO/13CO'
label = ret.Param.param_mathtext[key]
assert key in bestfit_params_dict, f'{key} not in bestfit_params_dict {bestfit_params_dict.keys()}'

new_values_dict = {'log_12CO/13CO': np.arange(1.6, 4.0+0.2, 0.2),
              'log_12CO/C18O': np.arange(2.0, 4.0+0.2, 0.2),
}
new_values = new_values_dict[key]
chi2_list = []
for new_value in new_values:
# new_value = 2.8
    bestfit_params_dict_copy = bestfit_params_dict.copy()
    print(f' Updating {key} from {bestfit_params_dict[key]:.2f} to {new_value:.2f}')
    bestfit_params_dict_copy[key] = new_value
    bestfit_params_new = np.array([bestfit_params_dict_copy[k] for k in ret.Param.param_keys])
    m_new, chi2_new = get_model(ret, bestfit_params_new, order, return_data=False)
    chi2_list.append(chi2_new)
    
main_label = 'CO'
isotope = 'carbon' if key == 'log_12CO/13CO' else 'oxygen'
isotope_posterior_file = base_path + target + '/retrieval_outputs/' + run + f'/{main_label}_{isotope}_isotope_posterior.npy'
if os.path.exists(isotope_posterior_file) and False: # FIXME: convert to linear scale
    isotope_posterior = np.load(isotope_posterior_file)

    quantiles = np.quantile(isotope_posterior, [0.16, 0.5, 0.84])
else:
    quantiles = [np.nan, bestfit_params_dict[key], np.nan]

fig, ax = plt.subplots(1,1, figsize=(5,5))
ax.plot(new_values, chi2_list, '--o', color='k', zorder=-1)

if quantiles[0] is not np.nan:
    ax.errorbar(quantiles[1], chi2, xerr=[[quantiles[1]-quantiles[0]], [quantiles[2]-quantiles[1]]], fmt='o', color='red', zorder=1)
else:
    ax.scatter(bestfit_params_dict[key], chi2,s=60)
ax.set(xlabel=label, ylabel='chi2', title=f'{target}')
# plt.show()
fig_name = path_plots / f'chi2_{key.replace("/","_")}.pdf'
fig.savefig(fig_name)
print(f' Saved {fig_name}')
plt.close(fig)
    
fig, ax = plt.subplots(3,1, figsize=(14,6), sharex=True, gridspec_kw={'height_ratios': [3, 2, 2]})
ax[0].plot(wave, flux, color='k', lw=0.5)
ax[0].plot(wave, m, color='green', lw=1.0)
ax[0].plot(wave, m_new, color='orange', lw=1.0) 


res_model = 100 * (m_new - m)
ax[1].plot(wave, res_model, color='blue', lw=1.0)
ax[1].set_ylabel('Model Residuals [%]')

res = 100*(flux-m)
ax[-1].plot(wave, res, color='green', lw=1.0, label=f'{label}={bestfit_params_dict[key]:.2f} ({chi2:.2f})')

res_new = 100*(flux-m_new)

MAD = np.nanmedian(np.abs(res))
ax[-1].text(s=f'MAD={MAD:.2f} %', x=0.03, y=0.8, transform=ax[-1].transAxes)
ax[-1].plot(wave, res_new, color='orange', lw=1.0, label=f'{label}={new_value:.2f} ({chi2_new:.2f})')
ax[-1].axhline(0, color='k', lw=0.5)
ax[-1].legend()
ax[-1].set_xlabel('Wavelength [$\mu m$]') 
ax[-1].set_ylabel('Residuals [%]')


ylim = ax[-1].get_ylim()
# ax[1].set_ylim(ylim)
# plt.show()
fig_name = path_plots / f'bestfit_residuals_{key.replace("/","_")}.pdf'
fig.savefig(fig_name)
print(f' Saved {fig_name}')
plt.close(fig)



