from retrieval_base.retrieval import Retrieval
import retrieval_base.figures as figs
from retrieval_base.config import Config
# import config_freechem as conf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import pathlib

# change working directory to the location of this script
# abspath = os.path.abspath(__file__)
# dname = os.path.dirname(abspath)
# os.chdir(dname)

base_path = '/home/dario/phd/retrieval_base/'

target = 'gl880'
if target not in os.getcwd():
    os.chdir(base_path + target)

# run = 'sphinx_4'
run = 'sphinx_4_v2'
config_file = 'config_freechem.txt'

def get_model(run, bestfit_params=None):
    
    conf = Config(path=base_path, target=target, run=run)(config_file)

    ret = Retrieval(
                conf=conf, 
                evaluation=False,
                )
    if bestfit_params is None:
        
        bestfit_params_dict, _ = ret.PMN_analyze(map=True, return_dict=True)
        bestfit_params = list(bestfit_params_dict.values())
        
    if isinstance(bestfit_params, dict):
        bestfit_params_dict = bestfit_params
    else:
        bestfit_params_dict = dict(zip(ret.Param.param_keys, bestfit_params))
    ret.get_bestfit_model(bestfit_params)
    
    return ret.d_spec['spirou'], ret.bestfit_model, ret.LogLike['spirou'].chi_squared_red, bestfit_params_dict


d_spec, (w, m), chi2, bfp_dict = get_model(run)
d_wave, d_flux = np.squeeze(d_spec.wave), np.squeeze(d_spec.flux)


update_params = {'alpha_K':-4.0, # K has no effect on the chi2...
                 'alpha_Si':-4.0, # Si has no effect on the chi2...
                 'log_Ni':-3.0, # improvement from chi2 ~ 357 to 350 at log_Ni = -3.0
                 'log_Mn':-6.0, # discarded at > -6.0, probably not needed
}
print(f' TODO: continue searching for the missing opacity sources...')


bfp_dict.update(update_params)
bfp = list(bfp_dict.values())

_, (w_new, m_new), chi2_new, _ = get_model('sphinx_4_v3', bestfit_params=bfp_dict)
print(f'Old chi2: {chi2:.3f}, new chi2: {chi2_new:.3f}')
n_orders = m.shape[0]
residuals = d_flux - m
# save wavelength and residuals in one npy file with two columns

save_residuals = True
if save_residuals:
    file_npy = pathlib.Path(base_path) / target / 'retrieval_outputs' / run / 'test_plots' / f'{run}_residuals.npy'
    assert d_wave.shape == residuals.shape, f'Wavelength and residuals shapes do not match: {d_wave.shape} vs {residuals.shape}'
    np.save(file_npy, np.array([d_wave, residuals]))
    print(f'Numpy file saved to {file_npy}')


color_old = 'magenta'
color_new = 'limegreen'

pdf_name = pathlib.Path(base_path) / target / 'retrieval_outputs' / run / 'test_plots' / f'{run}_comparison.pdf'
with PdfPages(pdf_name) as pdf:
    
    for order in range(n_orders):
        fig, ax = plt.subplots(2, 1, figsize=(14, 4), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        label = 'Data' if order==0 else None
        ax[0].plot(d_wave[order], d_flux[order], label=label, alpha=0.5, color='k')
        
        label = f'Old: {chi2:.3f}' if order==0 else None
        ax[0].plot(w[order], m[order], label=label, alpha=0.75, color=color_old)
        
        label = f'New: {chi2_new:.3f}' if order==0 else None
        ax[0].plot(w_new[order], m_new[order], label=label, alpha=0.75, color=color_new)
        
        if order==0:
            ax[0].legend()
        
        ax[1].plot(w[order], d_flux[order]-m[order], label='Old', alpha=0.65, color=color_old)
        ax[1].plot(w_new[order], d_flux[order]-m_new[order], label='New', alpha=0.65, color=color_new)
        
        print(f' MAD {order} models: {np.nanmean(np.abs(m[order]-m_new[order])):.3f}')
        pdf.savefig(fig)
        plt.close(fig)
print(f'PDF saved to {pdf_name}')





