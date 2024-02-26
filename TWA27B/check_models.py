import numpy as np
import matplotlib.pyplot as plt
import pathlib

from retrieval_base.retrieval import pre_processing, prior_check, Retrieval
from retrieval_base.spectrum_jwst import SpectrumJWST
from retrieval_base.pRT_model import pRT_model
import retrieval_base.auxiliary_functions as af
from retrieval_base.parameters import Parameters

import config_jwst as conf

target = 'TWA27b'
path = pathlib.Path('/home/dario/phd/retrieval_base/') / target.upper()

ret = Retrieval(conf=conf, evaluation=False)
# disable spline decomposition
ret.Param.params.update({'N_knots': 1})
 
ret.Param(np.random.rand(ret.Param.n_params))
sample = {k:ret.Param.params[k] for k in ret.Param.param_keys}
my_sample = {'log_a': -1.579, 'log_l': -0.7581, 'log_g': 3.9486, 
             'rv': 3.965, 
             'log_12CO': -3.8, 'log_13CO': -5.5, 'log_C18O': -6.6, 
             'log_C17O': -7.3, 'log_H2O': -4.0, 'log_CO2': -8.0, 
             'dlnT_dlnP_0': 0.22568, 'dlnT_dlnP_1': 0.1253, 
             'dlnT_dlnP_2': 0.08, 'dlnT_dlnP_3': 0.08, 
             'dlnT_dlnP_4': 0.09, 'T_0': 4601.,
             'res': 3306}

ret.Param.params.update(my_sample)

free_parameter = 'log_C17O'
assert free_parameter in ret.Param.param_keys, f'{free_parameter} not in {ret.Param.param_keys}'
is_PT_param = free_parameter if free_parameter.startswith('dlnT') or free_parameter.startswith('T_') else False
N = 4
bounds = ret.Param.param_priors[free_parameter]
values = np.linspace(*bounds, N)


order, det = 0, 0
# print(f'ln_L = {ln_L:.4e}\n')
x = ret.d_spec['G395H_F290LP'].wave[order,det]
fig, (ax_spec, ax_PT) = plt.subplots(1,2, 
                       figsize=(14,5),
                       gridspec_kw={'width_ratios': [3, 1],
                                    'wspace': 0.15,
                                    'left': 0.05,
                                    'right': 0.95,
                                    'top': 0.92,
                                    'bottom': 0.1,
                                    })

cmap = 'viridis'
colors = getattr(plt.cm, cmap)(np.linspace(0, 1, N))
im = ax_spec.scatter([], [], c=[], cmap=cmap, vmin=values.min(), vmax=values.max())
cbar = plt.colorbar(im, ax=ax_spec)
cbar.set_label(f'{free_parameter}')
cbar.set_ticks(values)
cbar.set_ticklabels([f'{v:.2f}' for v in values])


for i in range(N):
    new_sample = my_sample.copy()  
    new_sample[free_parameter] = values[i]
    

    ret.Param.params.update(new_sample)
    for k in ret.Param.param_keys:
        if k not in new_sample:
            ret.Param.params[k] = sample[k]
            
        if k.startswith('log_'):
            ret.Param.params = ret.Param.log_to_linear(ret.Param.params, k)
            
    ret.Param.read_PT_params()
    ret.Param.read_uncertainty_params()
    ret.Param.read_chemistry_params()
    # sample_i = {k:ret.Param.params[k] for k in ret.Param.param_keys}
    print({k:ret.Param.params[k] for k in ret.Param.param_keys})
    ln_L = ret.PMN_lnL_func()
    print(f'f = {ret.LogLike["G395H_F290LP"].f}')

    model = ret.LogLike['G395H_F290LP'].m_flux[order,det]
    ax_spec.plot(x, model, label=f'logL = {ln_L:.3e}', ls='-', lw=1., color=colors[i])
    # ax_spec.plot(x, ret.d_spec['G395H_F290LP'].flux[order,det], label='Data')
    if i > 0 and not is_PT_param:
        continue
    ax_PT.plot(ret.PT.temperature, ret.PT.pressure, lw=3.5, alpha=0.9, color=colors[i])
            
            
# show sample above plot as one row extending horizontally
text = ' '.join([f'{k} = {v:.2f}' for k,v in my_sample.items() if k != free_parameter])
ax_spec.text(0.00, 1.07, text, transform=ax_spec.transAxes, fontsize=7,
                verticalalignment='top')

ax_PT.set(yscale='log', ylim=(ret.PT.pressure.max(), ret.PT.pressure.min()),
        ylabel='Pressure [bar]', xlabel='Temperature [K]')
ax_spec.set(ylabel=f"Flux")
ax_spec.legend()


fig.savefig(path / f'{conf.prefix}plots/compare_model.pdf')
print(f'--> Saved {path / f"{conf.prefix}plots/compare_model.pdf"}')
# plt.show()
plt.close(fig)