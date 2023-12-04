from retrieval_base.retrieval import pre_processing, Retrieval
from retrieval_base.parameters import Parameters
# import config_eqchem as conf
import config_freechem as conf

import numpy as np
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

if 'dgonzalezpi' in os.getcwd():
    print('Running on Snellius.. disabling interactive plotting')
    import matplotlib
    # disable interactive plotting
    matplotlib.use('Agg')
    
run_pre_processing = True
if run_pre_processing:
    for conf_data_i in conf.config_data.values():
        pre_processing(conf=conf, conf_data=conf_data_i)

run_prior_predictive_check = True

# if run_prior_predictive_check:
    
    
# change working directory to the path of the python script

# generate model for each sample
# ret = Retrieval(conf=conf, evaluation=False)
# ret.Param(0.5 * np.ones(len(ret.Param.param_keys)))
# ln_L = ret.PMN_lnL_func()
def prior_check():
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gs

    fig = plt.figure(figsize=(16,8), layout='constrained')
    gs0 = fig.add_gridspec(4,5, hspace=0.00, wspace=0.1)

    ax = fig.add_subplot(gs0[:3,:3])
    plt.setp(ax.get_xticklabels(), visible=False)
    ax_res = fig.add_subplot(gs0[3,:3], sharex=ax)
    ax_PT = fig.add_subplot(gs0[:4,3:])
    
    ret = Retrieval(conf=conf, evaluation=False)
    order, det = 2,2

    for i in [0.0, 0.5, 1.0]:
        ret.Param(i * np.ones(len(ret.Param.param_keys)))

        
        sample = {k:ret.Param.params[k] for k in ret.Param.param_keys}
        print(sample)
        
        ln_L = ret.PMN_lnL_func()
        print(f'ln_L = {ln_L:.4e}')
        print('\n')
        
        x = ret.d_spec['K2166'].wave[order,det]
        
        f = ret.LogLike['K2166'].f[order,det]
        if i == 0.0:
            mask = ret.d_spec['K2166'].mask_isfinite[order,det]
            ax.plot(x, ret.d_spec['K2166'].flux[order,det], lw=1.5, label='data', color='k')
            
        model = f * ret.m_spec['K2166'].flux[order,det]
        ax.plot(x, model, lw=2.5, label=f'logL = {ln_L:.3e}', ls='--')

        res = ret.d_spec['K2166'].flux[order,det] - model
        res[~mask] = np.nan
        ax_res.plot(x, res, lw=2.5)

        ax_PT.plot(ret.PT.temperature, ret.PT.pressure, lw=4.5)
        
    ax_PT.set(yscale='log', ylim=(ret.PT.pressure.max(), ret.PT.pressure.min()))
    ax_res.axhline(0, color='k', ls='-', alpha=0.9) 
    ax.legend()
    plt.show()
    return ret
     
     
if run_prior_predictive_check:   
    ret = prior_check()



run_PMN = False
if run_PMN:
    ret = Retrieval(
                conf=conf, 
                evaluation=False
                )
    ret.PMN_run()