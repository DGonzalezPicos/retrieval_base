import pathlib
import numpy as np
import os
import matplotlib.pyplot as plt
# pdf pages
from matplotlib.backends.backend_pdf import PdfPages
import copy

from retrieval_base.retrieval import Retrieval
import retrieval_base.auxiliary_functions as af
from retrieval_base.config import Config

from retrieval_base.figures import fig_corner_VMRs_posterior
# import config_jwst as conf

path = af.get_path()
config_file = 'config_jwst.txt'
target = 'TWA27A'
# run = None
run = 'lbl15_KM_5'
w_set='NIRSpec'

cwd = os.getcwd()
if target not in cwd:
    nwd = os.path.join(cwd, target)
    print(f'Changing directory to {nwd}')
    os.chdir(nwd)


conf = Config(path=path, target=target, run=run)(config_file)        
    
ret = Retrieval(
    conf=conf, 
    evaluation=False
    )

bestfit_params, posterior = ret.PMN_analyze()
ret.get_PT_mf_envelopes(posterior)
ret.Chem.get_VMRs_posterior()

fig_corner_VMRs_posterior(ret.Chem, fig_name=conf.prefix+'plots/VMRs_posterior.pdf')
# bestfit_params_dict = dict(zip(ret.Param.param_keys, bestfit_params))

# ret.evaluate_model(bestfit_params)
# ret.PMN_lnL_func()


# if len(getattr(ret, 'VMRs_posterior', {}))==0:
#     print(f' - Computing VMRs posterior')
#     ret.Chem.get_VMRs_posterior()

