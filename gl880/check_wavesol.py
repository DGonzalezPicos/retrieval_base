from retrieval_base.retrieval import pre_processing_spirou, prior_check, Retrieval
from retrieval_base.config import Config
from retrieval_base.auxiliary_functions import get_path
path = get_path()

import config_freechem as conf

import os

config_file = 'config_freechem.txt'
target = 'gl880'
cwd = os.getcwd()
if target not in cwd:
    os.chdir(f'{path}/{target}')
    
    
run = 'run_3' # important to set this to the correct run

conf = Config(path=path, target=target, run=run)
conf(config_file)

ret = Retrieval(
    conf=conf, 
    evaluation=True,
    # plot_ccf=args.ccf
    )

bestfit_params, posterior = ret.PMN_analyze()
ret.evaluate_model(bestfit_params)
ret.PMN_lnL_func()

# for order in range(ret.m_spec['spirou'].n_orders):
    