import argparse
from retrieval_base.retrieval import Retrieval
import retrieval_base.figures as figs
from retrieval_base.config import Config
# import config_freechem as conf

import os

# change working directory to the location of this script
# abspath = os.path.abspath(__file__)
# dname = os.path.dirname(abspath)
# os.chdir(dname)

base_path = '/home/dario/phd/retrieval_base/'

target = 'gl880'
if target not in os.getcwd():
    os.chdir(base_path + target)

# run = 'sphinx_4'
# run = 'sphinx_8'
run = 'sphinx11'
config_file = 'config_freechem.txt'
conf = Config(path=base_path, target=target, run=run)(config_file)

ret = Retrieval(
            conf=conf, 
            evaluation=False,
            )

w_set = list(conf.config_data.keys())
assert len(w_set) == 1, 'Only one wavelength set is allowed'
w_set = w_set[0]

# free_parameter = 'C_O'
free_parameter = 'log_12CO/13CO'

run_all = True
if run_all:
    for free_param in ret.Param.param_keys:
        free_param_label = free_param.replace('/', '-')

        figs.fig_free_parameter_residuals(ret, free_param, 
                                fixed_parameters={},
                                N_points=3, 
                                w_set=w_set,
                                cmap='viridis',
                                fig_name=conf.prefix+f'plots/spec_{free_param_label}_free.pdf')
        
else:
    free_parameter_label = free_parameter.replace('/', '-')
    figs.fig_free_parameter_residuals(ret, free_parameter, 
                        # fixed_parameters=fixed_parameters,
                        fixed_parameters={},
                        N_points=3, 
                        w_set=w_set,
                        cmap='viridis',
                        fig_name=conf.prefix+f'plots/spec_{free_parameter_label}_free_residuals.pdf')
