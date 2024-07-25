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

run = 'run_7'
config_file = 'config_freechem.txt'
conf = Config(path=base_path, target=target, run=run)(config_file)

ret = Retrieval(
            conf=conf, 
            evaluation=False,
            )

w_set = list(conf.config_data.keys())
assert len(w_set) == 1, 'Only one wavelength set is allowed'
w_set = w_set[0]

free_parameter = 'dlnT_dlnP_RCE'

# fixed_parameters = {'log_a': 0.0,
#  'log_l': -1.4,
#  'log_g': 3.5, 
# 'epsilon_limb': 0.54, 
# 'vsini': 5.0, 
# 'rv': 14.8,
# 'log_12CO': -4.0,
# 'log_13CO': -6.0,
#  'log_H2O': -4.0,
#  'log_HF': -8.0, 
# 'log_Na': -6.0,
#  'log_Ca': -6.0, 
# 'log_Ti': -7.0,
#  'dlnT_dlnP_0': 0.31, 
#  'dlnT_dlnP_1': 0.11, 
#  'dlnT_dlnP_2': 0.08, 
#  'dlnT_dlnP_3': 0.08, 
#  'dlnT_dlnP_4': 0.04,
# 'dlnT_dlnP_5': 0.04,
# 'dlnT_dlnP_6': 0.04,
#  'T_0': 5500.0}

figs.fig_free_parameter(ret, free_parameter, 
                        # fixed_parameters=fixed_parameters,
                        fixed_parameters={},
                        N_points=3, 
                        w_set=w_set,
                        cmap='viridis',
                        fig_name=conf.prefix+f'plots/spec_{free_parameter}_free.pdf')