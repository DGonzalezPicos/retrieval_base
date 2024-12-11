from retrieval_base.retrieval import Retrieval
import retrieval_base.figures as figs
from retrieval_base.config import Config
from retrieval_base.auxiliary_functions import spirou_sample, read_spirou_sample_csv, compare_evidence, pickle_load
# import config_freechem as conf
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import pymultinest

base_path = '/home/dario/phd/retrieval_base/'
target = 'gl205'
run = 'fc5'
runs = ['fc5', 'fc5_no13CO']
config_file = 'config_freechem.txt'

bic_list = []
for run in runs:
    conf = Config(path=base_path, target=target, run=run)(config_file)
    ret = Retrieval(
                        conf=conf, 
                        evaluation=False,
                        )
    key = 'global evidence'
    log_Z = ret.PMN_stats()[key]

    bic = ret.BIC['spirou']
    bic_list.append(bic)
    print(f'{run}: BIC = {bic}')
    
print(f'BIC difference: {bic_list[0] - bic_list[1]}')


# conf_output = '/'.join(conf.prefix.split('/')[:-1])+'/test_output/'+conf.prefix.split('/')[-1]
# analyzer = pymultinest.Analyzer(
#             n_params=len(conf.free_params), 
#             outputfiles_basename=conf_output,
#             )
# # stats = analyzer.get_stats()
# bestfit = analyzer.get_best_fit()
# log_likelihood = bestfit['log_likelihood']

# data = analyzer.get_data()
# # bayesian information criterion
# n_params = len(conf.free_params)
# n_data = len(conf.y)
