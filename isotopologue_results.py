""" Read MAP, q50 and 1sigma confidence interval of 13CO, C18O, etc for all targets and save as txt file. """
from retrieval_base.retrieval import Retrieval
import retrieval_base.figures as figs
from retrieval_base.config import Config
from retrieval_base.auxiliary_functions import spirou_sample, read_spirou_sample_csv, compare_evidence, pickle_load
# import config_freechem as conf
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib

def main(target, run='fc5', params=['log_12CO/13CO'], cache=True):
    
    if target not in os.getcwd():
        os.chdir(base_path + target)

    outputs = pathlib.Path(base_path) / target / 'retrieval_outputs'
    test_output = outputs / run / 'test_output'
    # test_data = outputs / run / 'test_data'
    assert test_output.exists(), f'No test_output folder found in {test_output}'
    
    conf = Config(path=base_path, target=target, run=run)(config_file)
    ret = Retrieval(
                    conf=conf, 
                    evaluation=False,
                    )
    stats = ret.PMN_stats()
    
    # find indices of free parameters
    
    return stats
    
    
    
    

base_path = '/home/dario/phd/retrieval_base/'
config_file = 'config_freechem.txt'


df = read_spirou_sample_csv()
names = df['Star'].to_list()
ignore_targets = ['gl3622']

testing = False
run =   'fc5'
params = ['log_12CO/13CO', 'log_12CO/C18O', 'log_H2O/H2O_181', 'log_12CO/C17O']

results = {}
for i, name in enumerate(names):
    target = name.replace('Gl ', 'gl')
    if target in ignore_targets:
        print(f'---> Skipping {target}...')
        continue
    
    print(f'Target = {target}')
    
    # stats = main(target, run='fc5', species=['13CO'], cache=True)
    if target not in os.getcwd():
        os.chdir(base_path + target)

    outputs = pathlib.Path(base_path) / target / 'retrieval_outputs'
    test_output = outputs / run / 'test_output'
    # test_data = outputs / run / 'test_data'
    assert test_output.exists(), f'No test_output folder found in {test_output}'
    
    conf = Config(path=base_path, target=target, run=run)(config_file)
    all_params = list(conf.free_params.keys())
    ret = Retrieval(
                    conf=conf, 
                    evaluation=False,
                    )
    stats = ret.PMN_stats()
    
    # find indices of free parameters
    
    results[target] = {}
    
    for param in params:
        ip = all_params.index(param)
        print(f' {param} = {ip}')
        marginals = stats['marginals'][ip]
        median, one_sigma = marginals['median'], marginals['1sigma']
        
        maxp = stats['modes'][0]['maximum a posterior'][ip]
        print(f'** {param} **')
        print(f' - MAP = {maxp:.2f}')
        print(f' - Median = {median:.2f} +{one_sigma[1]-median:.2f} -{median-one_sigma[0]:.2f}')
        
        results[target][param] = {'MAP': maxp, 'median': median, '1sigma': one_sigma}
    
    if testing:
        break
    
    
# save dictionary results
import json
save_path = pathlib.Path(base_path) / 'paper/data/isotopologue_results.json'
with open(save_path, 'w') as f:
    json.dump(results, f)
print(f' Saved results to {save_path}')
    