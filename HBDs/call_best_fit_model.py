from retrieval_base.retrieval import pre_processing, Retrieval
from retrieval_base.parameters import Parameters
from retrieval_base.config import Config
import numpy as np
import matplotlib.pyplot as plt
# set fontsize to 16
# plt.rcParams.update({'font.size': 16})
plt.style.use('/home/dario/phd/retrieval_base/HBDs/my_science.mplstyle')

import pathlib
import pickle
import corner
import pandas as pd
import json
save_transparent_to = pathlib.Path('/home/dario/phd/presentations/october24/')

path = pathlib.Path('/home/dario/phd/retrieval_base')
# out_path = path / 'HBDs'
out_path = pathlib.Path('/home/dario/phd/Hot_Brown_Dwarfs_Retrievals/figures/')

# targets = dict(J1200='freechem_16', 
#                TWA28='freechem_13', 
#                J0856='freechem_14'
#                )
# targets = dict(J1200='freechem_15', 
#                TWA28='freechem_12', 
#                J0856='freechem_13'
#                )
targets = dict(
                # J1200='final_full',
                TWA28='final_full',
                # J0856='final_full',
                )
colors = dict(J1200='royalblue', TWA28='seagreen', J0856='indianred')

fig, ax = plt.subplots(1,1, figsize=(6,6), constrained_layout=True)

for i, (target, retrieval_id) in enumerate(targets.items()):
    data_path = pathlib.Path('/home/dario/phd/retrieval_base') / f'{target}'
    print(data_path)
    
    # bestfit_params = 
    conf = Config(path=path, target=target, run=retrieval_id)
    conf('config_freechem.txt')
    
    
    retrieval_path = data_path / f'retrieval_outputs/{retrieval_id}'
    assert retrieval_path.exists(), f'Retrieval path {retrieval_path} does not exist.'
    ret = Retrieval(conf, evaluation=False)