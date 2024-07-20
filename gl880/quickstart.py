import os
cwd = os.getcwd()
if 'dgonzalezpi' in cwd:
    path = '/home/dgonzalezpi/retrieval_base/'
if 'dario' in cwd:
    path = '/home/dario/phd/retrieval_base/'
    
from retrieval_base.retrieval import pre_processing_spirou, Retrieval
from retrieval_base.config import Config
import config_freechem as conf



config_file = 'config_freechem.txt'
target = 'gl880'
run = 'run_1' # important to set this to the correct run


for conf_data_i in conf.config_data.values():
        pre_processing_spirou(conf=conf, conf_data=conf_data_i)