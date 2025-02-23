"""Plot grid of CCFs for both targets"""

import numpy as np
import matplotlib.pyplot as plt

import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# pdf pages

from matplotlib.backends.backend_pdf import PdfPages
import copy
import pathlib

import subprocess as sp
from retrieval_base.retrieval import Retrieval
import retrieval_base.auxiliary_functions as af
from retrieval_base.config import Config

path = af.get_path(return_pathlib=True)
path_figures = pathlib.Path('/home/dario/phd/twa2x_paper/figures')
config_file = 'config_jwst.txt'
target = 'TWA28'
# run = None
w_set='NIRSpec'

runs = dict(
    TWA27A='lbl11_G1G2G3_fastchem_GP_0',
    TWA28='lbl11_G1G2G3_fastchem_GP_0',
            )

def get_ccf(target, run):
    cwd = os.getcwd()
    if target not in cwd:
        nwd = os.path.join(cwd, target)
        print(f'Changing directory to {nwd}')
        os.chdir(nwd)


    conf = Config(path=path, target=target, run=run)(config_file)        
    ccf_path = pathlib.Path(conf.prefix + 'plots/CCF/')
    ccf_files = list(ccf_path.glob('RV_CCF_ACF_*.txt'))
    
    if len(ccf_files) == 0:
        print(f' ** Running cross-correlation function for {target} {run}..')
        command = f'python {path}/retrieval_base/cross_correlation.py -t {target} -r {run}'
        sp.call(command, shell=True)
    
        ccf_files = list(ccf_path.glob('RV_CCF_ACF_*.txt'))
    
    ccf_files = [f.stem for f in ccf_files]
    print(ccf_files)
    return ccf_files

ccf_files = get_ccf(target, runs[target])





