import pathlib
import os
import numpy as np
import subprocess
import shutil
targets_rv = {
    'gl15A': 11.73,
    'gl15B': 11.17,
    'gl205': 8.5,
    'gl338B': 12.43,
    'gl382': 7.87,
    'gl408': 3.29,
    'gl411': -84.64,
    'gl412A': 68.84,
    'gl436': 9.59,
    'gl445': -111.51,
    'gl447': -30.66,
    'gl699': -110.11,
    'gl725A': -0.58,
    'gl725B': 1.19,
    'gl752A': 35.884,
    'gl687': -28.65,
    'gl849': -15.3,
    'gl876': -1.47,
    'gl880': -27.5,
    'gl905': -77.51,
    'gl1002': -33.7,
    'gl1151': -35.12,
    'gl1286': -41.0, # WARNING: SIMBAD has wrong RV (Davison+2015; RV = -40 km/s)
    'gl3622': 2.18,
    'gl4063': 12.533
    }
targets = list(targets_rv.keys())
# run = 'fc4_wo_C18O'
run = 'fc5'
ignore = 'C18O'
# ignore  = '13CO'
if ignore is not None:
    print(f' Ignoring {ignore} in retrieval...')
    run = f'{run}_no{ignore}'
targets_dict = dict(zip(targets, [run]*len(targets)))


base_path = pathlib.Path('/home/dario/phd/retrieval_base/')

for target in targets:
    subprocess.run(f'python retrieval_script.py -e --target {target} --run {run}', shell=True, check=True, cwd=str(base_path / target))
    print(f' Succesful evaluation for {target} run {run}!\n')