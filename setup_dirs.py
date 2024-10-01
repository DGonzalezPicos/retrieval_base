import pathlib
import os
import numpy as np
import subprocess

targets = [
            # 'gj338B', 
        #    'gl687', 
           'gl699', 'gl752A', 'gl832', 'gl849', 'gl876', 'gl1151']
base_path = pathlib.Path('/home/dario/phd/retrieval_base/')

new_dir = 'retrieval_outputs'

ref_target = 'gl687'
copy_files = ['config_freechem.py', 'retrieval_script.py', 'genoa.sh']
cache = False
for t, target in enumerate(targets):
    
    target_path = base_path / target
    (target_path / new_dir).mkdir(exist_ok=True)
    
    # check if copy_files in 
    for file in copy_files:
        if not (target_path / file).exists() or not cache:
            # overwrite files
            subprocess.run(f'cp -f {base_path / ref_target / file} {target_path / file}', shell=True, check=True)
            print(f' Copied {file} to {target_path}')
    
    
    