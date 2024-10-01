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
copy_files = ['config_freechem.py', 'retrieval_script.py']
cache = False
for t, target in enumerate(targets):
    
    target_path = base_path / target
    (target_path / new_dir).mkdir(exist_ok=True)
    
    # check if copy_files in 
    for file in copy_files:
        if not (target_path / file).exists() and cache:
            # pass
            subprocess.run(f'cp {base_path / ref_target / file} {target_path / file}', shell=True)
            # copy file
        # os.system(f'cp {base_path / ref_target / file} {target_path / file}')
        # os.unlink(target_path / file)
    
    
    
    