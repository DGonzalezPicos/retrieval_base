import pathlib
import os
import numpy as np
import subprocess
import shutil

targets_dict = {
                'gl436': 'fc1',
                # 'gl699': 'sphinx2',
                }
targets = list(targets_dict.keys())

base_path = pathlib.Path('/home/dario/phd/retrieval_base/')
for target in targets:
    # check if test_output dir exists
    test_output = base_path / target / f'retrieval_outputs/{targets_dict[target]}/test_output'
    if test_output.exists():
        # remove test_output dir
        # subprocess.run(f'rmdir {test_output}', shell=True, check=True)
        shutil.rmtree(test_output)
        
    # download from snellius using scp -r
    snellius_dir = f'/home/dgonzalezpi/retrieval_base/{target}/retrieval_outputs/{targets_dict[target]}/test_output'
    local_dir = str(base_path / target / f'retrieval_outputs/{targets_dict[target]}/test_output')
    
    print(f' Downloading {snellius_dir} to {local_dir}...')
    try:
        subprocess.run(f'scp -r dgonzalezpi@snellius.surf.nl:{snellius_dir} {local_dir}', shell=True, check=True)
    # catch error and print message
    except subprocess.CalledProcessError as e:
        print(f' -> Error downloading {snellius_dir} to {local_dir}:\n{e}')
        print(f' -> VPN must be disabled or set to NL!!')
        
    try:
        # change working directory to target
        subprocess.run(f'python retrieval_script.py -e', shell=True, check=True, cwd=str(base_path / target))
    except subprocess.CalledProcessError as e:
        print(f' -> Error evaluating {target}:\n{e}')
        
    print(f' Succesful download for {target}!\n')
    