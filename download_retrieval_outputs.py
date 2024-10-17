import pathlib
import os
import numpy as np
import subprocess
import shutil

def download_run(target, run, cache=False):
    
    # download from snellius using scp -r
    snellius_dir = f'/home/dgonzalezpi/retrieval_base/{target}/retrieval_outputs/{run}/test_output'
    local_dir = str(base_path / target / f'retrieval_outputs/{run}/test_output')
    print(f' Downloading {snellius_dir} to {local_dir}...')
    
    download_ok=False
    if not cache:
        try:
            # subprocess.run(f'scp -r dgonzalezpi@snellius.surf.nl:{snellius_dir} {local_dir}', shell=True, check=True)
            subprocess.run(f'rsync -av --progress dgonzalezpi@snellius.surf.nl:{snellius_dir}/ {local_dir}/', shell=True, check=True)
            print(f' Succesful download for {target} {run}!\n')
            download_ok = True
        # catch error and print message
        except subprocess.CalledProcessError as e:
            print(f' -> Error downloading {snellius_dir} to {local_dir}:\n{e}')
            print(f' -> VPN must be disabled or set to NL!!')
    else:
        print(f' Using cached retrieval outputs for {target} {run}!\n')
        download_ok = True
        
    try:
        # change working directory to target
        subprocess.run(f'python retrieval_script_jwst.py -e', shell=True, check=True, cwd=str(base_path / target))
        print(f' Succesful evaluation for {target} run {run}!\n')

    except subprocess.CalledProcessError as e:
        print(f' -> Error evaluating {target}:\n{e}')
    
    return download_ok


base_path = pathlib.Path('/home/dario/phd/retrieval_base/')
target = 'TWA28'
# run = 'lbl15_K2'
run = 'lbl15_G2G3'
cache = False

# check if test_output dir exists
test_output = base_path / target / f'retrieval_outputs/{run}/test_output'
if test_output.exists() and not cache:
    shutil.rmtree(test_output)
    download_run(target, run, cache)

    
print(f' Done.\n')