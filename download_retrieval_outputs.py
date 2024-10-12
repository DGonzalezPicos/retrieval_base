import pathlib
import os
import numpy as np
import subprocess
import shutil

# targets_dict = {
#                 'gl436': 'fc',
#                 # 'gl699': 'sphinx2',
#                 }
# targets = list(targets_dict.keys())

targets_rv = {
                'gl338B': 12.0,
                'gl382' : 8.0,
                'gl408' : 3.0,
                'gl411' :-85.0,
                'gl699' : -111.0,
                'gl905' : -78.0,
                'gl1286': 8.0,
}
targets = list(targets_rv.keys())

targets_dict = dict(zip(targets, ['fc1']*len(targets)))


base_path = pathlib.Path('/home/dario/phd/retrieval_base/')

def download_run(target, run, cache=False):
    
    # download from snellius using scp -r
    snellius_dir = f'/home/dgonzalezpi/retrieval_base/{target}/retrieval_outputs/{run}/test_output'
    local_dir = str(base_path / target / f'retrieval_outputs/{run}/test_output')
    print(f' Downloading {snellius_dir} to {local_dir}...')
    
    download_ok=False
    if not cache:
        try:
            subprocess.run(f'scp -r dgonzalezpi@snellius.surf.nl:{snellius_dir} {local_dir}', shell=True, check=True)
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
        subprocess.run(f'python retrieval_script.py -e --target {target} --run {run}', shell=True, check=True, cwd=str(base_path / target))
        print(f' Succesful evaluation for {target} run {run}!\n')

    except subprocess.CalledProcessError as e:
        print(f' -> Error evaluating {target}:\n{e}')
    
    return download_ok

cache = False
try_runs = [f'fc{i}' for i in range(1,3)][::-1] # fc1, fc2

for target in targets:
    
        
    for run in try_runs:
        
        # check if test_output dir exists
        test_output = base_path / target / f'retrieval_outputs/{run}/test_output'
        if test_output.exists() and not cache:
            shutil.rmtree(test_output)
            ok = download_run(target, run)
        if ok:
            break
        
    
    
    
# run scripts to generate paper figures
try:
    subprocess.run(f'python paper/carbon_isotope_period.py', shell=True, check=True, cwd=str(base_path))
except subprocess.CalledProcessError as e:
    print(f' -> Error running paper/carbon_isotope_period.py:\n{e}')
    
print(f' Done.\n')