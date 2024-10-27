import pathlib
import os
import numpy as np
import subprocess
import shutil
targets_rv = {
    'gl338B': 12.43,
    'gl382': 7.87,
    'gl408': 3.29,
    'gl411': -84.64,
    'gl436': 9.59,
    'gl699': -110.11,
    'gl752A': 35.884,
    'gl832': 13.164,
    'gl905': -77.51,
    'gl1286': -10.384,
    'gl15A': 11.73,
    'gl15B': 11.17,
    'gl687': -28.65,
    'gl725A': -0.58,
    'gl725B': 1.19,
    'gl849': -15.3,
    'gl876': -1.47,
    'gl880': -27.5,
    'gl1151': -35.12,
    'gl205': 8.56,
    'gl412A': 68.84,
    'gl445': -111.51,
    'gl447': -30.66,
    'gl1002': -33.7,
    'gl3622': 2.18,
    'gl4063': 12.533
    }
targets = list(targets_rv.keys())
run = 'fc4_no_C18O'
targets_dict = dict(zip(targets, [run]*len(targets)))


base_path = pathlib.Path('/home/dario/phd/retrieval_base/')

def download_run(target, run, cache=False):
    
    # download from snellius using scp -r
    snellius_dir = f'/home/dgonzalezpi/retrieval_base/{target}/retrieval_outputs/{run}/test_output'
    local_dir = str(base_path / target / f'retrieval_outputs/{run}/test_output')
    print(f' Downloading {snellius_dir} to {local_dir}...')
    
    download_ok=False
    if not cache:
        try:
            # subprocess.run(f'scp -r dgonzalezpi@snellius.surf.nl:{snellius_dir} {local_dir}', shell=True, check=True)
            if os.path.exists(local_dir):
                shutil.rmtree(local_dir)
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
        subprocess.run(f'python retrieval_script.py -e --target {target} --run {run}', shell=True, check=True, cwd=str(base_path / target))
        print(f' Succesful evaluation for {target} run {run}!\n')

    except subprocess.CalledProcessError as e:
        print(f' -> Error evaluating {target}:\n{e}')
    
    return download_ok

cache = False
# try_runs = [f'fc{i}' for i in [4]][::-1] # fc1, fc2
try_runs = [run]
ok = False
for target in targets:
    print(f' Downloading retrieval outputs for {target}...')
        
    for run in try_runs:
        
        # check if test_output dir exists
        test_output = base_path / target / f'retrieval_outputs/{run}/test_output'
        test_plots  = base_path / target / f'retrieval_outputs/{run}/plots'
        if test_output.exists() and not cache:
            print(f' Removing {test_output}...')
            shutil.rmtree(test_output)
            # if test_plots.exists():
            #     shutil.rmtree(test_plots)
        if not cache:
            ok = download_run(target, run)
        if ok:
            break
        
    
    
    
# run scripts to generate paper figures
generate_figs = False
if generate_figs:
    try:
        subprocess.run(f'python paper/carbon_isotope_teff.py', shell=True, check=True, cwd=str(base_path))
    except subprocess.CalledProcessError as e:
        print(f' -> Error running paper/carbon_isotope_teff.py:\n{e}')
        
    try:
        subprocess.run(f'python paper/best_fit_model.py', shell=True, check=True, cwd=str(base_path))
    except subprocess.CalledProcessError as e:
        print(f' -> Error running paper/best_fit_model.py:\n{e}')
        
        
print(f' Done.\n')