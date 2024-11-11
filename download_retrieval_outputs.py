import pathlib
import os
import numpy as np
import subprocess
import shutil
targets_rv = {
    # 'gl15A': 11.73,
    'gl15B': 11.17,
    'gl205': 8.5,
    # 'gl338B': 12.43,
    # 'gl382': 7.87,
    # 'gl408': 3.29,
    # 'gl411': -84.64,
    # 'gl412A': 68.84,
    # 'gl436': 9.59,
    # 'gl445': -111.51,
    # 'gl447': -30.66,
    # 'gl699': -110.11,
    # 'gl725A': -0.58,
    # 'gl725B': 1.19,
    'gl752A': 35.884,
    # 'gl687': -28.65,
    # 'gl849': -15.3,
    # 'gl876': -1.47,
    # 'gl880': -27.5,
    # 'gl905': -77.51,
    # 'gl1002': -33.7,
    # 'gl1151': -35.12,
    # 'gl1286': -41.0, # WARNING: SIMBAD has wrong RV (Davison+2015; RV = -40 km/s)
    # 'gl3622': 2.18,
    # 'gl4063': 12.533
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

def download_run(target, run, cache=False, eval=True):
    
    # download from snellius using scp -r
    snellius_dir = f'/home/dgonzalezpi/retrieval_base/{target}/retrieval_outputs/{run}/test_output'
    local_dir = str(base_path / target / f'retrieval_outputs/{run}/test_output')
    print(f' Downloading {snellius_dir} to {local_dir}...')
    
    download_ok=False
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
       
    if eval: 
        try:
            # change working directory to target
            subprocess.run(f'python retrieval_script.py -e --target {target} --run {run}', shell=True, check=True, cwd=str(base_path / target))
            print(f' Succesful evaluation for {target} run {run}!\n')

        except subprocess.CalledProcessError as e:
            print(f' -> Error evaluating {target}:\n{e}')
    
    return download_ok

cache = True
eval = False
# try_runs = [f'fc{i}' for i in [4]][::-1] # fc1, fc2
try_runs = [run]
ok = False
for target in targets:
    print(f' Downloading retrieval outputs for {target}...')
        
    for run in try_runs:
        
        # check if test_output dir exists
        test_output = base_path / target / f'retrieval_outputs/{run}/test_output'
        test_plots  = base_path / target / f'retrieval_outputs/{run}/plots'
        if test_output.exists():
            # chek it is not empty
            if len(list(test_output.iterdir())) > 0 and cache:
                print(f' {target} {run}: Found test_output folder with {len(list(test_output.iterdir()))} files.')
                ok = True
            else:
                print(f' {target} {run}: Found empty test_output folder.')
                print(f' Removing {test_output}...')
                shutil.rmtree(test_output)
                ok = download_run(target, run, cache=False, eval=eval)
        else:
            ok = download_run(target, run, cache=False, eval=eval)
            # if test_plots.exists():
            #     shutil.rmtree(test_plots)
        # if not cache:
        #     ok = download_run(target, run, cache, eval)
        if ok:
            break
        
    
    
    
# run scripts to generate paper figures
generate_figs = False
if generate_figs:
    try:
        subprocess.run(f'python paper/isotopes_metallicity.py', shell=True, check=True, cwd=str(base_path))
    except subprocess.CalledProcessError as e:
        print(f' -> Error running paper/carbon_isotope_teff.py:\n{e}')
        
    try:
        subprocess.run(f'python paper/best_fit_model.py', shell=True, check=True, cwd=str(base_path))
    except subprocess.CalledProcessError as e:
        print(f' -> Error running paper/best_fit_model.py:\n{e}')
        
        
print(f' Done.\n')