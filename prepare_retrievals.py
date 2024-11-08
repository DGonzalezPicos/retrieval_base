import pathlib
import os
import subprocess
import shutil


path = pathlib.Path('/home/dario/phd/retrieval_base/')
# folders = [f for f in path.iterdir() if f.is_dir()]
# targets = [f.name for f in folders if str(f.name).startswith('gl')]
ignore_targets = []

# query from simbad
targets_rv = {
    'gl15A': 11.73,
    'gl15B': 11.17,
    'gl205': 8.5,
    'gl338B': 12.43,
    'gl382': 7.87,
    'gl408': 3.29,
    'gl411': -84.64,
    'gl412A': 68.8,
    'gl436': 9.59,
    'gl445': -111.51,
    'gl447': -30.66,
    'gl687': -28.65,
    'gl699': -110.11,
    'gl725A': -0.58,
    'gl725B': 1.19,
    'gl752A': 35.884,
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
# DPG 2024-10-27: Gl 447 still needs to be rerun with all species

targets = list(targets_rv.keys())
print(f' len(targets) = {len(targets)}')
delta_rv = 20.0

copy_files = ['config_freechem_template.py', 'retrieval_script_template.py']
run = 'fc5'
ignore = 'C18O'
# ignore  = '13CO'
if ignore is not None:
    print(f' Ignoring {ignore} in retrieval...')
    run = f'{run}_no{ignore}'

testing = False
cache = "True"

def update_file(file, old_str, new_str):
    
    with open(file, 'r') as f:
        filedata = f.read()
    
    filedata = filedata.replace(old_str, new_str)
    
    with open(file, 'w') as f:
        f.write(filedata)
    print(f' Updated {file} with new {new_str}!')
        

# copy this file to all targets
for target in targets:
    if target in ignore_targets:
        continue
    # copy config_template to target
    
    for file in copy_files:
        newfile = str(path / target / file.replace('_template', ''))
        shutil.copy(file, newfile)
        
        update_file(newfile, 'run = \'fc1\'', f'run = \'{run}\'')
    
    rv = targets_rv[target]
    print(f' rv = {rv}')
    update_file(str(path / target / 'config_freechem.py'), 'rv_min = -100.0', f'rv_min = {rv-delta_rv:.1f}')
    update_file(str(path / target / 'config_freechem.py'), 'rv_max = 100.0', f'rv_max= {rv+delta_rv:.1f}')
    if ignore is not None:
        update_file(str(path / target / 'config_freechem.py'), 'ignore_opacity_params = []', f'ignore_opacity_params = [\'log_{ignore}\']')
    if testing:
        break
    
    
    # run retrieval script as python retrieval_script.py -pc
    try:
        print(f' Running retrieval_script -pc for {target}...')
        command = f'python retrieval_script.py -t {target} -pc -to_snellius --cache_pRT {cache}'
        subprocess.run(command, shell=True, check=True, cwd=str(path / target))
        # # copy this folder to snellius
        # snellius_dir = f'/home/dgonzalezpi/retrieval_base/{target}/retrieval_outputs/{run}'
        # local_dir = str(path / target / 'retrieval_outputs' / run)
        # print(f' Copying {local_dir} to {snellius_dir}...')
        
        # # if parent directory does not exist, create it on remote
        # try:
        #     # subprocess.run(f'rsync -av {local_dir}/ dgonzalezpi@snellius.surf.nl:{snellius_dir}/', shell=True, check=True)

        #     subprocess.run(f'rsync -av {local_dir}/ dgonzalezpi@snellius.surf.nl:{snellius_dir}/', shell=True, check=True)

        # except subprocess.CalledProcessError as e:
        #     print(f' -> Error copying {local_dir} to {snellius_dir} with rsync:\n{e}')
        #     print(f' -> Try to create parent directory on remote...')
        #     subprocess.run(f'ssh dgonzalezpi@snellius.surf.nl "mkdir -p {snellius_dir}"', shell=True, check=True)

        #     try:
        #         subprocess.run(f'rsync -av {local_dir}/ dgonzalezpi@snellius.surf.nl:{snellius_dir}/', shell=True, check=True)
        #     except:
        #         print(f' -> Error copying {local_dir} to {snellius_dir} again...')
        #         # print(f' -> VPN must be disabled or set to NL!!')

        # print(f' Succesful copy for {target}!\n')
        
        
    # catch error and print message
    except subprocess.CalledProcessError as e:
        print(f' -> Error running {target}:\n{e}')