import pathlib
import os
import subprocess
import shutil

testing = False
user = 'dario/phd' if testing else 'dgonzalezpi'
path = pathlib.Path(f'/home/{user}/retrieval_base/')
# folders = [f for f in path.iterdir() if f.is_dir()]
# targets = [f.name for f in folders if str(f.name).startswith('gl')]
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
targets = list(targets_rv.keys())
print(f' len(targets) = {len(targets)}')
# run = 'fc4_no_C18O'
run = 'fc5'

ignore_targets = ['gl15A']

def update_file(file, old_str, new_str):
    
    with open(file, 'r') as f:
        filedata = f.read()
    
    filedata = filedata.replace(old_str, new_str)
    
    with open(file, 'w') as f:
        f.write(filedata)
    print(f' Updated {file} with new {new_str}!')
# copy genoa.sh file to all targets

genoa_template = 'genoa_template.sh'
genoa_file = 'genoa.sh'
for target in targets:
    if target in ignore_targets:
        continue
    # copy genoa_template to target
    shutil.copy(genoa_template, str(path / target / genoa_file))
    # enable permissions
    subprocess.run(f"chmod +x {genoa_file}", shell=True, check=True, cwd=str(path / target)) # is this necessary?
    update_file(str(path / target / genoa_file), 'target=gl436', f'target={target}')
    # update_file(str(path / target / genoa_file), '#SBATCH --job-name=fc2', f'#SBATCH --job-name={run}')
    update_file(str(path / target / genoa_file), '#SBATCH --job-name=fc2', f'#SBATCH --job-name={target}_{run}')
    
    # schedule job with sbatch
    try:
        print(f' Running genoa for {target}...')
        subprocess.run(f'sbatch {genoa_file}', shell=True, check=True, cwd=str(path / target))
    except subprocess.CalledProcessError as e:
        print(f' -> Error running genoa for {target}:\n{e}')
        
print(f' Succesful scheduling for {len(targets)} targets!\n')