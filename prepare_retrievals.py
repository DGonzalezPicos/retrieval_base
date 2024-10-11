import pathlib
import os
import subprocess
import shutil


path = pathlib.Path('/home/dario/phd/retrieval_base/')
# folders = [f for f in path.iterdir() if f.is_dir()]
# targets = [f.name for f in folders if str(f.name).startswith('gl')]
ignore_targets = ['gl436']

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
delta_rv = 20.0

copy_files = ['config_freechem_template.py', 'retrieval_script_template.py']
run = 'fc2'
testing = False

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
        shutil.copy(file, str(path / target / file.replace('_template', '')))
        
    update_file(str(path / target / 'config_freechem.py'), 'run = \'fc1\'', f'run = \'{run}\'')
    
    rv = targets_rv[target]
    print(f' rv = {rv}')
    update_file(str(path / target / 'config_freechem.py'), 'rv_min = -100.0', f'rv_min = {rv-delta_rv:.1f}')
    update_file(str(path / target / 'config_freechem.py'), 'rv_max = 100.0', f'rv_max= {rv+delta_rv:.1f}')
        
    if testing:
        break
    
    
    # run retrieval script as python retrieval_script.py -pc
    try:
        print(f' Running retrieval_script -pc for {target}...')
        subprocess.run(f'python retrieval_script.py -pc', shell=True, check=True, cwd=str(path / target))
        # copy this folder to snellius
        snellius_dir = f'/home/dgonzalezpi/retrieval_base/{target}/retrieval_outputs/'
        local_dir = str(path / target / 'retrieval_outputs' / run)
        print(f' Copying {local_dir} to {snellius_dir}...')
        subprocess.run(f'scp -r {local_dir} dgonzalezpi@snellius.surf.nl:{snellius_dir}', shell=True, check=True)
        print(f' Succesful copy for {target}!\n')
        
        
    # catch error and print message
    except subprocess.CalledProcessError as e:
        print(f' -> Error evaluating {target}:\n{e}')