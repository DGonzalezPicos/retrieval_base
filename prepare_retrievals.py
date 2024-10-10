import pathlib
import os
import subprocess
import shutil


path = pathlib.Path('/home/dario/phd/retrieval_base/')
folders = [f for f in path.iterdir() if f.is_dir()]
targets = [f.name for f in folders if str(f.name).startswith('gl')]
ignore_targets = ['gl436']


copy_files = ['config_freechem_template.py', 'retrieval_script_template.py']
run = 'fc1'

# copy this file to all targets
for target in targets:
    if target in ignore_targets:
        continue
    # copy config_template to target
    for file in copy_files:
        shutil.copy(file, str(path / target / file.replace('_template', '')))
    
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