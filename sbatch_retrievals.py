import pathlib
import os
import subprocess
import shutil


path = pathlib.Path('/home/dgonzalezpi/retrieval_base/')
folders = [f for f in path.iterdir() if f.is_dir()]
targets = [f.name for f in folders if str(f.name).startswith('gl')]
ignore_targets = ['gl436']

# copy genoa.sh file to all targets

genoa_template = 'genoa_template.sh'
genoa_file = 'genoa.sh'
for target in targets:
    if target in ignore_targets:
        continue
    # copy genoa_template to target
    shutil.copy(genoa_template, str(path / target / genoa_file))
    # enable permissions
    subprocess.run(f"chmod +x {genoa_file}", shell=True, check=True, cwd=str(path / target))

    # schedule job with sbatch
    try:
        print(f' Running genoa for {target}...')
        subprocess.run(f'sbatch {genoa_file}', shell=True, check=True, cwd=str(path / target))
    except subprocess.CalledProcessError as e:
        print(f' -> Error running genoa for {target}:\n{e}')
        
print(f' Succesful scheduling for {len(targets)} targets!\n')
