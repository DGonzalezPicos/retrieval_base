import pathlib
import os
import subprocess
import shutil

testing = True
user = 'dario/phd' if testing else 'dgonzalezpi'
path = pathlib.Path(f'/home/{user}/retrieval_base/')
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
    
    # schedule job with sbatch
    try:
        print(f' Running genoa for {target}...')
        subprocess.run(f'sbatch {genoa_file}', shell=True, check=True, cwd=str(path / target))
    except subprocess.CalledProcessError as e:
        print(f' -> Error running genoa for {target}:\n{e}')
        
print(f' Succesful scheduling for {len(targets)} targets!\n')