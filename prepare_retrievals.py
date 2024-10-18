import pathlib
import os
import subprocess
import shutil


path = pathlib.Path('/home/dario/phd/retrieval_base/')
# folders = [f for f in path.iterdir() if f.is_dir()]
# targets = [f.name for f in folders if str(f.name).startswith('gl')]
ignore_targets = []

targets_rv = {
                # 'gl338B': 12.0,
                # 'gl382' : 8.0,
                # 'gl408' : 3.0,
                # 'gl411' :-85.0,
                # 'gl436' : -40.0,
                # 'gl699' : -111.0,
                # 'gl752A': 36.0,
                # 'gl832': 36.0,
                # 'gl905' : -78.0,
                # 'gl1286': 8.0,
                # 'gl15A': 12.0,
                # 'gl15B': 11.0,
                # 'gl687': -29.0,
                # 'gl725A': -31.0,
                # 'gl725B': 1.0,
                # 'gl849': -15.0,
                # 'gl876': -2.0,
                # 'gl880': -27.0,
                # 'gl1151': -35.0,
                # 'gl205': -40.0,
                # 'gl412A': 9.0,
                # 'gl445': 9.0,
                # 'gl447': -112.0,
                'gl1002': -40.0,
                'gl412A': 69.0,
                'gl1286': -41.0,
                'gl3622': 2.0,
                'gl4063': 12.0,
                
}
targets = list(targets_rv.keys())
delta_rv = 20.0

copy_files = ['config_freechem_template.py', 'retrieval_script_template.py']
run = 'fc4'
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
    if testing:
        break
    
    
    # run retrieval script as python retrieval_script.py -pc
    try:
        print(f' Running retrieval_script -pc for {target}...')
        subprocess.run(f'python retrieval_script.py -pc --cache_pRT {cache}', shell=True, check=True, cwd=str(path / target))
        # copy this folder to snellius
        snellius_dir = f'/home/dgonzalezpi/retrieval_base/{target}/retrieval_outputs/{run}'
        local_dir = str(path / target / 'retrieval_outputs' / run)
        print(f' Copying {local_dir} to {snellius_dir}...')
        
        # if parent directory does not exist, create it on remote
        try:
            # subprocess.run(f'rsync -av {local_dir}/ dgonzalezpi@snellius.surf.nl:{snellius_dir}/', shell=True, check=True)

            subprocess.run(f'rsync -av {local_dir}/ dgonzalezpi@snellius.surf.nl:{snellius_dir}/', shell=True, check=True)

        except subprocess.CalledProcessError as e:
            print(f' -> Error copying {local_dir} to {snellius_dir} with rsync:\n{e}')
            print(f' -> Try to create parent directory on remote...')
            subprocess.run(f'ssh dgonzalezpi@snellius.surf.nl "mkdir -p {snellius_dir}"', shell=True, check=True)

            try:
                subprocess.run(f'rsync -av {local_dir}/ dgonzalezpi@snellius.surf.nl:{snellius_dir}/', shell=True, check=True)
            except:
                print(f' -> Error copying {local_dir} to {snellius_dir} again...')
                # print(f' -> VPN must be disabled or set to NL!!')

        print(f' Succesful copy for {target}!\n')
        
        
    # catch error and print message
    except subprocess.CalledProcessError as e:
        print(f' -> Error evaluating {target}:\n{e}')