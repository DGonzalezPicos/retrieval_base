from retrieval_base.retrieval import Retrieval
import retrieval_base.figures as figs
from retrieval_base.config import Config
from retrieval_base.auxiliary_functions import spirou_sample, read_spirou_sample_csv, compare_evidence
# import config_freechem as conf
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib

base_path = '/home/dario/phd/retrieval_base/'


df = read_spirou_sample_csv()
names = df['Star'].to_list()
teff =  dict(zip(names, [float(t.split('+-')[0]) for t in df['Teff (K)'].to_list()]))
valid = dict(zip(names, df['Valid'].to_list()))

ignore_targets = [name.replace('Gl ', 'gl') for name in names if valid[name] == 0]
ignore_more_targets = ['gl3622']
ignore_targets += ignore_more_targets

def get_evidence(target, run=None, key='global evidence'):
    
    config_file = 'config_freechem.txt'
    conf = Config(path=base_path, target=target, run=run)(config_file)

    try:
        ret = Retrieval(
                    conf=conf, 
                    evaluation=False,
                    )

        log_Z = ret.PMN_stats()[key]
    except Exception as e:
        print(f'Error for {target} {run}: {e}')
        log_Z = None
        
    return log_Z

def main(target, run=None, species='C18O', key='global evidence', cache=True):
    
    if target not in os.getcwd():
        os.chdir(base_path + target)

    outputs = pathlib.Path(base_path) / target / 'retrieval_outputs'
    # find dirs in outputs
    # print(f' outputs = {outputs}')
    dirs = [d for d in outputs.iterdir() if d.is_dir() and 'fc' in d.name and '_' not in d.name]
    print(f' dirs = {dirs}')
    runs = [int(d.name.split('fc')[-1]) for d in dirs]
    print(f' runs = {runs}')
    print(f' {target}: Found {len(runs)} runs: {runs}')
    assert len(runs) > 0, f'No runs found in {outputs}'
    if run is None:
        run = 'fc'+str(max(runs))
    else:
        run = 'fc'+str(run)
        assert run in [d.name for d in dirs], f'Run {run} not found in {dirs}'
        
        
    # print('Run:', run)
    # check that the folder 'test_output' is not empty
    test_output = outputs / run / 'test_output'
    assert test_output.exists(), f'No test_output folder found in {test_output}'
    if len(list(test_output.iterdir())) == 0:
        print(f' {target}: No files found in {test_output}')
        return None
    
    # run_wo_species = sorted([d for d in outputs.iterdir() if d.is_dir() and 'fc' in d.name and species in d.name])
    run_wo_species = f'{run}_no{species}'
    sigma_file = test_output / f'B_sigma_{species}.dat' # contains two values: B, sigma
    if sigma_file.exists() and cache:
        print(f' {target}: Found {sigma_file}')
        B, sigma = np.loadtxt(sigma_file)
    else:
        
        # if len(run_wo_species) == 0:
        #     print(f' {target}: No runs found without {species} in {outputs}')
        #     return None
        # runs = [run, run_wo_species[-1].name]
        runs = [run, run_wo_species]
        # print(f' {target}: Found runs: {runs}')
        
        log_Z_list = [get_evidence(target, run, key=key) for run in runs]
        if any([log_Z is None for log_Z in log_Z_list]):
            print(f' {target}: Some runs failed...')
            return None, None
        
        B, sigma = compare_evidence(*log_Z_list)
        # save file
        np.savetxt(sigma_file, [B, sigma])
        print(f' {target}: Saved {sigma_file}')
    
    
    return B, sigma

# name = 'Gl 338B' # sigma(AB)=nan, sigma(BA)=2.4
# name = 'Gl 408'    # sigma(AB)=2.2, sigma(BA)=1.7
# name = 'Gl 880'    # sigma(AB)=4.1, sigma(BA)=nan
# name = 'Gl 752A'   # sigma(AB)=nan, sigma(BA)=nan
name = 'Gl 205'
# target = name.replace('Gl ', 'gl')
# stats = main(target, key='global evidence')

ignore_targets = [name.replace('Gl ', 'gl') for name in names if valid[name] == 0]
ignore_more_targets = ['gl3622']
ignore_targets += ignore_more_targets
cache = True
sigma_dict = {}
for name in names:
    target = name.replace('Gl ', 'gl')
    if target in ignore_targets:
        print(f'---> Skipping {target}...')
        continue
    
    print(f'Target = {target}')
    try:
        B, sigma = main(target, key='global evidence', species='C18O', cache=cache)
    except Exception as e:
        print(f'Error for {target}: {e}')
        sigma = None
        continue
    
    sigma_dict[target] = sigma

    

    
    