from retrieval_base.retrieval import Retrieval
import retrieval_base.figures as figs
from retrieval_base.config import Config
from retrieval_base.auxiliary_functions import spirou_sample, read_spirou_sample_csv, compare_evidence, pickle_load
# import config_freechem as conf
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib

base_path = '/home/dario/phd/retrieval_base/'


df = read_spirou_sample_csv()
names = df['Star'].to_list()
teff =  dict(zip(names, [float(t.split('+-')[0]) for t in df['Teff (K)'].to_list()]))

def get_evidence(target, run=None, key='global evidence'):
    
    config_file = 'config_freechem.txt'
    conf = Config(path=base_path, target=target, run=run)(config_file)

    try:
        ret = Retrieval(
                    conf=conf, 
                    evaluation=False,
                    )

        log_Z = ret.PMN_stats()[key]
        loglike = pickle_load("/".join(ret.conf_output.split('/')[:-2]) + '/test_data/bestfit_LogLike_spirou.pkl')
        chi2 = loglike.chi_squared_red
        
    except Exception as e:
        print(f'Error for {target} {run}: {e}')
        # log_Z = None
        return None, None
        
    return log_Z, chi2

def main(target, run=None, species='C18O', key='global evidence', cache=True):
    
    if target not in os.getcwd():
        os.chdir(base_path + target)

    outputs = pathlib.Path(base_path) / target / 'retrieval_outputs'
    # find dirs in outputs
    # print(f' outputs = {outputs}')
    dirs = [d for d in outputs.iterdir() if d.is_dir() and 'fc' in d.name and '_' not in d.name]
    # print(f' dirs = {dirs}')
    runs = [int(d.name.split('fc')[-1]) for d in dirs]
    # print(f' runs = {runs}')
    # print(f' {target}: Found {len(runs)} runs: {runs}')
    assert len(runs) > 0, f'No runs found in {outputs}'
    if run is None:
        run = 'fc'+str(max(runs))
    else:
        run = 'fc'+str(run)
        assert run in [d.name for d in dirs], f'Run {run} not found in {dirs}'
        
        
    # print('Run:', run)
    # check that the folder 'test_output' is not empty
    test_output = outputs / run / 'test_output'
    # test_data = outputs / run / 'test_data'
    assert test_output.exists(), f'No test_output folder found in {test_output}'
    if len(list(test_output.iterdir())) == 0:
        print(f' {target}: No files found in {test_output}')
        return None
    
    # run_wo_species = sorted([d for d in outputs.iterdir() if d.is_dir() and 'fc' in d.name and species in d.name])
    run_wo_species = f'{run}_no{species}'
    sigma_file = test_output / f'lnB_sigma_{species}.dat' # contains two values: B, sigma
    # chi2_file = test_output / f'delta_chi2_{species}.dat' # contains two values: B, sigma
    # if all([f.exists() for f in [sigma_file, chi2_file]]+[cache]):
    if sigma_file.exists() and cache:
        # print(f' {target}: Found {sigma_file}')
        lnB, sigma = np.loadtxt(sigma_file)
        # delta_chi2 = np.loadtxt(chi2_file)
        delta_chi2=0.0
    else:
        
        # if len(run_wo_species) == 0:
        #     print(f' {target}: No runs found without {species} in {outputs}')
        #     return None
        # runs = [run, run_wo_species[-1].name]
        runs = [run, run_wo_species]
        # print(f' {target}: Found runs: {runs}')
        log_Z_list, chi2_r_list = zip(*[get_evidence(target, run, key=key) for run in runs])
        print(f' Target: log Z = {log_Z_list}')
        if any([log_Z is None for log_Z in log_Z_list]):
            print(f' {target}: Some runs failed...')
            return None, None
        
        lnB, sigma = compare_evidence(*log_Z_list)
        delta_chi2 = chi2_r_list[1] - chi2_r_list[0] # smaller chi2 better fit! should be *positive* if species is detected
        # save file
        np.savetxt(sigma_file, [lnB, sigma])
        print(f' {target}: Saved {sigma_file}')
        
        # np.savetxt(chi2_file, [delta_chi2])
        # print(f' {target}: Saved {chi2_file}')
    
    
    return lnB, sigma, delta_chi2

# ignore_targets = [name.replace('Gl ', 'gl') for name in names if valid[name] == 0]
# ignore_more_targets = ['gl3622']
ignore_targets = []
cache = True
evidence_dict = {}

testing = False
# species = '13CO'
species = 'C18O'
ignore_chi2 = True
for i, name in enumerate(names):
    target = name.replace('Gl ', 'gl')
    if target in ignore_targets:
        print(f'---> Skipping {target}...')
        continue
    
    print(f'Target = {target}')
    try:
        lnB, sigma, delta_chi2 = main(target, key='global evidence', species=species, cache=cache)
        print(f' ln B = {lnB:.2f}, sigma = {sigma:.2f}, delta_chi2 = {delta_chi2:.2f}')
    except Exception as e:
        print(f'Error for {target}: {e}')
        sigma = None
        # continue
    
    if testing:
        break
    
    # sigma_dict[target] = sigma
    evidence_dict[target] = (lnB, sigma)
    


save_path = pathlib.Path(base_path) / 'paper/data' / f'lnB_sigma{species}.dat'
# create array with three columns: name, lnB, sigma
np.savetxt(save_path, np.array([[name, *evidence_dict[name]] for name in evidence_dict.keys()]), fmt='%s')
print(f' Saved {save_path}')

    
    