from retrieval_base.retrieval import Retrieval
from retrieval_base.config import Config
from retrieval_base.auxiliary_functions import pickle_load

import numpy as np
import pymultinest

base_path = '/home/dario/phd/retrieval_base/'


def compare_evidence(ln_Z_A, ln_Z_B):
    '''Convert log-evidences of two models to a sigma confidence level
    
    Adapted from samderegt/retrieval_base'''

    from scipy.special import lambertw as W
    from scipy.special import erfcinv

    ln_list = [ln_Z_B, ln_Z_A]
    
    for i in range(2):
        ln_list = ln_list[::-1] if i == 1 else ln_list
        labels = ['A', 'B'] if i == 1 else ['B', 'A']
        ln_B = ln_list[0] - ln_list[1]
        B = np.exp(ln_B)
        p = np.real(np.exp(W((-1.0/(B*np.exp(1))),-1)))
        sigma = np.sqrt(2)*erfcinv(p)
        
        print(f'{labels[0]} vs. {labels[1]}: ln(B)={ln_B:.2f} | sigma={sigma:.2f}')
    return B, sigma

def get_evidence(target, run=None, key='global evidence'):
    
    config_file = 'config_jwst.txt'
    conf = Config(path=base_path, target=target, run=run)(config_file)

    try:
        ret = Retrieval(
                    conf=conf, 
                    evaluation=False,
                    )

        # log_Z = ret.PMN_stats()[key]
        analyzer = pymultinest.Analyzer(
                    n_params=len(conf.free_params), 
                    outputfiles_basename=conf.prefix + 'output/test_'
                    )
        stats = analyzer.get_stats()

        # log_Z = stats['nested importance sampling global log-evidence']
        log_Z = stats['global evidence']
        loglike = pickle_load("/".join(ret.conf_output.split('/')[:-2]) + '/test_data/bestfit_LogLike_NIRSpec.pkl')
        chi2 = loglike.chi_squared_red
        
    except Exception as e:
        print(f'Error for {target} {run}: {e}')
        # log_Z = None
        return None, None
        
    return log_Z, chi2


# Define runs for a single target - multiple runs will be compared
runs = dict(
    TWA28=['freeslab_lbl10_G1_1', 'freeslab_lbl10_G1_2'],
)

# Get the target name (assuming only one target in the dictionary)
target = list(runs.keys())[0]
run_names = runs[target]

log_Z_list = []
chi2_list = []
for run in run_names:
    log_Z, chi2 = get_evidence(target, run)
    log_Z_list.append(log_Z)
    chi2_list.append(chi2)

compare_evidence(log_Z_list[0], log_Z_list[1])







