import numpy as np
import matplotlib.pyplot as plt
plt.style.use('/home/dario/phd/retrieval_base/HBDs/my_science.mplstyle')
import pathlib
import os
import pymultinest
import pickle

from retrieval_base.config import Config
from retrieval_base.retrieval import Retrieval

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

targets = dict(
                # J1200='final_full',
                # TWA28='final_full',
                J0856='final_full',
                )

colors = dict(J1200='royalblue', TWA28='seagreen', J0856='indianred')
# target = 'TWA28'
# cwd = str(os.getcwd())
# if target not in cwd:
#     os.chdir(cwd+f'/{target}')

path = pathlib.Path('/home/dario/phd/retrieval_base')
out_path = pathlib.Path('/home/dario/phd/Hot_Brown_Dwarfs_Retrievals/tables/')
# run = targets[target]
runs = ['final_full', 'final_no13CO', 'final_noH218O']

log_Z_list = []
for i, (target) in enumerate(targets.keys()):
    data_path =  path / f'{target}'
    log_Z_list = []
    chi2_r_list = []
    B_list = []
    sigma_list = []
    print(f'\n ** {target} **')
    for j, retrieval_id in enumerate(runs):
        retrieval_path = data_path / f'retrieval_outputs/{retrieval_id}'

        conf = Config(path=path, target=target, run=retrieval_id)('config_freechem.txt')
        # ret = Retrieval(conf=conf, evaluation=False)
        loglike = pickle.load(open(retrieval_path / 'test_data/bestfit_LogLike_K2166.pkl', 'rb'))
        chi2_r = loglike.chi_squared_red
        print(f' chi2_r = {chi2_r:.2f}')
        chi2_r_list.append(chi2_r)

        outputfiles_basename = str(path / target / (conf.prefix[2:]))

        analyzer = pymultinest.Analyzer(
                    n_params=len(conf.free_params), 
                    outputfiles_basename=outputfiles_basename
                    )
        stats = analyzer.get_stats()

        # log_Z = stats['nested importance sampling global log-evidence']
        log_Z = stats['global evidence']
        # print(f' log_Z = {log_Z}')
        log_Z_list.append(log_Z)
        B = '-'
        if j > 0:
            print(f'retrieval_id = {retrieval_id}')
            # print(f' log_Z_list = {log_Z_list}')
            B, sigma = compare_evidence(log_Z_list[0], log_Z_list[-1])
            B_list.append(B)        
            sigma_list.append(sigma)
        
        
    # create latex table with the chi2_r, lnB and sigma
    # increase vertical spacing
    
    table = f'''\\begin{{table}}[h]
    \\centering
    \\renewcommand{{\\arraystretch}}{{1.2}}
    \\begin{{tabular}}{{cccc}}
    \\hline
    \\hline
    \\textbf{{Model}} & $\\chi^2_r$ & ln B\\textsubscript{{m}}& $\\sigma$ \\\\
    \\hline
    Full & {chi2_r_list[0]:.2f} & - & - \\\\
    No \\textsuperscript{{13}}CO & {chi2_r_list[1]:.2f} & {np.log(B_list[0]):.2f} & {sigma_list[0]:.2f} \\\\
    No H\\textsubscript{{2}}\\textsuperscript{{18}}O & {chi2_r_list[2]:.2f} & {np.log(B_list[1]):.2f}&{sigma_list[1]:.2f} \\\\
    \\hline
    \\end{{tabular}}
    \\caption{{Comparison of the models for {target}.}}
    \\label{{tab:{target}_comparison}}
\\end{{table}}
            '''
    with open(out_path / f'{target}_comparison.tex', 'w') as f:
        f.write(table)
    print(f'Wrote table to {out_path / f"{target}_comparison.tex"}')
    
    
