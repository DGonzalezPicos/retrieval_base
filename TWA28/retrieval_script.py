import argparse
import subprocess
import os
cwd = os.getcwd()
if 'dgonzalezpi' in cwd:
    path = '/home/dgonzalezpi/retrieval_base/'
if 'dario' in cwd:
    path = '/home/dario/phd/retrieval_base/'
    
from retrieval_base.retrieval import pre_processing, Retrieval
from retrieval_base.config import Config
import config_freechem as conf



config_file = 'config_freechem.txt'
target = 'TWA28'
run = 'rev_3' # important to set this to the correct run 

if __name__ == '__main__':

    # Instantiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--pre_processing', '-p', action='store_true')
    parser.add_argument('--check', '-c', action='store_true')
    parser.add_argument('--retrieval', '-r', action='store_true')
    parser.add_argument('--evaluation', '-e', action='store_true')
    parser.add_argument('--ccf', '-ccf', action='store_true', help='Cross-correlation function', default=False)
    # parser.add_argument('--synthetic', action='store_true')
    args = parser.parse_args()

    if args.pre_processing:
        assert conf.run == run, f'Run {run} does not match run in config file {conf.run}'
        subprocess.run(['python', 'config_freechem.py'])
        
        for conf_data_i in conf.config_data.values():
            pre_processing(conf=conf, conf_data=conf_data_i)
            
                        
    if args.check:
        ret = Retrieval(
            conf=conf, 
            evaluation=args.evaluation
            )
        ret.prior_check()

    if args.retrieval:
        conf = Config(path=path, target=target, run=run)
        conf(config_file)
    
        ret = Retrieval(
            conf=conf, 
            evaluation=args.evaluation
            )
        ret.PMN_run()

    if args.evaluation:
        conf = Config(path=path, target=target, run=run)
        conf(config_file)
    
        ret = Retrieval(
            conf=conf, 
            evaluation=args.evaluation,
            plot_ccf=args.ccf
            )
            
        ret.PMN_callback_func(
            n_samples=None, 
            n_live=None, 
            n_params=None, 
            live_points=None, 
            posterior=None, 
            stats=None,
            max_ln_L=None, 
            ln_Z=None, 
            ln_Z_err=None, 
            nullcontext=None
            )

