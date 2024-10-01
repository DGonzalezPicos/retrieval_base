import argparse
import subprocess
    
from retrieval_base.retrieval import pre_processing_spirou, prior_check, Retrieval
from retrieval_base.config import Config
from retrieval_base.auxiliary_functions import get_path
path = get_path()

import config_freechem as conf



config_file = 'config_freechem.txt'
target = 'gl876'
run = 'sphinx1' # important to set this to the correct run

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
        
        # for conf_data_i in conf.config_data.values():
        pre_processing_spirou(conf=conf, conf_data=conf.config_data['spirou'])            
                        
    if args.check:
        # ret = Retrieval(
        #     conf=conf, 
        #     evaluation=args.evaluation
        #     )
        # ret.prior_check()
        prior_check(conf,
                    n=3,
                    random=True,
                    w_set='spirou',
                    fig_name=conf.prefix + 'plots/prior_check.pdf',
        )

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
            # plot_ccf=args.ccf
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

