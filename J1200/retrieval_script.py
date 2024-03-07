import argparse
from retrieval_base.retrieval import pre_processing, Retrieval

import config_freechem as conf

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
        for conf_data_i in conf.config_data.values():
            pre_processing(conf=conf, conf_data=conf_data_i)
            
    if args.check:
        ret = Retrieval(
            conf=conf, 
            evaluation=args.evaluation
            )
        ret.prior_check()

    if args.retrieval:
        ret = Retrieval(
            conf=conf, 
            evaluation=args.evaluation
            )
        ret.PMN_run()

    if args.evaluation:
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

