import argparse
import subprocess
import os

from retrieval_base.retrieval import pre_processing_spirou, prior_check, Retrieval
from retrieval_base.config import Config
from retrieval_base.auxiliary_functions import get_path
path = get_path()

import config_freechem as conf



config_file = 'config_freechem.txt'
# target = 'gl436'
run = 'fc5_noC18O' # important to set this to the correct run

if __name__ == '__main__':

    # Instantiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--pre_processing', '-p', action='store_true')
    parser.add_argument('--check', '-c', action='store_true')
    parser.add_argument('--retrieval', '-r', action='store_true')
    parser.add_argument('--evaluation', '-e', action='store_true')
    parser.add_argument('--ccf', '-ccf', action='store_true', help='Cross-correlation function', default=False)
    parser.add_argument('--target', '-t', type=str, help='Target name', default='gl436')
    parser.add_argument('--run', '-run', type=str, help='Run name', default='None')
    parser.add_argument('--resume', '-resume', type=int, help='Resume from last run', default=1)
    parser.add_argument('--cache_pRT', '-cache_pRT', type=str, help='Cache pRT', default='False')
    parser.add_argument('--to_snellius', '-to_snellius', action='store_true')
    # parser.add_argument('--synthetic', action='store_true')
    args = parser.parse_args()
    target = args.target
    run = args.run if args.run != 'None' else run

    if args.pre_processing:
        # assert conf.run == run, f'Run {run} does not match run in config file {conf.run}'
        subprocess.run(['python', 'config_freechem.py'])
        
        # for conf_data_i in conf.config_data.values():
        pre_processing_spirou(conf=conf, conf_data=conf.config_data['spirou'], cache_pRT=(args.cache_pRT == 'True'))
                        
    if args.check:
        # ret = Retrieval(
        #     conf=conf, 
        #     evaluation=args.evaluation
        #     )
        # ret.prior_check()
        prior_check(conf,
                    n=3,
                    random=False,
                    w_set='spirou',
                    fig_name=conf.prefix + 'plots/prior_check.pdf',
        )
    if args.to_snellius:
        
        # copy this folder to snellius
        snellius_dir = f'/home/dgonzalezpi/retrieval_base/{target}/retrieval_outputs/{conf.run}'
        local_dir=f'/home/dario/phd/retrieval_base/{target}/retrieval_outputs/{conf.run}'
        print(f' Copying {local_dir} to {snellius_dir}...')
        
        # if parent directory does not exist, create it on remote
        # subprocess.run(f'scp -r {local_dir} dgonzalezpi@snellius.surf.nl:{snellius_dir}', shell=True, check=True)
        # use rync -av --delete instead of scp -r
        rsync_command = f'rsync -av --delete {local_dir}/ dgonzalezpi@snellius.surf.nl:{snellius_dir}/'
        try:
            subprocess.run(rsync_command, shell=True, check=True)
            
        except:
            proxy_jump = 'ssh.strw.leidenuniv.nl'
        # Define SSH command with all options for skipping confirmation
            ssh_command = f"ssh -J picos@{proxy_jump} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o CheckHostIP=no"
            # Set the RSYNC_RSH environment variable to the SSH command
            os.environ["RSYNC_RSH"] = ssh_command

            # Now run rsync without needing the -e option
            print(f' Attempting to use RSYNC_RSH with command:\n{rsync_command}')
            
            subprocess.run(rsync_command, shell=True, check=True)
            print(f' Successful download for {target} {run}!\n')

        print(f' Succesful copy for {target}!\n')
        
    if args.retrieval:
        conf = Config(path=path, target=target, run=run)(config_file)
    
        ret = Retrieval(
            conf=conf, 
            evaluation=args.evaluation
            )
        ret.PMN_resume = bool(args.resume)
        ret.PMN_run()

    if args.evaluation:
        conf = Config(path=path, target=target, run=run)(config_file)
    
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

