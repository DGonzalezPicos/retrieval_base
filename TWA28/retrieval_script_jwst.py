import argparse
import pathlib

from retrieval_base.retrieval import pre_processing, prior_check, Retrieval
from retrieval_base.spectrum_jwst import SpectrumJWST
import retrieval_base.auxiliary_functions as af
from retrieval_base.parameters import Parameters

import config_jwst as conf


# create output directory

for key in ['data', 'plots']:
    pathlib.Path(f'{conf.prefix}{key}/').mkdir(parents=True, exist_ok=True)
    # print(f'--> Created {conf.prefix}{key}/')
if __name__ == '__main__':

    # Instantiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--pre_processing', '-p', action='store_true', default=False)
    parser.add_argument('--prior_check', '-c', action='store_true', default=False)
    parser.add_argument('--retrieval', '-r', action='store_true', default=False)
    parser.add_argument('--evaluation', '-e', action='store_true', default=False)
    args = parser.parse_args()

    if args.pre_processing:
        spec = SpectrumJWST(file='jwst/TWA28_g395h-f290lp.fits')
        spec.split_grism(4155., keep=1)
        # spec.sigma_clip(sigma=3, width=5, max_iter=5, fun='median')
        spec.sigma_clip(spec.err, sigma=3, width=50, max_iter=5, fun='median')
        spec.reshape(1,1)
        spec.prepare_for_covariance()

        af.pickle_save(f'{conf.prefix}data/d_spec_{spec.w_set}.pkl', spec)
        print(f'--> Saved {f"{conf.prefix}data/d_spec_{spec.w_set}.pkl"}')

    if args.prior_check:
        print('--> Running prior predictive check..')
        figs_path = pathlib.Path(f'{conf.prefix}plots/')
        figs_path.mkdir(parents=True, exist_ok=True)
        
        prior_check(conf=conf, fig_name=figs_path / 'prior_predictive_check.pdf')

    if args.retrieval:
        ret = Retrieval(
            conf=conf, 
            evaluation=args.evaluation
            )
        ret.PMN_run()

    if args.evaluation:
        ret = Retrieval(
            conf=conf, 
            evaluation=args.evaluation
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
