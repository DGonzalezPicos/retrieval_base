import pyOptimalEstimation as pyOE

import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os

from retrieval_base.retrieval import Retrieval
from retrieval_base.config import Config

config_file = 'config_jwst.txt'
target = 'TWA27A'
run = 'lbl15_KM7'

path = pathlib.Path('/home/dario/phd/retrieval_base') 
cwd = os.getcwd()
if target not in cwd:
    nwd = os.path.join(cwd, target)
    print(f'Changing directory to {nwd}')
    os.chdir(nwd)

conf = Config(path=path, target=target, run=run)(config_file)
ret = Retrieval(conf=conf, evaluation=False)
w_set = 'NIRSpec'
n_orders = len(ret.d_spec[w_set].wave)
order = 0

# Implement optimal_estimation function
def forward_model(x):
    ret.Param(x)
    sample = {k:v for k,v in ret.Param.params.items() if k in ret.Param.param_keys}
    print(sample)
    # print(f' ret.Param.params: {ret.Param.params}')
    ln_L = ret.PMN_lnL_func()
    nans = np.isnan(ret.d_spec[w_set].flux[order,0])
    
    # print(f' shape of ret.LogLike[w_set].m_flux[order]: {ret.LogLike[w_set].m_flux[order,0].shape}')
    # print(f' shape of nans: {nans.shape}')
    return ret.LogLike[w_set].m_flux[order,0,~nans]




# define names for X and Y
x_vars = ret.Param.param_keys
x_a = 0.5 * np.ones(ret.Param.n_params)
# test forward model
flux = forward_model(x_a)
# covariance matrix of state x
S_a = np.diag((0.1 * x_a)**2)
# names of the elements of state vector x
# observed measurements
y_obs = ret.d_spec[w_set].flux[order,0]
nans = np.isnan(y_obs)
y_obs = y_obs[~nans]
y_vars = [f'flux_{i}' for i in range(len(y_obs))]

S_y = np.diag(ret.Cov[w_set][order][0].cov)


# ret.d_spec[w_set].wave[order,0] = ret.d_spec[w_set].wave[order,0, ~nans]

# print shape of everything
# print(f'x_vars.shape: {x_vars.shape}')
print(f'x_a.shape: {x_a.shape}')
print(f'S_a.shape: {S_a.shape}')
# print(f'y_vars.shape: {y_vars.shape}')
print(f'y_obs.shape: {y_obs.shape}')
print(f'S_y.shape: {S_y.shape}')


# call optimal estimation
oe = pyOE.optimalEstimation(x_vars=x_vars,
                            y_vars=y_vars,
                            forward=forward_model,
                            x_a=x_a,
                            S_a=S_a,
                            y_obs=y_obs,
                            S_y=S_y,
                            )

# run the retrieval
oe.doRetrieval(maxIter=2, maxTime=10.0)
# plot the result
oe.plotIterations()
plt.savefig("pRT_oe_result.png")