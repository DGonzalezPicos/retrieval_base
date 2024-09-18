import autograd.numpy as np
from autograd import jacobian
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
def forward_model(params, x):
    ret.Param(params)
    sample = {k: ret.Param.params[k] for k in ret.Param.param_keys}
    print(sample)
    ln_L = ret.PMN_lnL_func()
    return ret.LogLike[w_set].m_flux[order]

# Vectorize the forward model using numpy's broadcasting capabilities
def vmap_forward_model(params, x):
    return np.array([forward_model(params, xi) for xi in x])

# Set prior information for the parameters
x_a = np.array([np.mean(v[0]) for _, v in conf.free_params.items()])  # prior state vector
# Estimate the covariance matrix from the prior uncertainties
S_a = np.diag((0.1 * x_a)**2)  # prior uncertainties

# Define the observation uncertainty
# S_e = np.diag(np.array([np.mean(v[1])**2 for _,v in conf.free_params.items()]))
# S_e = np.array([ret.Cov[w_set][i][0]] for i in range(n_orders))

# Forward model for the Jacobian computation
def forward_model_for_jacobian(params):
    return vmap_forward_model(params, ret.d_spec[w_set].wave)

# Compute Jacobian using autograd
def compute_jacobian(params):
    return jacobian(forward_model_for_jacobian)(params)

# Optimal estimation routine
def optimal_estimation(y_observed, x_a, S_a, S_e, max_iterations=100, tolerance=1e-6):
    K = compute_jacobian(x_a)
    S_a_inv = np.linalg.inv(S_a)
    S_e_inv = np.linalg.inv(S_e)
    K_T = K.T
    A = S_a_inv + K_T @ S_e_inv @ K

    def body_fun(state):
        iteration, x_i = state
        y_i = vmap_forward_model(x_i, ret.d_spec[w_set].wave)
        b = K_T @ S_e_inv @ (y_observed - y_i + K @ (x_i - x_a))
        x_new = x_a + np.linalg.solve(A, b)
        return (iteration + 1, x_new)

    def cond_fun(state):
        iteration, x_i = state
        y_i = vmap_forward_model(x_i, ret.d_spec[w_set].wave)
        b = K_T @ S_e_inv @ (y_observed - y_i + K @ (x_i - x_a))
        x_new = x_a + np.linalg.solve(A, b)
        return np.linalg.norm(x_new - x_i) > tolerance

    state = (0, x_a)
    iteration, x_final = state
    for i in range(max_iterations):
        iteration, x_final = body_fun(state)
        if not cond_fun(state):
            break
        state = (iteration, x_final)

    return x_final

# Run optimal estimation
y_observed = ret.d_spec[w_set].flux[order]
S_e = ret.Cov[w_set][order][0]

optimal_params = optimal_estimation(y_observed, x_a, S_a, S_e)

# Plot the results (You can add plotting logic here)
