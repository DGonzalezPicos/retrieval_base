import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pathlib
import numpy as np
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
    sample = {k:ret.Param.params[k] for k in ret.Param.param_keys}
    print(sample)
    ln_L = ret.PMN_lnL_func()

    return ret.LogLike[w_set].m_flux[order]

# vectorized forward model
vmap_forward_model = jax.vmap(forward_model, in_axes=(None, 0))

# Set prior information for the parameters
x_a = jnp.array([np.mean(v[0]) for _,v in conf.free_params.items()]) # prior state vector
# estimate the covariance matrix from the prior uncertainties
S_a = jnp.diag((0.1  * x_a)**2) # prior uncertainties

# Define the observation uncertainty
# S_e = jnp.diag(jnp.array([np.mean(v[1])**2 for _,v in conf.free_params.items()]))
# S_e = np.array([ret.Cov[w_set][i][0]] for i in range(n_orders))

def forward_model_for_jacobian(params):
    return vmap_forward_model(params, ret.d_spec[w_set].wave)

def compute_jacobian(params):
    return jax.jacobian(forward_model_for_jacobian)(params)

def optimal_estimation(y_observed, x_a, S_a, S_e, max_iterations=100, tolerance=1e-6):
    
    K = compute_jacobian(x_a)
    S_a_inv = jnp.linalg.inv(S_a)
    S_e_inv = jnp.linalg.inv(S_e)
    K_T = K.T
    A = S_a_inv + K_T @ S_e_inv @ K
    
    def body_fun(state):
        iteration, x_i = state
        y_i = vmap_forward_model(x_i, ret.d_spec[w_set].wave)
        b = K_T @ S_e_inv @ (y_observed - y_i + K @ (x_i - x_a))
        
        x_new = x_a + jax.scipy.linalg.solve(A, b)
        return (iteration + 1, x_new)
    
    def cond_fun(state):
        iteration, x_i = state
        y_i = vmap_forward_model(x_i, ret.d_spec[w_set].wave)
        b = K_T @ S_e_inv @ (y_observed - y_i + K @ (x_i - x_a))
        x_new = x_a + jax.scipy.linalg.solve(A, b)
        
        return jnp.linalg.norm(x_new - x_i) > tolerance
    
    state = (0, x_a)
    iteration, x_final = jax.lax.while_loop(cond_fun, body_fun, state)
    
    return x_final

# Run optimal estimation
y_observed = ret.d_spec[w_set].flux[order]
S_e = ret.Cov[w_set][order][0]

optimal_estimation(y_observed, np.array(x_a), S_a, S_e)



# Plot the results

    
