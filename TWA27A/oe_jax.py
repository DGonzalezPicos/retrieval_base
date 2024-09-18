import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# 1. Generate fake data (true model with noise)
true_m = 2.5    # True slope
true_b = 1.0    # True intercept
x = jnp.linspace(0, 10, 50)
y_true = true_m * x + true_b

# Add some noise to the observations
noise_std = 1.0
key = jax.random.PRNGKey(42)
noise = jax.random.normal(key, shape=(len(x),)) * noise_std
y_observed = y_true + noise

# 2. Define the forward model (linear function)
def forward_model(params, x):
    m, b = params
    return m * x + b

# Vectorized forward model using vmap
vmap_forward_model = jax.vmap(forward_model, in_axes=(None, 0))

# 3. Set prior information for the parameters m and b
prior_m = 1.0  # Initial guess for slope
prior_b = 0.0  # Initial guess for intercept
x_a = jnp.array([prior_m, prior_b])  # Prior state vector (initial guess)
S_a = jnp.diag(jnp.array([0.5, 0.5]))  # Prior uncertainties (covariance matrix)

# 4. Define the observation uncertainty
S_e = jnp.diag(jnp.array([noise_std**2] * len(x)))  # Observation error covariance matrix

# 5. Automatic Jacobian calculation using jax.jacobian
def forward_model_for_jacobian(params):
    return vmap_forward_model(params, x)

# Use jax.jacobian to compute the Jacobian of the forward model with respect to the parameters
def compute_jacobian(params):
    return jax.jacobian(forward_model_for_jacobian)(params)

# 6. Optimal Estimation (OE) function with lax.while_loop for convergence
def optimal_estimation(y_observed, x_a, S_a, S_e, max_iterations=100, tolerance=1e-6):
    K = compute_jacobian(x_a)  # Precompute Jacobian

    # Precompute matrices that don't change over iterations
    S_a_inv = jnp.linalg.inv(S_a)
    S_e_inv = jnp.linalg.inv(S_e)
    K_T = K.T
    A = S_a_inv + K_T @ S_e_inv @ K

    def body_fun(state):
        iteration, x_i = state
        y_i = vmap_forward_model(x_i, x)  # Forward model
        b = K_T @ S_e_inv @ (y_observed - y_i + K @ (x_i - x_a))
        
        # Solve the system
        x_new = x_a + jax.scipy.linalg.solve(A, b)
        return (iteration + 1, x_new)

    def cond_fun(state):
        iteration, x_i = state
        y_i = vmap_forward_model(x_i, x)
        b = K_T @ S_e_inv @ (y_observed - y_i + K @ (x_i - x_a))
        x_new = x_a + jax.scipy.linalg.solve(A, b)
        
        # Check the convergence condition
        return jnp.linalg.norm(x_new - x_i) > tolerance

    # Initial state (iteration = 0, starting with x_a)
    state = (0, x_a)
    
    # Perform the loop using jax.lax.while_loop
    iteration, x_final = jax.lax.while_loop(cond_fun, body_fun, state)

    return x_final

# 7. Perform the optimal estimation
x_optimal = optimal_estimation(y_observed, x_a, S_a, S_e)
optimal_m, optimal_b = x_optimal

# 8. Plot the results
y_fitted = forward_model(x_optimal, x)
plt.scatter(x, y_observed, label='Noisy Observations', color='red')
plt.plot(x, y_true, label='True Model', color='green')
plt.plot(x, y_fitted, label=f'Fitted Model (y = {optimal_m:.2f}x + {optimal_b:.2f})', color='blue')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Print the optimal parameters
print(f' * True Slope (m): {true_m:.2f}')
print(f" -> Optimal Slope (m): {optimal_m:.2f}")

print(f' * True Intercept (b): {true_b:.2f}')
print(f"-> Optimal Intercept (b): {optimal_b:.2f}")
