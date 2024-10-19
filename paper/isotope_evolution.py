import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Data points
x_vals = np.array([-0.08, 0.43, 0.25, 0.13, -0.43, -0.33, -0.37, 0.1, 0.26, -0.09,
                   -0.01, -0.25, 0.42, -0.01, -0.27, 0.11, -0.24, -0.46, -0.3, -0.13,
                   -0.16, 0.05, -0.26, -0.2])
x_vals = 10**x_vals

y_vals = np.array([82.1, 69.7, 78.8, 61.7, 107.1, 94.8, 166.3, 95.1, 54.6, 62.0,
                   62.5, 138.3, 55.5, 101.0, 154.0, 56.2, 247.1, 301.7, 96.5, 90.7,
                   80.8, 86.3, 138.9, 98.8])

# Function to fit y = A/x + B
def func(x, A, B):
    return A / x + B

# Use curve_fit to fit the model to the data
params, cov = curve_fit(func, x_vals, y_vals)

# Extract the best-fit values for A and B
A_fit, B_fit = params

# Generate a smooth curve for plotting the fitted function
x_fit = np.linspace(min(x_vals), max(x_vals), 1000)
y_fit = func(x_fit, A_fit, B_fit)

# Plot data points and the fitted function
fig, ax = plt.subplots(1,1, figsize=(5,5), tight_layout=True)
ax.scatter(x_vals, y_vals, color='black', label='Data')
ax.plot(x_fit, y_fit, color='red', label='Fitted function')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()

# Save the plot
path = '/home/dario/phd/retrieval_base/'
fig_name = path + 'fitted_function_A_div_x_plus_B.pdf'
plt.savefig(fig_name)
plt.show()

print(f'--> Fitted parameters: A = {A_fit:.2f}, B = {B_fit:.2f}')
print(f'--> Saved {fig_name}')
