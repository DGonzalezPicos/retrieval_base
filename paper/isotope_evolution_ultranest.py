import numpy as np
import matplotlib.pyplot as plt
import ultranest
from ultranest.plot import cornerplot

# Data points
x_vals = np.array([-0.08, 0.43, 0.25, 0.13, -0.43, -0.33, -0.37, 0.1, 0.26, -0.09,
                   -0.01, -0.25, 0.42, -0.01, -0.27, 0.11, -0.24, -0.46, -0.3, -0.13,
                   -0.16, 0.05, -0.26, -0.2])
x_vals = 10**x_vals

y_vals = np.array([82.1, 69.7, 78.8, 61.7, 107.1, 94.8, 166.3, 95.1, 54.6, 62.0,
                   62.5, 138.3, 55.5, 101.0, 154.0, 56.2, 247.1, 301.7, 96.5, 90.7,
                   80.8, 86.3, 138.9, 98.8])

y_unc = np.array([[1.914, 1.919], [0.637, 0.637], [0.01, 0.01], [1.507, 1.448],
                  [14.154, 18.084], [9.058, 10.584], [37.922, 62.214], [4.586, 5.097],
                  [1.498, 1.449], [3.739, 4.308], [3.038, 3.266], [26.972, 44.624],
                  [1.382, 1.527], [9.035, 11.778], [40.189, 71.621], [2.534, 2.390],
                  [81.059, 152.198], [113.262, 238.573], [16.310, 20.935], [9.202, 12.729],
                  [7.101, 7.741], [4.947, 5.374], [21.682, 29.702], [8.526, 10.579]])

# Model: y = A/x + B
# def model(x, A, B, C):
#     return A / (x**C) + B

def model(x, A, B):
    return A * np.exp(-B * x)

# Define the likelihood function that accounts for asymmetric uncertainties
def log_likelihood(params):
    A, B = params
    y_model = model(x_vals, *params)
    logL = 0
    for i in range(len(y_vals)):
        if y_model[i] < y_vals[i]:
            sigma = y_unc[i][0]  # lower uncertainty
        else:
            sigma = y_unc[i][1]  # upper uncertainty
        logL += -0.5 * ((y_vals[i] - y_model[i]) / sigma)**2
    return logL

# Define the prior ranges for A and B
def prior_transform(cube):
    A_min, A_max = 0, 400  # Priors on A
    B_min, B_max = 0, 10  # Priors on B
    # C_min, C_max = 0, 1000  # Priors on C
    A = cube[0] * (A_max - A_min) + A_min
    B = cube[1] * (B_max - B_min) + B_min
    # C= cube[2] * (C_max - C_min) + C_min
    
    return [A, B]

# Set up Ultranest sampler
theta_names = ['A', 'B']
sampler = ultranest.ReactiveNestedSampler(
    theta_names, log_likelihood, prior_transform
)

# Run the nested sampling
result = sampler.run(min_num_live_points=200)
sampler.print_results()

# Plot posterior distributions using corner plot
cornerplot(result)

# Extract best-fit values and 1-sigma intervals
theta_median = np.median(result['samples'], axis=0)
theta_1sigma = np.std(result['samples'], axis=0)

# Generate a smooth curve for plotting the fitted function
x_fit = np.linspace(min(x_vals), max(x_vals), 1000)
y_fit = model(x_fit, *theta_median)

# Plot data points, best-fit model, and 1-sigma uncertainty
fig, ax = plt.subplots(1, 1, figsize=(5, 5), tight_layout=True)

# Plot the data points
ax.errorbar(x_vals, y_vals, yerr=y_unc.T, fmt='o', color='black', label='Data')

# Plot best-fit curve
ax.plot(x_fit, y_fit, color='red', label=f'Best fit: A = {theta_median[0]:.2f}, B = {theta_median[1]:.2f}')

# get model envelope, evaluate for all equally-weighted samples
envelope_samples = result['samples'][::len(result['samples'])//1000]
alpha_env = 10.0 / len(envelope_samples)
for theta in envelope_samples:
    y_fit = model(x_fit, *theta)
    ax.plot(x_fit, y_fit, color='red', alpha=alpha_env, lw=0.5)

# Set labels and legend
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()

# Save the plot
path = '/home/dario/phd/retrieval_base/'
fig_name = path + 'ultranest_fitted_model_with_uncertainty.pdf'
plt.savefig(fig_name)
plt.show()

print(f'--> Best-fit parameters:')
for i, name in enumerate(theta_names):
    print(f'    {name} = {theta_median[i]:.2f} +/- {theta_1sigma[i]:.2f}')
print(f'--> Saved {fig_name}')
