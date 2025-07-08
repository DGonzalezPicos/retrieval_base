"""
This script is used to plot the g140h spectrum covering features of TiO isotopes.

* Compare best-fit models from multiple runs of the same target.

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
import copy
from retrieval_base.retrieval import Retrieval
import retrieval_base.auxiliary_functions as af
from retrieval_base.config import Config

from fig1_insets import create_insets

path = af.get_path(return_pathlib=True)
path_figures = pathlib.Path('/home/dario/phd/twa2x_paper/figures')
config_file = 'config_jwst.txt'
w_set = 'NIRSpec'

# Define runs for a single target - multiple runs will be compared
runs = dict(
    TWA28=['freeslab_lbl10_G1_1', 'freeslab_lbl10_G1_2'],
)

# Get the target name (assuming only one target in the dictionary)
target = list(runs.keys())[0]
run_names = runs[target]

# Define colors and line styles for different runs
model_colors = ['#D55E00', '#009E73', '#CC79A7', '#F0E442', '#0072B2']
line_styles = ['-', '--', '-.', ':', '-']

dw = 90
xc = [1110, 2290, 4510]
y_factor = 0.96e14


def load_data(target: str, run: str) -> tuple:
    """Load spectral data and model for a given target and run."""
    cwd = os.getcwd()
    if target not in cwd:
        os.chdir(f'{path}/{target}')
        print(f'Changed directory to {target}')

    conf = Config(path=path, target=target, run=run)(config_file)        
        
    m_spec = af.pickle_load(f'{conf.prefix}data/bestfit_m_spec_NIRSpec.pkl')
    d_spec = af.pickle_load(f'{conf.prefix}data/d_spec_NIRSpec.pkl')
    
    cov = af.pickle_load(f'{conf.prefix}data/bestfit_Cov_NIRSpec.pkl')
    err = np.array([cov[cov_i,0].get_err(mask=d_spec.mask_isfinite[cov_i]) for cov_i in range(len(cov))])
    d_spec.err = err
    print(f'Data shape for {run}: {d_spec.err.shape}')

    m_spec.flux = m_spec.flux.squeeze()
    m_spec.wave = d_spec.wave 
    m_spec.flux_bb = m_spec.blackbody_disk(**m_spec.blackbody_disk_args).squeeze()
    d_spec.squeeze()
    return d_spec, m_spec


# Load data for all runs of the target
d_specs, m_specs = [], []
for run in run_names:
    d_spec, m_spec = load_data(target, run)
    d_specs.append(d_spec)
    m_specs.append(m_spec)

# Data color (same for all since data is the same)
data_color = 'k'
lw = 0.9


def plot_chunk(d_spec, m_spec, idx: int = 0, run_idx: int = 0, 
               plot_data: bool = True, run_label: str = '') -> None:
    """Plot spectral data and model for a given order and run."""
    
    nans = np.isnan(d_spec.flux[idx])
    
    # Plot data only for the first run (since it's the same for all runs)
    if plot_data:
        ax_spec.plot(d_spec.wave[idx], d_spec.flux[idx], 
                    color=data_color, lw=lw, alpha=0.8, ls='-', 
                    label='Data' if idx == 0 else '')
    
    # Plot model with unique color and line style for each run
    color = model_colors[run_idx % len(model_colors)]
    ls = line_styles[run_idx % len(line_styles)]
    
    ax_spec.plot(d_spec.wave[idx], np.where(nans, np.nan, m_spec.flux[idx]), 
                color=color, lw=lw, alpha=0.8, ls=ls,
                label=f'Model {run_label}' if idx == 0 else '')
    
    # Plot residuals
    res = (d_spec.flux[idx] - m_spec.flux[idx]) / d_spec.flux[idx]
    ax_residuals.plot(d_spec.wave[idx], np.where(nans, np.nan, res), 
                     color=color, lw=lw, alpha=0.8, ls='', 
                     marker='o', markersize=0.7)


def calculate_goodness_of_fit(d_spec, m_spec, idx: int) -> tuple[float, float, int]:
    """Calculate MAD and chi-squared goodness of fit metrics."""
    
    # Get data and model fluxes for this order
    data_flux = d_spec.flux[idx]
    model_flux = m_spec.flux[idx]
    error = d_spec.err[idx]
    
    # Create mask for finite values
    finite_mask = np.isfinite(data_flux) & np.isfinite(model_flux) & np.isfinite(error)
    
    if not np.any(finite_mask):
        return np.nan, np.nan, 0
    
    # Apply mask
    data_clean = data_flux[finite_mask]
    model_clean = model_flux[finite_mask]
    error_clean = error[finite_mask]
    
    # Calculate metrics
    residuals = data_clean - model_clean
    n_points = len(data_clean)
    
    # Mean Absolute Deviation (MAD)
    mad = np.mean(np.abs(residuals))
    
    # Chi-squared
    chi2 = np.sum((residuals / error_clean) ** 2)
    
    return mad, chi2, n_points


def format_metrics_text(run_names: list, mad_values: list, chi2_values: list, 
                       n_points_list: list) -> str:
    """Format goodness of fit metrics for display."""
    
    text_lines = ["Goodness of Fit Metrics:"]
    text_lines.append("-" * 25)
    
    for i, run_name in enumerate(run_names):
        mad = mad_values[i]
        chi2 = chi2_values[i]
        n_points = n_points_list[i]
        
        # Calculate reduced chi-squared (assuming some degrees of freedom)
        # For simplicity, using n_points - 1 as DOF
        dof = max(1, n_points - 1)
        chi2_reduced = chi2 / dof
        
        text_lines.append(f"{run_name}:")
        text_lines.append(f"  MAD: {mad:.3e}")
        text_lines.append(f"  χ²: {chi2:.1f}")
        text_lines.append(f"  χ²ᵣ: {chi2_reduced:.2f}")
        text_lines.append(f"  N: {n_points}")
        text_lines.append("")
    
    # Determine best fit
    if len(mad_values) >= 2:
        best_mad_idx = np.argmin(mad_values)
        best_chi2_idx = np.argmin([chi2/n for chi2, n in zip(chi2_values, n_points_list)])
        
        text_lines.append("Best Fit:")
        text_lines.append(f"  Lowest MAD: {run_names[best_mad_idx]}")
        text_lines.append(f"  Lowest χ²ᵣ: {run_names[best_chi2_idx]}")
    
    return "\n".join(text_lines)


# Create figure
fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True, height_ratios=[3, 3, 1])
ax_spec, ax_models, ax_residuals = ax[0], ax[1], ax[2]
# Get number of orders from the first dataset
n_orders = len(d_specs[0].wave)
print(f'Number of orders: {n_orders}')

# Define which orders to plot
orders = [0, 1]

# Plot all runs
total_mad_values = [0.0] * len(run_names)
total_chi2_values = [0.0] * len(run_names)
total_n_points = [0] * len(run_names)

for idx, order in enumerate(orders):
    # Store model fluxes for difference calculation
    model_fluxes = []
    wavelengths = None
    
    for run_idx, run_name in enumerate(run_names):
        d_spec = d_specs[run_idx]
        m_spec = m_specs[run_idx]
        
        # Plot data only for the first run
        plot_data = (run_idx == 0)
        
        plot_chunk(d_spec, m_spec, idx=order, run_idx=run_idx, 
                  plot_data=plot_data, run_label=run_name)
        
        # Calculate goodness of fit metrics
        mad, chi2, n_points = calculate_goodness_of_fit(d_spec, m_spec, order)
        total_mad_values[run_idx] += mad if not np.isnan(mad) else 0
        total_chi2_values[run_idx] += chi2 if not np.isnan(chi2) else 0
        total_n_points[run_idx] += n_points
        
        # Store model flux for difference calculation
        model_fluxes.append(m_spec.flux[order])
        if wavelengths is None:
            wavelengths = d_spec.wave[order]
    
    # Plot model difference in the middle panel (assuming 2 models)
    if len(model_fluxes) >= 2:
        nans = np.isnan(model_fluxes[0]) | np.isnan(model_fluxes[1])
        model_diff = model_fluxes[1] - model_fluxes[0]  # Model 2 - Model 1
        
        ax_models.plot(wavelengths, np.where(nans, np.nan, model_diff), 
                      color='purple', lw=lw, alpha=0.8, ls='-',
                      label=f'{run_names[1]} - {run_names[0]}' if idx == 0 else '')

# Calculate average MAD values across orders
avg_mad_values = [mad / len(orders) for mad in total_mad_values]

# Add labels and formatting
ax_spec.set_ylabel('Flux')
ax_spec.legend(loc='upper right')
ax_spec.grid(True, alpha=0.3)

ax_models.set_ylabel('Model Difference')
ax_models.legend(loc='upper right')
ax_models.grid(True, alpha=0.3)
ax_models.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

ax_residuals.set_xlabel('Wavelength (nm)')
ax_residuals.set_ylabel('Relative Residuals')
ax_residuals.grid(True, alpha=0.3)
ax_residuals.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# Add goodness of fit metrics as text
metrics_text = format_metrics_text(run_names, avg_mad_values, total_chi2_values, total_n_points)
ax_spec.text(0.02, 0.02, metrics_text, transform=ax_spec.transAxes, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            fontsize=8, verticalalignment='bottom', fontfamily='monospace')

plt.tight_layout()
plt.show()