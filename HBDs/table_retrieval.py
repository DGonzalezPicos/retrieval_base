import numpy as np
import pathlib
import json

from tabulate import tabulate



path = pathlib.Path('/home/dario/phd/retrieval_base')
targets = dict(J1200='freechem_10', 
               TWA28='freechem_6', 
               J0856='freechem_9'
               )

out_path = pathlib.Path('/home/dario/phd/Hot_Brown_Dwarfs_Retrievals/')


# Load priors of the parameters
config_path = f"/home/dario/phd/retrieval_base/J0856/retrieval_outputs/{targets['J0856']}/test_data/"
config_file = f'{config_path}config_freechem.txt'
with open(config_file, 'r') as file:
    load_file = json.load(file)
    
# remove key for R_p
# load_file['free_params'].pop('R_p')


free_params = load_file['free_params']
names_params = list(free_params.keys())
# add description for each parameter

descriptions = {'log_a': 'GP amplitude',
                'log_l': 'GP lengthscale',
                'R_p': 'radius of the planet',
                'log_g': 'log surface gravity',
                'epsilon_limb': 'limb darkening coefficient',
                'vsini': 'projected rotational velocity',
                'rv': 'radial velocity',
                'log_12CO': 'log mixing ratio of \\twelveCO',
                'log_13CO': 'log mixing ratio of \\thirteenCO',
                'log_H2O': 'log mixing ratio of \\water',
                'log_HF': 'log mixing ratio of HF',
                'log_Na': 'log mixing ratio of Na',
                'log_Ca': 'log mixing ratio of Ca',
                'log_Ti': 'log mixing ratio of Ti',
                'dlnT_dlnP_0': 'temperature gradient at $10^2$ bar',
                'dlnT_dlnP_1': 'temperature gradient at $10^{1}$ bar',
                'dlnT_dlnP_2': 'temperature gradient at $10^{-1}$ bar',
                'dlnT_dlnP_3': 'temperature gradient at $10^{-3}$ bar',
                'dlnT_dlnP_4': 'temperature gradient at $10^{-5}$ bar',
                'T_0': 'temperature at $10^{2}$ bar',
                }

# Convert the parameters to a list for tabulate
# priors_table = [(label, f"[{low}, {high}]", desc) for label, (low, high), desc in free_params.values()]

ignore_params = ['R_p']
table = [[val[1], descriptions[key], f"[{val[0][0]}, {val[0][1]}]"] for (key, val) in free_params.items()
         if key not in ignore_params]
# Define the headers
headers = ["Parameter", "Description", "Prior Range"]

GP_eq_block = {}
for i, (target, retrieval_id) in enumerate(targets.items()):
    data_path = path / f'{target}'
    headers += [target]
    # bestfit_params = 
    retrieval_path = data_path / f'retrieval_outputs/{retrieval_id}'
    assert retrieval_path.exists(), f'Retrieval path {retrieval_path} does not exist.'
        
    equal_weighted_file = retrieval_path / 'test_post_equal_weights.dat'
    posterior = np.loadtxt(equal_weighted_file)
    posterior = posterior[:,:-1]
    
    assert len(names_params) == posterior.shape[1], 'Number of parameters does not match the number of columns in the posterior.'
    quantiles = np.quantile(posterior, [0.16, 0.5, 0.84], axis=0)
    bestfit_params = dict(zip(names_params, quantiles.T))
    
    sample_rate = 7.802e-03 # nm at 2345 nm
    # convert to resolution element in km/s using the spectral resolution 50000
    sample_rate_kms = 3e5 * sample_rate / 2345
    print(f'sample rate = {sample_rate_kms:.3f} km/s')
    # print best-fit parameters for a, l
    a = 10**bestfit_params['log_a'][1]
    da_low = (bestfit_params['log_a'][1] - bestfit_params['log_a'][0]) * a * np.log(10)
    da_up = (bestfit_params['log_a'][2] - bestfit_params['log_a'][1]) * a * np.log(10)
    
    l = 10**bestfit_params['log_l'][1]
    l_pixels = np.round(l / sample_rate, 1)
    dl_low = (bestfit_params['log_l'][1] - bestfit_params['log_l'][0]) * l * np.log(10)
    dl_up = (bestfit_params['log_l'][2] - bestfit_params['log_l'][1]) * l * np.log(10)
    print(f'Target: {target}')
    print(f'best-fit a = {a:.4f} +{da_up:.4f} -{da_low:.4f}')
    print(f'best-fit l = {l:.4f} +{dl_up:.4f} -{dl_low:.4f}\n')
    print(f'best-fit l = {l_pixels:.1f} pixels\n')
    
    vsini = bestfit_params['vsini'][1]
    # add the target as a suffix to log_a and log_l
    # include vsini next to the targets name
    GP_eq_block[target] = f'\\text{{{target}}}& \quad (v\\sin i \\approx {vsini:.1f} \\text{{ km s}}^{{-1}})\\\\ \n'
    GP_eq_block[target] += 'a & = ' + f"{a:.2f}^{{+{da_up:.2f}}}_{{-{da_low:.2f}}} \\\\ \n"
    GP_eq_block[target] += 'l & = ' + f"{l:.4f}^{{+{dl_up:.4f}}}_{{-{dl_low:.4f}}}"
    GP_eq_block[target] += '\\approx ' + f"{l_pixels:.1f} \\text{{ pixels}} \\\\ \n"
    
    
    j = -1
    for j_i, (key, val) in enumerate(bestfit_params.items()):
        if key in ignore_params:
            continue
        
        j += 1
        if abs(val[2]-val[1]) < 5e-3:
            table[j] += [f"${val[1]:.3f}^{{+{val[2]-val[1]:.3f}}}_{{-{val[1]-val[0]:.3f}}}$"]
        elif abs(val[2]-val[1]) < 5e-2:
            table[j] += [f"${val[1]:.2f}^{{+{val[2]-val[1]:.2f}}}_{{-{val[1]-val[0]:.2f}}}$"]
        elif abs(val[2]-val[1]) < 5e-1:
            table[j] += [f"${val[1]:.1f}^{{+{val[2]-val[1]:.1f}}}_{{-{val[1]-val[0]:.1f}}}$"]
        else:
            table[j] += [f"${val[1]:.0f}^{{+{val[2]-val[1]:.0f}}}_{{-{val[1]-val[0]:.0f}}}$"]
       


# Generate LaTeX table
# increase spacing between rows
latex_table = tabulate(table, headers=headers, tablefmt="latex_raw", floatfmt=".2f")
latex_table = latex_table.replace("\\begin{tabular}", "\\renewcommand{\\arraystretch}{1.5}\n\\begin{tabular}")

# replace keys
replace = {
    '$\\log\\ l$' : '$\\log\\ l$ [nm]',
    '$\\log\\ g$' : '$\\log\\ g$ [cm s$^{-2}$]',
    '$v\\ \\sin\\ i$':'$v\\ \\sin\\ i$ [km s$^{-1}$]', 
    '$v_\\mathrm{rad}$':'$v_\\mathrm{rad}$ [km s$^{-1}$]',
    '$T_0$' : '$T_0$ [K]',
              }

for key, val in replace.items():
    latex_table = latex_table.replace(key, val)
    
    
# make GP_eq
GP_eq = '\\begin{align*}\n'
targets_id = ['J0856', 'J1200', 'TWA28']
for target in targets_id:
    GP_eq += GP_eq_block[target]

GP_eq += '\end{align*}'
# save equation
with open(out_path / "equations/GP_eq.tex", "w") as f:
    f.write(GP_eq)
    print(f'Equation saved to {out_path / "equations/GP_eq.tex"}')
print(GP_eq)

# Print or save the LaTeX table
# print(latex_table)
# Alternatively, save to a file
# wrap around a table* environment
# add label for table 
caption = "\\caption{Summary of retrieved parameters and their uncertainties. The table includes the Gaussian Process parameters, physical properties (surface gravity and rotational velocity), and volume mixing ratios (VMR) of the chemical species. Uncertainties are represented as $1\sigma$ intervals.}\n"
label = "\\label{tab:retrieval_results}\n"
latex_table = "\\begin{table*}\n\centering\n" + latex_table + caption + label + "\n\end{table*}"
# save to a file
with open(out_path / "tables/table_retrieval.tex", "w") as f:
    f.write(latex_table)
    print(f'Table saved to {out_path / "tables/table_retrieval.tex"}')
