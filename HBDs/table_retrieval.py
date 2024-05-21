import numpy as np
import pathlib
import json
import pickle
from tabulate import tabulate

from retrieval_base.chemistry import Chemistry
atomic_mass = {k:v[2] for k,v in Chemistry.species_info.items()}

def save_equation(equation, filename):
    with open(filename, "w") as f:
        f.write(equation)
        print(f'Equation saved to {filename}')
        

path = pathlib.Path('/home/dario/phd/retrieval_base')
targets = dict(J1200='final_full',
                TWA28='final_full',
                J0856='final_full',
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

descriptions = {
                
                # 'R_p': 'radius of the planet',
                'log_12CO': 'log mixing ratio of \\twelveCO',
                'log_13CO': 'log mixing ratio of \\thirteenCO',
                'log_H2O': 'log mixing ratio of $\mathrm{H_2^{16}O}$',
                'log_H2O_181': 'log mixing ratio of \\eighteenOwater',
                'log_HF': 'log mixing ratio of HF',
                'log_Na': 'log mixing ratio of Na',
                'log_Ca': 'log mixing ratio of Ca',
                'log_Ti': 'log mixing ratio of Ti',
                'log_g': 'log surface gravity',
                'epsilon_limb': 'limb darkening coefficient',
                'vsini': 'projected rotational velocity',
                'rv': 'radial velocity',
                'dlnT_dlnP_0': 'temperature gradient at $P_0=10^2$ bar',
                'dlnT_dlnP_1': 'temperature gradient at $P_1+\Delta P$',
                'dlnT_dlnP_2': 'temperature gradient at $P_2+\Delta P$',
                'dlnT_dlnP_3': 'temperature gradient at $P_3+\Delta P$',
                'dlnT_dlnP_4': 'temperature gradient at $P_4+\Delta P$',
                'dlnT_dlnP_5': 'temperature gradient at $P_5+\Delta P$',
                'dlnT_dlnP_6': 'temperature gradient at $P_6+\Delta P$',
                'dlnT_dlnP_7': 'temperature gradient at $P_7=10^{-5}$ bar',
                'dlog_P': 'log pressure shift of PT knots',
                'T_0': 'temperature at $10^{2}$ bar',
                'alpha': 'veiling factor at $\lambda = 1.90\mu$m',
                'beta': 'veiling power law exponent',
                'log_a': 'GP amplitude',
                'log_l': 'GP lengthscale',
                }

# Convert the parameters to a list for tabulate
# priors_table = [(label, f"[{low}, {high}]", desc) for label, (low, high), desc in free_params.values()]
# sort dictionary `free_params` to follow the same key order as in `descriptions`
free_params = {k: free_params[k] for k in descriptions.keys()}

ignore_params = ['R_p']
zero_decimal_keys = ['T_0']
one_decimal_keys = ['log_12CO', 'log_13CO', 'log_H2O', 'log_H2O_181', 'log_HF', 'log_Na', 'log_Ca', 'log_Ti', 'vsini']
table = []
for (key, val) in free_params.items():
    if key in ignore_params:
        continue
    print(f'key = {key}')
    if key in zero_decimal_keys:
        table.append([val[1], descriptions[key], f"[{val[0][0]:.0f}, {val[0][1]:.0f}]"])
    elif key in one_decimal_keys:
        table.append([val[1], descriptions[key], f"[{val[0][0]:.1f}, {val[0][1]:.1f}]"])
    else:
        table.append([val[1], descriptions[key], f"[{val[0][0]:.2f}, {val[0][1]:.2f}]"])
    
# Define the headers
headers = ["Parameter", "Description", "Prior Range"]

GP_eq_block = {}
C_O_eq_block = {}
vsini_eq_block = {}
C_ratio_eq_block = {}
O_ratio_eq_block = {}
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
    # sort dictionary `bestfit_params` by key as in `descriptions`
    bestfit_params = {k: bestfit_params[k] for k in descriptions.keys()}
    
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
    
    # C_O = 
    chem = pickle.load(open(retrieval_path / 'test_data/bestfit_Chem.pkl', 'rb'))
    C_O = chem.CO_posterior
    C_O_quantiles = np.quantile(C_O, [0.16, 0.5, 0.84])
    C_O_low = C_O_quantiles[1] - C_O_quantiles[0]
    C_O_up = C_O_quantiles[2] - C_O_quantiles[1]
    # add space to target
    target_s =' ' + target
    C_O_eq_block[target] = f'\mathrm{{C/O}}_\\text{{{target_s}}} & = ' + f"{C_O_quantiles[1]:.2f}^{{+{C_O_up:.2f}}}_{{-{C_O_low:.2f}}}"
    C_O_eq_block[target] += '\\\\ \n'
    
    vsini_low = bestfit_params['vsini'][1] - bestfit_params['vsini'][0]
    vsini_up = bestfit_params['vsini'][2] - bestfit_params['vsini'][1]
    vsini_eq_block[target] = f'v\\sin{{i}}_\\text{{{target_s}}} & = '
    # check for number of decimals
    if abs(vsini_up) < 5e-2 or abs(vsini_low) < 5e-2:
        vsini_eq_block[target] += f"{vsini:.2f}^{{+{vsini_up:.2f}}}_{{-{vsini_low:.2f}}}"
    else:
        vsini_eq_block[target] += f"{vsini:.1f}^{{+{vsini_up:.1f}}}_{{-{vsini_low:.1f}}}"
    vsini_eq_block[target] += '\\text{{ km s}}^{{-1}}'
    
    # C_ratio 
    if hasattr(chem, 'VMRs_posterior'):
        C_ratio = chem.VMRs_posterior['12_13CO']
    else:
        CO_12 = chem.mass_fractions_posterior['CO_high'] / atomic_mass['12CO'] 
        CO_13 = chem.mass_fractions_posterior['CO_36_high'] / atomic_mass['13CO']
        C_ratio = CO_12 / CO_13
    C_ratio_quantiles = np.quantile(C_ratio, [0.16, 0.5, 0.84])
    C_ratio_low = C_ratio_quantiles[1] - C_ratio_quantiles[0]
    C_ratio_up = C_ratio_quantiles[2] - C_ratio_quantiles[1]
    C_ratio_eq_block[target] = f'\mathrm{{\\textsuperscript{{12}}C/\\textsuperscript{{13}}C}}_\\text{{{target_s}}} & = '
    C_ratio_eq_block[target] += f"{C_ratio_quantiles[1]:.0f}^{{+{C_ratio_up:.0f}}}_{{-{C_ratio_low:.0f}}}"
    C_ratio_eq_block[target] += '\\\\ \n'
    
    O_ratio = chem.VMRs_posterior['H2_16_18O']
    O_ratio_quantiles = np.quantile(O_ratio, [0.16, 0.5, 0.84])
    O_ratio_low = O_ratio_quantiles[1] - O_ratio_quantiles[0]
    O_ratio_up = O_ratio_quantiles[2] - O_ratio_quantiles[1]
    if target == 'J1200':
        # only lower limit for J1200
        O_ratio_eq_block[target] = f'\mathrm{{\\textsuperscript{{16}}O/\\textsuperscript{{18}}O}}_\\text{{{target_s}}} & > '
        O_ratio_eq_block[target] += f"{O_ratio_quantiles[0]:.0f}"
    else:
        O_ratio_eq_block[target] = f'\mathrm{{\\textsuperscript{{16}}O/\\textsuperscript{{18}}O}}_\\text{{{target_s}}} & = '
        O_ratio_eq_block[target] += f"{O_ratio_quantiles[1]:.0f}^{{+{O_ratio_up:.0f}}}_{{-{O_ratio_low:.0f}}}"
    O_ratio_eq_block[target] += '\\\\ \n'
    
    
    
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
    '$\Delta\log\ P$' : '$\log\Delta P$ [bar]',
    '$\log\ \mathrm{H_2O}$' : '$\log\ \mathrm{H_2^{16}O}$',
    '$\alpha$' : '$r_0$',
    '$\beta$' : '$\\alpha$',
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
# with open(out_path / "equations/GP_eq.tex", "w") as f:
#     f.write(GP_eq)
#     print(f'Equation saved to {out_path / "equations/GP_eq.tex"}')
save_equation(GP_eq, out_path / "equations/GP_eq.tex")
# print(GP_eq)


# make C_O_eq --> #TODO: check this equation generation works
C_O_eq = '\\begin{align*}\n'
targets_id = ['J1200', 'J0856', 'TWA28']
for target in targets_id:
    C_O_eq += C_O_eq_block[target]
# save 
C_O_eq += '\end{align*}'
# save equation
with open(out_path / "equations/C_O_eq.tex", "w") as f:
    f.write(C_O_eq)
    print(f'Equation saved to {out_path / "equations/C_O_eq.tex"}')
    
# vsini equation, first one block for TWA28, then a block with J1200 and J0856
vsini_eq_TWA28 = vsini_eq_block['TWA28']
vsini_eq = '\\begin{align*}\n'
vsini_eq += vsini_eq_TWA28
vsini_eq += '\end{align*}'
# save equation
save_equation(vsini_eq, out_path / "equations/vsini_eq_TWA28.tex")

# vsini equation for J1200 and J0856
vsini_eq = '\\begin{align*}\n'
vsini_eq += vsini_eq_block['J1200']
vsini_eq += '\\\\ \n'
vsini_eq += vsini_eq_block['J0856']
vsini_eq += '\end{align*}'
save_equation(vsini_eq, out_path / "equations/vsini_eq_J1200_J0856.tex")

# 12C/13C equation block
C_ratio_eq = '\\begin{align*}\n'
for target in targets_id:
    C_ratio_eq += C_ratio_eq_block[target]
C_ratio_eq += '\end{align*}'
save_equation(C_ratio_eq, out_path / "equations/C_ratio_eq.tex")

# O ratio
O_ratio_eq = '\\begin{align*}\n'
for target in targets_id:
    O_ratio_eq += O_ratio_eq_block[target]
O_ratio_eq += '\end{align*}'
save_equation(O_ratio_eq, out_path / "equations/O_ratio_eq.tex")

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
