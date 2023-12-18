import numpy as np
import pathlib
import json

from tabulate import tabulate



path = pathlib.Path('/home/dario/phd/retrieval_base')
targets = dict(J1200='freechem_8', 
               TWA28='freechem_4', 
               J0856='freechem_3')

# Load priors of the parameters
config_path = '/home/dario/phd/retrieval_base/J0856/retrieval_outputs/freechem_3/test_data/'
config_file = f'{config_path}config_freechem.txt'
with open(config_file, 'r') as file:
    load_file = json.load(file)
free_params = load_file['free_params']
names_params = list(free_params.keys())
# Convert the parameters to a list for tabulate
# priors_table = [(label, f"[{low}, {high}]", desc) for label, (low, high), desc in free_params.values()]
table = [[val[1], f"[{val[0][0]}, {val[0][1]}]"] for key, val in free_params.items()]
# Define the headers
headers = ["Parameter", "Prior Range"]

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
    for j, (key, val) in enumerate(bestfit_params.items()):
        
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

# Print or save the LaTeX table
# print(latex_table)
# Alternatively, save to a file

out_path = pathlib.Path('/home/dario/phd/Hot_Brown_Dwarfs_Retrievals/tables')
with open(out_path / "table_retrieval.tex", "w") as f:
    f.write(latex_table)
