import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import json
import retrieval_base.auxiliary_functions as af
plt.style.use('default')
plt.style.use(['sans'])
def load_json(file):    
    """
    Load and parse a JSON file.

    Parameters:
    file (str): The path to the JSON file to be loaded.

    Returns:
    dict: The parsed contents of the JSON file as a dictionary.
    """
    with open(file, 'r') as f:
        return json.load(f)
    
    

base_path = '/home/dario/phd/retrieval_base/'
nat_path = '/home/dario/phd/nat/figures/'


testing = False

data = load_json(Path(base_path) / 'paper/data/isotopologue_results.json')

path = Path(__file__).parent / 'data/tsuji2016_table3.csv'

tsuji = pd.read_csv(path)

fig, ax = plt.subplots(1,1, figsize=(5, 5))


for k in data.keys():
    median = data[k]['log_12CO/13CO']['median']
    error = np.abs(np.array(data[k]['log_12CO/13CO']['1sigma']) - median)
    # print(f'{k}: {median:.2f}')
    
    k_tsuji = k.replace('gl', 'GJ ')
    if k_tsuji in tsuji['object_name'].values:
        tsuji_val = tsuji[tsuji["object_name"] == k_tsuji]["log_12C/13C"].values[0]
        lower_limit = tsuji[tsuji["object_name"] == k_tsuji]["is_lower_limit"].values[0]
        
        # add text offset to clearly read label next to the point
        offset = 0.05 if lower_limit else -0.02
        ax.text(tsuji_val + offset, median, k_tsuji, fontsize=12,
                ha='left' if lower_limit else 'right',
                va='bottom',
                color='k' if not lower_limit else 'red',
                )

        if not lower_limit:
            print(f'Tsuji: {k_tsuji}: {tsuji_val:.2f}, {median:.2f}')
            # ax.scatter(tsuji_val, median, color='k', marker='o')
            ax.errorbar(tsuji_val,
                        median,
                        yerr=np.array(error)[:,None],
                        color='k', marker='o')
        else:
            print(f'Tsuji: {k_tsuji}: >{tsuji_val:.2f}, {median:.2f}')
            # show as error bar with lower limit
            ax.scatter(tsuji_val, median, color='red', marker='^')
    
# draw line with slope 1 and intercept 0
xlim = [1.4, 2.6]
ax.plot(xlim, xlim, color='k', linestyle='--')
ax.set_xlabel('Tsuji (2016)')
ax.set_ylabel('This work')
ax.set_xlim(xlim)
ax.set_ylim(xlim)
plt.savefig(Path(nat_path) / 'carbon_isotope_comparison_tsuji2016.pdf', dpi=300, bbox_inches='tight')
# save as png too
plt.savefig(Path(nat_path) / 'png/carbon_isotope_comparison_tsuji2016.png', dpi=300, bbox_inches='tight')
plt.show()
