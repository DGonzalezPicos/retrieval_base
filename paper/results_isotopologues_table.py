""" Create Latex Table with isotopologue bayesian and CCF metrics """

import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib


import json
import retrieval_base.auxiliary_functions as af

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
nat_path = '/home/dario/phd/nat/tables/'


testing = False

data = load_json(pathlib.Path(base_path) / 'paper/data/isotopologue_results.json')
evidence = np.loadtxt(pathlib.Path(base_path) / 'paper/data' / f'lnB_sigma13CO.dat', dtype=object)

evidence_dict = {}

targets = list(data.keys())

run = 'fc5'
species = ['13CO', 'C18O', 
        #    'H2O_181', 'C17O',
           ]
n = len(species)

plot = False

if plot:
    fig, ax = plt.subplots(1, n, figsize=(n*3, 3), sharex=True, gridspec_kw={'wspace': 0.05})

ccf_snr = {}

for i, target in enumerate(targets):
    print(f'** {target} **')
    
    run_dir = pathlib.Path(base_path) / target / 'retrieval_outputs' / run

    ccf_snr[target] = {}
    evidence_dict[target] = {}
    for s, sp in enumerate(species):
        rv, ccf, acf = np.loadtxt(run_dir / f'test_plots/CCF/RV_CCF_ACF_{sp}.txt', unpack=True)
        if plot:
            ax[s].plot(rv, ccf)
        snr_peak = np.max(ccf)
        rv_peak = rv[np.argmax(ccf)]
        # print(f' {sp}: {snr_peak:.1f}, {rv_peak:.1f}')
        
        if (abs(rv_peak) < 5.0) and (snr_peak > 2.0):
            ccf_snr[target][sp] = snr_peak
        else:
            ccf_snr[target][sp] = np.nan
            
        evidence = np.loadtxt(pathlib.Path(base_path) / 'paper/data' / f'lnB_sigma{sp}.dat', dtype=object)
        evidence_dict[target][sp] = evidence[i,1:]

        
    print()
    # if testing:
    #     break
    
if plot:
    ax[0].set_xlim(-200.0, 200.0)

    for axi in ax:
        axi.axvline(0, color='k', lw=0.5)
    plt.show()


## Create Latex Table with \begin{sideways}

# tex = '\\begin{sidewaystable}\n'
# create regular table, increase vertical spacing of rows


tex = '\\begin{table}\n'
# increase vertical spacing of rows
tex += '\\renewcommand{\\arraystretch}{1.3}\n'
tex += '\\caption{Retrieval results and detection significances of the CO isotopologues.}\n'
tex += '\\label{tab:isotopologue_ccf_snr}\n'
# tex += '\\begin{tabular*}{\\textheight}{@{\\extracolsep\\fill}lcccccccc}\n'
tex += '\\begin{tabular*}{\\textwidth}{lcccccccccc}\n'
tex += '\\toprule\n'
tex += '& \\multicolumn{4}{@{}c@{}}{\\rule[0.0ex]{40pt}{0pt}$^{13}$CO}& \\multicolumn{6}{@{}c@{}}{\\rule[0.0ex]{44pt}{0pt}C$^{18}$O} \\\\' 
# tex += '\\cmidrule{2-5}\\cmidrule{7-10}\n'
# tex += '\\midrule\n'
tex += '\\cmidrule{3-6}\\cmidrule{8-11}\n'



tex += 'Gl & &'
for i in range(len(species)):
    if i == 0:
        tex += '$\log{} ^{12}$C/$^{13}$C'
    else:
        tex += '$\log{} ^{16}$O/$^{18}$O'
    
    tex += '& $\\ln B\\footnotemark[1]$ & $\\sigma$ & CCF\\footnotemark[2]'
    if i == 0:
        tex += ' & &'

tex += '\\\\\n'
# tex += '\\midrule\n'
tex += '\\cmidrule{1-1}\\cmidrule{3-6}\\cmidrule{8-11}\n'


iso_dict = {'13CO': 'log_12CO/13CO',
            'C18O': 'log_12CO/C18O',
            'H2O_181': 'log_H2O/H2O_181',
}
            

for target in targets:
    tex += target.replace('gl', '') + ' & &'
    for s, sp in enumerate(species):
        d = data[target][iso_dict[sp]]
        med = d['median']
        erru = d['1sigma'][1] - med
        errl = med - d['1sigma'][0]
        
        ed = evidence_dict[target][sp]
        sigma = float(ed[1])
        
        if sigma >= 3.0:
        
            tex += f'${med:.2f}' + '^{+' + f'{erru:.2f}' + '}_{-' + f'{errl:.2f}' + '}$'
        else:
            # show lower limit
            tex += f'$ >{med+erru:.2f}$'
        # tex += f'{med:.2f}'
        tex += ' & '
        
        # show map
        maxp = d['MAP']
        # tex += f'{maxp:.2f} & '
        
        # show lnB
        tex += f'{float(ed[0]):.1f}' + ' & '
        
        # show sigma
        if sigma >= 10.0:
            tex += '$> 10$ &'
        elif sigma <= 1.0:
            tex += '-- &' 
        
        else:
            tex += f'{sigma:.1f} &'
        
        # show CCF SNR
        tex += f'{ccf_snr[target][sp]:.1f}'
        
        
        if s < len(species) - 1:
            tex += ' &  &'
            
    
        
        
    tex += '\\\\\n'
    if testing:
        break
        
        
# replace nan with --

tex = tex.replace('nan', '--')
tex += '\\botrule\n'
tex += '\\end{tabular*}\n'
# tex += '\\end{sidewaystable}'
tex += '\\footnotetext[1]{Logarithm of the Bayes factor between the fiducial model and a model without the given species.}\n'
tex += '\\footnotetext[2]{Signal-to-noise ratio of the cross-correlation function. Missing values denote non-detections.}\n'
tex += '\\end{table}'

with open(pathlib.Path(nat_path) / 'isotopologue_ccf_snr.tex', 'w') as f:
    f.write(tex)

print(f'Wrote table to {nat_path}isotopologue_ccf_snr.tex')
