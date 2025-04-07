"""Read results and generate latex TABLE for the paper with the the parameters of the retrieval."""
import numpy as np
import matplotlib.pyplot as plt
# set fontsize to 16
plt.style.use('/home/dario/phd/retsupjup/GQLupB/paper/gqlupb.mplstyle')
# plt.rcParams.update({'font.size': 16})

import pathlib
import os

from tabulate import tabulate

from retsupjup.config import Config
import retsupjup.auxiliary_functions as aux
from retsupjup.auxiliary_functions import report_value
from retsupjup.retrieval import Retrieval
    
        
        
target = 'GQLupB'
cwd = str(os.getcwd())
if target not in cwd:
    os.chdir(cwd+f'/{target}')
path_paper = pathlib.Path('/home/dario/phd/gqlupb_paper/')
# path_figs = path_paper / 'figures'
path_eq = path_paper / 'equations'

# run = 'asympt_4'
# run = 'PT_7_akima_diabatic_10sknots'
run = 'final_DG'
run_dir = pathlib.Path(f'/home/dario/phd/retsupjup/GQLupB/retrieval_outputs/{run}')


conf = Config(run=run)('config_freechem.txt', config_path=run_dir/'test_data')
ret = Retrieval(conf=conf, evaluation=True)

bestfit_params, posterior = ret.PMN_analyzer(return_posterior=True)
ret.get_PT_mf_envelopes(posterior=posterior)

posterior_dict = dict(zip(ret.Param.param_keys, posterior.T))
quantiles = np.quantile(posterior, [0.16, 0.5, 0.84], axis=0)
bestfit_params_dict = dict(zip(ret.Param.param_keys, bestfit_params))

bestfit_params_q = dict(zip(ret.Param.param_keys, quantiles.T))


descriptions = {
                
                # 'R_p': 'radius of the planet',
                'log_12CO': 'log mixing ratio of \\twelveCO',
                'log_13CO': 'log mixing ratio of \\thirteenCO',
                'log_H2O': 'log mixing ratio of $\mathrm{H_2 O}$',
                'log_H2O_181':'ignore...',
                'log_HF': 'log mixing ratio of HF',
                'log_Na': 'log mixing ratio of Na',
                'log_Ca': 'log mixing ratio of Ca',
                'log_Ti': 'log mixing ratio of Ti',
                'log_g': 'log surface gravity',
                # 'epsilon_limb': 'limb darkening coefficient',
                'vsini': 'projected rotational velocity',
                'rv': 'radial velocity',
                'dlnT_dlnP_0': 'temperature gradient at $P_0=10^2$ bar',
                'dlnT_dlnP_1': 'temperature gradient at $P_\mathrm{RCE}+2\Delta P_\mathrm{bot}$',
                'dlnT_dlnP_2': 'temperature gradient at $P_\mathrm{RCE}+\Delta P_\mathrm{bot}$',
                'dlnT_dlnP_RCE': 'temperature gradient at $P_\mathrm{RCE}$',
                'dlnT_dlnP_3': 'temperature gradient at $P_\mathrm{RCE}-\Delta P_\mathrm{top}$',
                'dlnT_dlnP_4': 'temperature gradient at $P_\mathrm{RCE}-2\Delta P_\mathrm{top}$',
                'dlnT_dlnP_5': 'temperature gradient at $P_5=10^{-5}$ bar',
                'dlog_P_1': 'log pressure shift of lower PT knots',
                'dlog_P_3': 'log pressure shift of upper PT knots',
                'log_P_RCE': 'log pressure of RCE',
                'T_0': 'temperature at $10^{2}$ bar',
                'X': 'airmass',
                'log_a': 'GP amplitude',
                'log_l': 'GP lengthscale',
                }


free_params = {k: conf.free_params[k] for k in descriptions.keys()}

class Table:
    
    def __init__(self, free_params, descriptions):
        
        self.free_params = free_params # dictionary with prior ranges
        self.descriptions = descriptions
        
        self.table = []
        
        
    def print_prior(self, prior, decimals=2):
        # print(f'Prior: {prior[0]:.{decimals}f} - {prior[1]:.{decimals}f}')
        # create string for table
        return f'[{prior[0]:.{decimals}f}, {prior[1]:.{decimals}f}]'
        
    def print_bestfit(self, bestfit, decimals=2):
        # bestfit contains 3 values corresponding to the 16th, 50th and 84th percentiles
        v_low = bestfit[1] - bestfit[0]
        v_high = bestfit[2] - bestfit[1]
        return f'${bestfit[1]:.{decimals}f}^{{+{v_high:.{decimals}f}}}_{{-{v_low:.{decimals}f}}}$'

    def make_descriptions_priors(self, ignore_params=[]):
        
        self.ignore_params = ignore_params
        for (key, val) in self.free_params.items():
            if key in self.ignore_params:
                continue
            
            dec = 2
            
            if key in ['T_0']:
                dec = 0
            if key.startswith('log_') and key not in ['log_a', 'log_l', 'log_P_RCE']:
                dec = 1
            
            self.table.append([val[1], self.descriptions[key], self.print_prior(val[0], decimals=dec)])
        return self
    
    def add_bestfit_values(self, bestfit_params_dict):
        
        # assert len(bestfit_params_dict) == len(self.table), f'Length of bestfit_params_dict ({len(bestfit_params_dict)}) does not match the length of the table ({len(self.table)})'
        # sort according to the order of the table
        bestfit_d = {k: bestfit_params_dict[k] for k in self.free_params.keys()}
        
        row = -1
        for j, (k, v) in enumerate(bestfit_d.items()):
            if k in self.ignore_params:
                continue
            
            if k not in self.free_params.keys():
                print(f'Warning: {k} not in free_params')
                continue
            
            row += 1
            # choose the number of decimals according to the difference between the 16th and 84th percentiles
            diff = abs(v[2] - v[0])
            if diff < 5e-3:
                decimals = 4
            elif diff < 1e-2:
                decimals = 3
            elif diff < 1e-1:
                decimals = 2
            elif diff < 1e0:
                decimals = 1
            else:
                decimals = 0
            
            if k in ['log_g', 'dlnT_dlnP_0', 'dlnT_dlnP_5', 'dlog_P_3']:
                decimals = 2
            if k in ['log_Ti']:
                decimals = 1
            
            # find row with matching key
            self.table[row] += [self.print_bestfit(v, decimals=decimals)]
            
        return self
        

    def make_tex(self, 
                 headers=["Parameter", "Description", "Prior Range", 'Best-Fit'], 
                 floatfmt=".2f",
                 stretch=1.5):
        self.tex = tabulate(self.table, headers=headers, floatfmt=floatfmt, tablefmt='latex_raw')
        if stretch > 0.0:
            self.tex = self.tex.replace("\\begin{tabular}", "\\renewcommand{\\arraystretch}{"+str(stretch)+"}\n\\begin{tabular}")
            
        return self
    
    def replace_keys(self, dictionary):
        for key, value in dictionary.items():
            self.tex = self.tex.replace(key, value)
        return self
    
    def add_caption(self, caption):
        self.caption = caption
        return self
    def add_label(self, label):
        self.label = label
        return self
    
    def save(self, filename):
        save_table = "\\begin{table}\n\\centering\n" + self.tex + "\n\\caption{" + self.caption + "}\n\\label{" + self.label + "}\n\\end{table}"
        with open(filename, "w") as f:
            f.write(save_table)
            
        print(f'- Table saved to {filename}')
        return self
    
if __name__ == '__main__':
    
    tab = Table(free_params, descriptions)
    tab.make_descriptions_priors(ignore_params=['log_H2O_181'])
    tab.add_bestfit_values(bestfit_params_q)
    tab.add_caption('Prior ranges and best-fit values of the free parameters in the retrieval.')
    tab.add_label('tab:free_params')
    tab.make_tex()
    tab.replace_keys({'$\\nabla_{T,RCE}$' : '$\\nabla_{T,\\mathrm{RCE}}$',
                      '$\\Delta\\log\\ P_1$' : '$\\Delta\\log\\ P_\mathrm{bot}$',
                      '$\\Delta\\log\\ P_3$' : '$\\Delta\\log\\ P_\mathrm{top}$',
                      })
    tab.save(path_paper / 'tables' / 'table_free_params.tex')