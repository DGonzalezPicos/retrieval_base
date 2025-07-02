"""Generate LaTeX table comparing retrieval parameters between TWA 27A and TWA 28."""
from pathlib import Path
import numpy as np
from tabulate import tabulate
import os
import json
from retrieval_base.config import Config
from retrieval_base.retrieval import Retrieval
import retrieval_base.auxiliary_functions as aux

path = Path('/home/dario/phd/retrieval_base')

class ComparisonTable:
    """Class to generate LaTeX tables comparing retrieval parameters between targets."""
    
    def __init__(self, targets: list[str], runs: dict[str, str], replace_names: dict[str, str], descriptions: dict[str, str]):
        """Initialize the comparison table.
        
        Args:
            targets: List of target names to compare
            runs: Dictionary mapping target names to their retrieval run IDs
            replace_names: Dictionary mapping parameter keys to their LaTeX names
            descriptions: Dictionary mapping parameter keys to their descriptions
        """
        self.targets = targets
        self.runs = runs
        self.replace_names = replace_names
        self.descriptions = descriptions
        self.table = []
        
    def print_prior(self, prior: tuple[float, float], decimals: int = 2,
                    gaussian: bool = False) -> str:
        """Format prior range for LaTeX table.
        
        Args:
            prior: Tuple of (min, max) prior values
            decimals: Number of decimal places to show
            
        Returns:
            Formatted string for LaTeX table
        """
        distribution = r'$\mathcal{N}$' if gaussian else r'$\mathcal{U}$'
        return f'{distribution}({prior[0]:.{decimals}f}, {prior[1]:.{decimals}f})'
        
    def print_bestfit(self, bestfit: tuple[float, float, float], decimals: int = 2,
                      show_sign: bool = True,
                      is_lower_limit: bool = False,
                      is_upper_limit: bool = False) -> str:
        """Format best-fit value with uncertainties for LaTeX table.
        
        Args:
            bestfit: Tuple of (16th, 50th, 84th) percentile values
            decimals: Number of decimal places to show
            
        Returns:
            Formatted string for LaTeX table
        """
        v_low = bestfit[1] - bestfit[0]
        v_high = bestfit[2] - bestfit[1]
        sign = ''
        if show_sign:
            sign = '+' if bestfit[1] > 0 else ''
        if is_lower_limit:
            return f' < {sign}{bestfit[0]:.{decimals}f}'
        elif is_upper_limit:
            return f' > {sign}{bestfit[2]:.{decimals}f}'
        else:
            return f'${sign}{bestfit[1]:.{decimals}f}^{{+{v_high:.{decimals}f}}}_{{-{v_low:.{decimals}f}}}$'
    
    def load_data(self, target: str, cache: bool = True) -> dict:
        """Load data for a target.
        
        Args:
            target: Target name
            cache: Whether to use cached results
            
        Returns:
            Dictionary of best-fit parameters
        """
        cwd = os.getcwd()
        if target not in cwd:
            os.chdir(path / target)
            print(f'Changed to {target} directory')
        self.conf = Config(target=target, run=self.runs[target])('config_jwst.txt')
        
        bestfit_params_q_file = self.conf.output_path / 'bestfit_params_q.json'
        if cache and bestfit_params_q_file.exists():
            print(f'Loading bestfit_params_q from {bestfit_params_q_file}')
            with open(bestfit_params_q_file, 'r') as f:
                bestfit_params_q = json.load(f)
        else:
            print(f'Getting bestfit_params_q for {target} from retrieval object')
            ret = Retrieval(conf=self.conf, evaluation=True)
            
            _, posterior = ret.PMN_analyze()
            quantiles = np.quantile(posterior, [0.16, 0.5, 0.84], axis=0)
            bestfit_params_q = dict(zip(ret.Param.param_keys, quantiles.T.tolist()))
            # save bestfit_params_q to file for quick access
            with open(bestfit_params_q_file, 'w') as f:
                json.dump(bestfit_params_q, f)
        
        return bestfit_params_q

    def sort_parameters(self, params: list[str], use_descriptions: bool = True) -> list[str]:
        """Sort parameters according to the order defined in descriptions dictionary.
        
        The order is determined by the grouping comments in the descriptions dictionary:
        1. Atmospheric structure parameters
        2. Temperature-pressure profile gradients
        3. Disk parameters
        4. Data processing parameters
        5. Molecular abundances and ratios
        6. Physical parameters
        
        Args:
            params: List of parameter names to sort
            use_descriptions: Whether to use the descriptions dictionary to determine the order
        Returns:
            Sorted list of parameter names
        """
        if use_descriptions:
            ordered_params = [p for p in self.descriptions if p in params]
        else:
            # Define parameter groups and their order
            param_groups = {
                'structure': ['R_p', 'log_g', 'T_0', 'log_P_RCE', 'dlog_P_1', 'dlog_P_3', 'rv'],
                'gradients': ['dlnT_dlnP_0', 'dlnT_dlnP_1', 'dlnT_dlnP_2', 'dlnT_dlnP_3', 
                            'dlnT_dlnP_4', 'dlnT_dlnP_5', 'dlnT_dlnP_RCE'],
                'disk': ['log_R_d', 'T_d', 'log_R_jup', 'log_N_mol', 'log_T_ex', 'rv_disk'],
                'data': ['log_l_G', 'b_g140h', 'b_g235h', 'b_g395h'],
                'molecules': [param for param in params if param.startswith(('alpha_', 'log_'))
                            and param not in ['log_R_d', 'log_R_jup', 'log_N_mol', 'log_T_ex', 
                                            'log_l_G', 'log_P_RCE', 'log_g', 'log_Hminus']],
                'physical': ['log_Hminus']
            }
            
            # Create ordered list of parameters
            ordered_params = []
            for group in param_groups.values():
                # Add parameters that exist in both the input list and the group
                ordered_params.extend([p for p in group if p in params])
                
            # Add any remaining parameters that weren't in any group
            remaining = [p for p in params if p not in ordered_params]
            if remaining:
                print(f"Warning: Some parameters were not in any group: {remaining}")
                ordered_params.extend(remaining)
            
        return ordered_params

    def make_table(self, ignore_params: list[str] = []) -> 'ComparisonTable':
        """Generate the comparison table data.
        
        Args:
            ignore_params: List of parameter keys to exclude from table
            
        Returns:
            Self for method chaining
        """
        # Get data for each target
        target_data = {target: {} for target in self.targets}
        for target in self.targets:
            bestfit_params_q = self.load_data(target, cache=True)
            
            self.free_params_keys = list(self.conf.free_params.keys())
            
            target_data[target]['priors'] = {}
            target_data[target]['names'] = {}
            target_data[target]['descriptions'] = {}
            for k,v in self.conf.free_params.items():
                if k in ignore_params:
                    print(f'Ignoring {k}')
                    continue
                target_data[target]['priors'][k] = v[0]
                target_data[target]['names'][k] = self.replace_names.get(k, v[-1])
                target_data[target]['descriptions'][k] = self.descriptions.get(k, v[-1])
            target_data[target]['bestfit'] = bestfit_params_q

        # Sort parameters according to defined order
        sorted_params = self.sort_parameters(
            [k for k in self.free_params_keys if k not in ignore_params],
            use_descriptions=True
            )
        
        print(sorted_params)
        decimals = {
            '0': ['T_0'],
            '1': [k for k in sorted_params if k.startswith('log_')],
            '2': ['log_N_mol','log_AlO','log_g', 'log_P_RCE', 'log_Hminus', 'log_12CO/13CO', 'log_12CO/C18O', 'log_12CO/C17O', 'log_H2O/H2O_181'],
            '3': ['log_T_ex','R_p', 'log_R_d', 'log_R_jup', 'log_l_G', 'log_CH4'] +
            [k for k in sorted_params if k.startswith('dlnT_dlnP_')]
            
        }
        prior_decimals = {
            '0': ['rv', 'T_0','T_d', 'rv_disk',
                  'log_N_mol', 'log_R_jup', 'log_R_d', 'log_CH4', 'log_AlO','log_H2S', 'log_Hminus']+
            [k for k in sorted_params if k.startswith('alpha_')]+
            [k for k in sorted_params if k.startswith('b_')],
            '1': ['log_g','R_p', 'log_T_ex', 'log_P_RCE','log_l_G','dlog_P_1','dlog_P_3',
                  'log_12CO/13CO', 'log_12CO/C18O', 'log_12CO/C17O', 'log_H2O/H2O_181'],
            
            '2': []+
             [k for k in sorted_params if k.startswith('dlnT_dlnP_')],
             
            # '3': ['log_T_ex','R_p', 'log_R_d', 'log_R_jup', 'log_l_G', 'log_CH4']
        }
        prior_decimals_rev = {}
        for k, v in prior_decimals.items():
            for param in v:
                prior_decimals_rev[param] = k
        
        lower_limits = {
            'TWA27A': ['dlnT_dlnP_0'],
            'TWA28': ['dlnT_dlnP_0']
        }
        upper_limits = {
            'TWA27A': ['log_T_ex'],
            'TWA28': []
        }
        
        decimals_rev = {}
        for k, v in decimals.items():
            for param in v:
                decimals_rev[param] = k
        # Build table rows
        for key in sorted_params:
            dec = decimals_rev.get(key, 2)
                
            # Get prior from first target (assuming same priors)
            prior_str = self.print_prior(target_data[self.targets[0]]['priors'][key], 
                                         decimals=prior_decimals_rev.get(key, dec),
                                         gaussian=key in self.conf.gaussian_params)
            
            # Get bestfit values for each target
            bestfit_values = []
            for target in self.targets:
                if key in target_data[target]['bestfit']:
                    bestfit_values.append(self.print_bestfit(target_data[target]['bestfit'][key], 
                                                             decimals=dec, 
                                                             show_sign=True,
                                                             is_lower_limit=key in lower_limits[target],
                                                                is_upper_limit=key in upper_limits[target]))
                else:
                    bestfit_values.append('---')
                    
            # Add row with parameter name, description, prior, bestfits, and notes
            self.table.append([
                target_data[self.targets[0]]['names'][key],
                target_data[self.targets[0]]['descriptions'][key],
                prior_str
            ] + bestfit_values)
            
        return self
        
    def make_tex(self, 
                 headers: list[str] = None,
                 floatfmt: str = ".2f",
                 stretch: float = 1.5,
                 split_table: str = None) -> 'ComparisonTable':
        """Generate LaTeX table.
        
        Args:
            headers: Column headers
            floatfmt: Format for float values
            stretch: Row spacing multiplier
            split_table: Parameter key to split the table into two tables
        Returns:
            Self for method chaining
        """
        if headers is None:
            headers = ["Parameter", "Description", "Prior Range"] + [f"{target}" for target in self.targets] + ["Notes"]
    
            
        
        
            
        self.tex = tabulate(self.table, headers=headers, floatfmt=floatfmt, tablefmt='latex_raw')
        if stretch > 0.0:
            self.tex = self.tex.replace("\\begin{tabular}", 
                                      f"\\renewcommand{{\\arraystretch}}{{{stretch}}}\n\\begin{{tabular}}")
        return self
    
    def replace_keys(self, dictionary: dict[str, str]) -> 'ComparisonTable':
        """Replace text in table with LaTeX formatted versions.
        
        Args:
            dictionary: Mapping of text to replace with LaTeX versions
            
        Returns:
            Self for method chaining
        """
        for key, value in dictionary.items():
            self.tex = self.tex.replace(key, value)
        return self
    
    def add_caption(self, caption: str) -> 'ComparisonTable':
        """Add table caption.
        
        Args:
            caption: Table caption text
            
        Returns:
            Self for method chaining
        """
        self.caption = caption
        return self
        
    def add_label(self, label: str) -> 'ComparisonTable':
        """Add table label.
        
        Args:
            label: Table reference label
            
        Returns:
            Self for method chaining
        """
        self.label = label
        return self
    
    def add_note(self, note: str) -> 'ComparisonTable':
        """Add table note.
        
        Args:
            note: Table note text
            
        Returns:
            Self for method chaining
        """
        self.note = note
        return self
    
    def save(self, filename: Path) -> 'ComparisonTable':
        """Save table to LaTeX file.
        
        Args:
            filename: Path to save table
            
        Returns:
            Self for method chaining
        """
        save_table = (
            "\\begin{table*}\n\\centering\n" + 
            "\n\\caption{" + self.caption + "}\n" + 
            self.tex + 
            "\n\\label{" + self.label + "}\n\\end{table*}\n"
            # self.note
        )
       
        with open(filename, "w") as f:
            f.write(save_table)
            
        print(f'- Table saved to {filename}')
        return self


def main():
    """Generate and save the comparison table."""
    # Define targets and their runs
    targets = ['TWA27A', 'TWA28']
    runs = {
        'TWA27A': 'freeslab_lbl10_G1G2G3_1',
        'TWA28': 'freeslab_lbl10_G1G2G3_1'
    }
    
    # Parameter descriptions with detailed explanations
    descriptions_1 = {
        # Atmospheric structure parameters
        'R_p': 'Radius in Jupiter radii',
         # Kinematic parameters
        'rv': 'Radial velocity',
        'log_g': 'Surface gravity of the atmosphere',
        'T_0': 'Temperature at the bottom of the atmosphere',
        'log_P_RCE': 'Pressure at radiative-convective equilibrium',
        'dlog_P_1': 'Pressure spacing for lower atmosphere layers',
        'dlog_P_3': 'Pressure spacing for upper atmosphere layers',
        
        # Temperature-pressure profile gradients
        'dlnT_dlnP_0': 'Temperature gradient at $P_0 = 100$ bar',
        'dlnT_dlnP_1': 'Temperature gradient at $P_1=P_{\\text{RCE}}-2 \\Delta P_{\\text{low}}$',
        'dlnT_dlnP_2': 'Temperature gradient at $P_2=P_{\\text{RCE}}-1 \\Delta P_{\\text{low}}$',
        'dlnT_dlnP_3': 'Temperature gradient at $P_3=P_{\\text{RCE}}+1 \\Delta P_{\\text{high}}$',
        'dlnT_dlnP_4': 'Temperature gradient at $P_4=P_{\\text{RCE}}+2 \\Delta P_{\\text{high}}$',
        'dlnT_dlnP_5': 'Temperature gradient at $P_5=10^{-5}$ bar',
        'dlnT_dlnP_RCE': 'Temperature gradient at $P_{\\text{RCE}}$',
        
        # Disk parameters
        'log_R_d': 'Effective radius of the blackbody emission',
        'T_d': 'Temperature of the blackbody emission',
        
        'log_R_jup': 'Effective radius of the slab model',
        'log_N_mol': 'Column density of slab model',
        'log_T_ex': 'Excitation temperature of slab model',
        'rv_disk': 'Radial velocity of disk emission',
        
        
         # Data processing parameters
        'log_l_G': 'Global correlation length',
        'b_g140h': 'Error scaling factor for G140H grating',
        'b_g235h': 'Error scaling factor for G235H grating',
        'b_g395h': 'Error scaling factor for G395H grating',
        
        # Molecular abundances and ratios
        'alpha_12CO': 'Deviation from chemical equilibrium for $^{12}$CO',
        'alpha_13CO': 'Deviation from chemical equilibrium for $^{13}$CO',
        'alpha_H2O': 'Deviation from chemical equilibrium for H$_2$O',
        'alpha_SiO': 'Deviation from chemical equilibrium for SiO',
        'alpha_CO2': 'Deviation from chemical equilibrium for CO$_2$',
        'alpha_TiO': 'Deviation from chemical equilibrium for TiO',
        'alpha_VO': 'Deviation from chemical equilibrium for VO',
        'alpha_FeH': 'Deviation from chemical equilibrium for FeH',
        'alpha_CrH': 'Deviation from chemical equilibrium for CrH',

        
        
    }
    descriptions_2 = {
        'alpha_NaH': 'Deviation from chemical equilibrium for NaH',
        'alpha_AlH': 'Deviation from chemical equilibrium for AlH',
        'alpha_HF': 'Deviation from chemical equilibrium for HF',
        'alpha_HCl': 'Deviation from chemical equilibrium for HCl',
        'alpha_Na': 'Deviation from chemical equilibrium for Na',
        'alpha_Ca': 'Deviation from chemical equilibrium for Ca',
        'alpha_K': 'Deviation from chemical equilibrium for K',
        'alpha_Mg': 'Deviation from chemical equilibrium for Mg',
        'alpha_Al': 'Deviation from chemical equilibrium for Al',
        'alpha_Fe': 'Deviation from chemical equilibrium for Fe',
        'alpha_Ti': 'Deviation from chemical equilibrium for Ti',
        'log_CH4': 'Abundance of CH$_4$',
        'log_H2S': 'Abundance of H$_2$S',
        'log_AlO': 'Abundance of AlO',
        'log_Hminus': 'Abundance of H$^{-}$ bound-free opacity',
        'log_12CO/13CO': 'Carbon isotope ratio $^{12}$C/$^{13}$C in CO',
        'log_12CO/C18O': 'Oxygen isotope ratio $^{16}$O/$^{18}$O in CO',
        'log_12CO/C17O': 'Oxygen isotope ratio $^{16}$O/$^{17}$O in CO',
        'log_H2O/H2O_181': 'Oxygen isotope ratio $^{16}$O/$^{18}$O in H$_2$O',
        
        

    }
    
    # define the ordering of the parameters according to the 'descriptions' dictionary

    
    replace_names = {
        # 'R_p' : '$R_{\\text{p}} / R_{\\text{jup}}$',
        'log_R_d' : '$\\log R_{\\text{d}} / R_{\\text{jup}}$',
        'log_R_jup' : '$\\log R_{\\text{slab}} / R_{\\text{jup}}$',
        'log_T_ex' : '$\\log T_{\\text{ex}} / \\text{K}$',
        'log_N_mol' : '$\\log N_{\\text{mol}} / \\text{cm}^{-2}$',
        'log_Hminus' : '$\\log \\text{H}^{-}$',
        'rv' : '$v_{\\text{rad}} / \\text{km s}^{-1}$',
        'rv_disk' : '$v_{\\text{disk}} / \\text{km s}^{-1}$',
        'log_l_G' : '$\\log l_{\\text{G}} / \\text{km s}^{-1}$',
        'dlog_P_1' : '$\\log \\Delta P_{\\text{low}} / \\text{bar}$',
        'dlog_P_3' : '$\\log \\Delta P_{\\text{high}} / \\text{bar}$',
        'log_P_RCE' : '$\\log P_{\\text{RCE}} / \\text{bar}$',
    }
        
        
        
    # Create and save table
    path_tables = Path('/home/dario/phd/twa2x_paper/tables')
    captions = [
        'Summary of the free parameters and the retrieved values with 1$\\sigma$ uncertainties. The prior ranges and the distributions used (uniform or normal) are indicated.',
        'Continued from Table 1.'
    ]
    for d, descriptions in enumerate([descriptions_1, descriptions_2]):
        tab = ComparisonTable(targets, runs, replace_names, descriptions)
        tab.make_table()
        tab.add_caption(captions[d])
        tab.add_label(f'tab:free_params_comparison_{d}')
        tab.make_tex(stretch=1.5)
        tab.save(path_tables / f'table_free_params_comparison_{d}.tex')


if __name__ == '__main__':
    main()
