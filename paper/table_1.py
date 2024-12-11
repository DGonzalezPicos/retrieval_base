import pandas as pd
import numpy as np
base_path = '/home/dario/phd/retrieval_base/'
nat_path = '/home/dario/phd/nat/tables/'

refs = {'Mann15': 'mannHOWCONSTRAINYOUR2015',
        'Passegger2019': 'passeggerCARMENESSearchExoplanets2019',
        'Cristofari2022': 'cristofariEstimatingAtmosphericProperties2022',
        'Cristofari2023': 'cristofariMeasuringSmallscaleMagnetic2023',
        'GaiaEDR3': 'gaiacollaborationGaiaEarlyData2021',
}
        

def generate_latex_table(csv_file, output_file):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)
    
    attrs_dict = {'Star': 'Star', 
             'SpT': 'SpT',
             'Distance (pc)': 'Distance / pc',
             'M/M_sun': r'M/M$_{\odot}$',
            #  'log g': 'log g',
             'Teff (K)': r'T$_{\rm eff}$ / K',
             '[M/H]': '[M/H] dex', 
            #  'Ref': 'References',
            #  'Period (days)':r'P$_{\rm rot}$ / day'
            }
    attrs  = list(attrs_dict.keys())
    labels = list(attrs_dict.values())
    # Begin the LaTeX table
    latex_table = r'''\begin{table*}[ht]
\renewcommand{\arraystretch}{1.3}
\centering
'''
    latex_table += "\\caption{"
    latex_table += "Fundamental Parameters of the Stars in the Sample. "
    latex_table += "The spectral types, effective temperatures and masses are from \citep{"
    latex_table += f"{refs['Cristofari2022']}"
    latex_table += "} and references therein. "
    latex_table += "The distances from Gaia EDR3 \citep{"
    latex_table += f"{refs['GaiaEDR3']}"
    latex_table += "}. "
    latex_table += "The metallicities are from \citep{"
    latex_table += f"{refs['Cristofari2023']}"
    latex_table += "}}."                                                    
    #   {refs['Cristofari2022']}} and references therein. \
            # The distances from  Gaia EDR3 \citep{refs['GaiaEDR3']}. \
            #     The metallicities are from \citep{refs['Cristofari2023']}."
                
    # latex_table += r'''}
    
    latex_table += "\\label{tab:fundamental_parameters}\n"
    latex_table += "\\begin{tabular}{lccccccc}\\hline\n"
    
    latex_table += ' & '.join(labels) + r'\\' + '\n' + r'\hline' + '\n'

    # Loop through the DataFrame and format each row for LaTeX
    # attrs = ['Star', 'SpT', 'Distance (pc)', 'M/M_sun', 'Teff (K)', '[M/H]']
    for index, row in df.iterrows():
        print(f' row {index}', row)
        valid = row['Valid']
        if not valid:
            continue
        for attr in attrs:
            # check if it exists
            if attr in row:
                if isinstance(row[attr], str) and "+-" in row[attr]:
                    print(f' row[attr] = {row[attr]}')
                    v, err = row[attr].split('+-')
                    v = float(v)
                    err = float(err)
                    print(f' v = {v}, err = {err}')
                    # latex_table += "$" + row[attr].replace('+-', '\pm') + '$ & '
                    if attr == 'Teff (K)':
                        latex_table += "$" + f"{v:.0f} \pm {err:.0f}" + '$ & '
                    else:
                        latex_table += "$" + f"{v:.2f} \pm {err:.2f}" + '$ & '
                else:
                    try:
                        v = float(row[attr])
                        latex_table += "$" + f"{v:.2f}" + '$ & '
                    except:
                        latex_table += f"{row[attr]}" + ' & '
                    
                
            else:
                print(f'Attribute {attr} not found in row {index}')
                
        # End the row
        latex_table = latex_table[:-2] + r'\\' + '\n'

    # End the LaTeX table
    latex_table += r'''\hline
\end{tabular}
\end{table*}
'''

    # Write the LaTeX table to the output file
    with open(output_file, 'w') as f:
        f.write(latex_table)
    
    print(f"LaTeX table written to {output_file}")

# Example usage
# generate_latex_table('star_data.csv', 'star_table_with_periods.tex')


# read csv file

csv_file = f'{base_path}paper/data/fundamental_parameters.csv'
df = pd.read_csv(csv_file)

generate_latex_table(csv_file, f'{nat_path}/fundamental_parameters_table.tex')

print(f' TODO:  add references as last column!! with (1), (2) and legend below the table')