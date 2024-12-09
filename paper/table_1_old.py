import pandas as pd

def generate_latex_table(csv_file, output_file):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)
    
    attrs_dict = {'Star': 'Star', 
             'SpT': 'Spectral Type',
             'Distance (pc)': 'Distance / pc',
             'M/M_sun': r'M/M$_{\odot}$',
             'log g': 'log g',
             'Teff (K)': r'T$_{\rm eff}$ / K',
             '[M/H]': '[M/H]', 
             'Period (days)':r'P$_{\rm rot}$ / day'}
    attrs  = list(attrs_dict.keys())
    labels = list(attrs_dict.values())
    # Begin the LaTeX table
    latex_table = r'''\begin{table*}[ht]
\centering
\begin{tabular}{lccccccc}
\hline
'''
    latex_table += ' & '.join(labels) + r'\\' + '\n' + r'\hline' + '\n'

    # Loop through the DataFrame and format each row for LaTeX
    attrs = ['Star', 'SpT', 'Distance (pc)', 'M/M_sun', 'log g', 'Teff (K)', '[M/H]', 'Period (days)']
    for index, row in df.iterrows():
        for attr in attrs:
            # check if it exists
            if attr in row:
                if "+-" in row[attr]:
                    latex_table += "$" + row[attr].replace('+-', '\pm') + '$ & '
                else:
                
                    latex_table += f"{row[attr]}" + ' & '
                    
                
            else:
                print(f'Attribute {attr} not found in row {index}')
                
        # End the row
        latex_table = latex_table[:-2] + r'\\' + '\n'

    # End the LaTeX table
    latex_table += r'''\hline
\end{tabular}
\caption{Star Data Table with Periods}
\end{table*}
'''

    # Write the LaTeX table to the output file
    with open(output_file, 'w') as f:
        f.write(latex_table)
    
    print(f"LaTeX table written to {output_file}")

# Example usage
# generate_latex_table('star_data.csv', 'star_table_with_periods.tex')


# read csv file

csv_file = 'paper/data/fundamental_parameters.csv'
df = pd.read_csv(csv_file)

generate_latex_table(csv_file, 'paper/latex/tables/fundamental_parameters_table.tex')

print(f' TODO:  add references as last column!! with (1), (2) and legend below the table')