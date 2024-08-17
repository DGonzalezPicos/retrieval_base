from retrieval_base.chemistry import Chemistry
import pandas as pd


create_csv = False

if create_csv:
    species_info = Chemistry.species_info
    species_plot_info = Chemistry.species_plot_info

    # create dictionary with columns: name, pRT_name, mass, color, mathtext_name, C, O, H
    df = pd.DataFrame({'name': [], 'pRT_name': [], 'mass': [], 'color': [], 'mathtext_name': [], 'C': [], 'O': [], 'H': []})
    c_i = 0
    for i, (k, v) in enumerate(species_info.items()):
        name = k
        pRT_name = v[0]
        # hill_notation = v[1] # ignore for now...abs
        mass = v[2]
        C, O, H = v[3]
        
        if k in species_plot_info:
            color = species_plot_info[k][0]
            mathtext_name = species_plot_info[k][1]
        else:
            c_i += 1
            color = f'C{c_i}'
            mathtext_name = k
            
        if i == 0:
            print(f'{name} & {pRT_name} & {mass} & {color} & {mathtext_name} & {C} & {O} & {H} \\\\')
            
        # add row to dataframe
        df = pd.concat([df, pd.DataFrame({'name': [name], 'pRT_name': [pRT_name], 'mass': [mass], 
                                            'color': [color], 'mathtext_name': [mathtext_name], 'C': [C], 'O': [O], 'H': [H]})])
        
    # save as csv
    df.to_csv('data/species_info.csv', index=False)
    # print(df)
    # load csv to check it
    df_load = pd.read_csv('data/species_info.csv', index_col=False)


# test read_species_info
species_i = '14CO'
keys = ['mass', 'pRT_name', 'C', 'O', 'H', 'color', 'mathtext_name']
for k in keys:
    print(f'{k}: {Chemistry.read_species_info(species_i, k)}')
