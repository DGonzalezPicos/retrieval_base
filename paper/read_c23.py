import pandas as pd
import pathlib
import numpy as np

table_id = 3 # 3 is logg free, 4 is logg derived from L/Lsun and Teff
assert table_id in [3,4], f'table_id must be 3 or 4, not {table_id}'

file = pathlib.Path(f'paper/data/c23_table{table_id}_raw.tex')

df = pd.read_csv(file, sep='&', header=0, engine='python', skiprows=[1])
print(df.columns)
# select columns of interest
attrs = ['Star ', r' $\teff$ (K) ',' $\logg$ (dex) ', ' $\mh$ (dex) ', r' $\afe$ (dex) ']
rename_attrs = ['name', 'teff', 'logg', 'mh', 'alpha_fe']
df = df[attrs]
df.columns = rename_attrs

names = df['name'].str.replace(r'\,', '').str.strip().values
names = [n.replace('GJ', 'Gl') for n in names]

def extract_column(df, colname='mh'):
    # extract values and errors
    assert colname in df.columns, f'Column {colname} not found in dataframe {df.columns}'
    col = df[colname].str.split(r'\\pm')
    values = np.array(col.str[0].str.replace('$', '').str.strip(), dtype=float)
    err = np.array(col.str[1].str.replace('$', '').str.strip(), dtype=float)
    # create tuple with values and errors
    values_err = [(v, e) for v, e in zip(values, err)]

    col_dict = dict(zip(names, values_err))
    # save dict as txt with three columns: name, value, error
    np.savetxt(f'paper/data/c23_table{table_id}_{colname}.txt', np.array([names, values, err]).T, fmt='%s')
    print(f' Saved {f"paper/data/c23_table{table_id}_{colname}.txt"}')

    return col_dict

# mh = extract_column(df, colname='mh')
alpha_h = extract_column(df, colname='alpha_fe')