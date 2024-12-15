import pandas as pd
import pathlib
import numpy as np

table_id = 3 # 3 is logg free, 4 is logg derived from L/Lsun and Teff
assert table_id in [3,4], f'table_id must be 3 or 4, not {table_id}'

file = pathlib.Path(f'paper/data/c23_table{table_id}_raw.tex')

df = pd.read_csv(file, sep='&', header=0, engine='python', skiprows=[1])
print(df.columns)
# select columns of interest
attrs = ['Star ', r' $\teff$ (K) ',' $\logg$ (dex) ', ' $\mh$ (dex) ']
rename_attrs = ['name', 'teff', 'logg', 'mh']
df = df[attrs]
df.columns = rename_attrs

names = df['name'].str.replace(r'\,', '').str.strip().values
names = [n.replace('GJ', 'Gl') for n in names]

mh = df['mh'].str.split(r'\\pm')
values = np.array(mh.str[0].str.replace('$', '').str.strip(), dtype=float)
err = np.array(mh.str[1].str.replace('$', '').str.strip(), dtype=float)
# create tuple with values and errors
values_err = [(v, e) for v, e in zip(values, err)]

mh_dict = dict(zip(names, values_err))
# save dict as txt with three columns: name, value, error
np.savetxt(f'paper/data/c23_table{table_id}_mh.txt', np.array([names, values, err]).T, fmt='%s')

# load the data
# md_load = np.loadtxt('paper/data/c23_mh.txt', dtype=object)