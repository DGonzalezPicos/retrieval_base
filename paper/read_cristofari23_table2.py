import pandas as pd
from pathlib import Path
import numpy as np
path = Path(__file__).parent
file = path / 'data/cristofari23_table2.tex'

def parse_latex_table(file: Path) -> pd.DataFrame:
    with open(file, 'r') as file:
        lines = file.readlines()

    data = []
    
    columns = ['name','spectral_type', 'mass', 'radius', 'tau', 'rotation_period', 'ro']
    for line in lines:
        if ('GJ' in line or 'Gl' in line) and not 'caption' in line:
            parts = line.split('&')
            # print(parts)
            # Extracting and cleaning data
            target_data = [part.strip().replace('$', '') for part in parts]
            data.append(target_data)

    # Create DataFrame from the parsed data
    df = pd.DataFrame(data, columns=columns)
    return df

# Example usage
df = parse_latex_table(file)

periods = {}
spt = {}
for i, (k, v) in enumerate(zip(df['name'], df['rotation_period'])):
    print(k, v)
    if '...' in v:
        # print(k, v)
        continue
    else:
        # periods[k] = float(v)
        name = k.replace('\,', ' ').replace('GJ', 'Gl')
        value = float(v.split('\pm')[0].replace('^*',''))
        error = float(v.split('\pm')[1])
        periods[name] = (value, error)
        spt[name] = df['spectral_type'].iloc[i]
# print(periods)
# save as csv with four columns: name, spectral_type, value, error
array_to_save = np.array(list(periods.keys()))
array_to_save = np.vstack((array_to_save, 
                           np.array(list(spt.values())).T,
                           np.array(list(periods.values())).T))
np.savetxt(path.parent / 'paper/data/cristofari23_table2.csv', array_to_save.T, delimiter=',', fmt='%s')
print(f' file saved to {path.parent / "paper/data/cristofari23_table2.csv"}')