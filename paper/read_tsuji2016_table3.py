import pandas as pd
import re

def parse_latex_table(file_path: str) -> pd.DataFrame:
    """
    Parses a LaTeX table to extract object names and log_13C/12C values.

    Args:
        file_path (str): Path to the LaTeX file containing the table.

    Returns:
        pd.DataFrame: DataFrame with columns 'object_name' and 'log_13C/12C'.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = []
    for line in lines:
        if 'GJ' in line:
            print(line)
            name = line.split('&')[0].strip()
            print(name)
            log_13C_12C_str = line.split('&')[-2].strip()
            log_13C_12C = float(log_13C_12C_str.split('$')[-1])
            is_lower_limit = '<' in log_13C_12C_str
            print(log_13C_12C)
            print(is_lower_limit)
            data.append({'object_name': name.replace('\,', ' '), 'log_12C/13C': log_13C_12C, 'is_lower_limit': is_lower_limit})

    return pd.DataFrame(data)

if __name__ == "__main__":
    file_path = '/home/dario/phd/retrieval_base/paper/data/tsuji2016_table3.tex'
    df = parse_latex_table(file_path)
    print(df) 
    
    import numpy as np
    
    # save as csv
    df.to_csv(file_path.replace('.tex', '.csv'), index=False)
    print('saved to tsuji2016_table3.csv')
