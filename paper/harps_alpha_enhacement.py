import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Tuple, List, Dict
import seaborn as sns
import sys
from scipy.stats import gaussian_kde

# Define the path to the file
path = Path('/home/dario/phd/retrieval_base/paper/data')
file = path / 'Adibekyan2012_table4.txt'



def calculate_alpha_abundance(df: pd.DataFrame,
                              alpha_elements: List[str] = ['mg','si','ca','ti']) -> pd.Series:
    """
    Calculate the average alpha-element abundance for each star.
    
    Args:
        df: DataFrame containing abundance measurements
    
    Returns:
        Series containing mean alpha abundance for each star
    """
    # Alpha elements: Mg, Si, Ca, Ti
    alpha_cols = [f'{el}h' for el in alpha_elements]
    return df[alpha_cols].mean(axis=1) - df['feh']

def create_scatter_with_histograms(
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    skip_scatter: bool = False
) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
    """
    Create a scatter plot with marginal histograms.
    
    Args:
        x: x-axis data
        y: y-axis data
        xlabel: x-axis label
        ylabel: y-axis label
        title: plot title
    
    Returns:
        fig: Figure object
        axes: Dictionary containing all axes objects
    """
    # Create figure with custom layout
    
    fig = plt.figure(figsize=(9, 8))
    gs = GridSpec(3, 3)
    
    # Create axes
    ax_scatter = fig.add_subplot(gs[1:, :-1])
    ax_hist_x = fig.add_subplot(gs[0, :-1], sharex=ax_scatter)
    ax_hist_y = fig.add_subplot(gs[1:, -1], sharey=ax_scatter)
    
    # Plot scatter
    if not skip_scatter:
        scatter = ax_scatter.scatter(x, y, alpha=0.6, c='darkblue', s=30)
    
    # Plot histograms
    ax_hist_x.hist(x, bins=30, color='darkblue', alpha=0.6, density=True)
    ax_hist_y.hist(y, bins=300, color='darkblue', alpha=0.6, 
                  orientation='horizontal', density=True)
    
    # Customize scatter plot
    ax_scatter.grid(True, linestyle='--', alpha=0.3)
    ax_scatter.set_xlabel(xlabel, fontsize=12)
    ax_scatter.set_ylabel(ylabel, fontsize=12)
    
    # Remove histogram ticks
    ax_hist_x.tick_params(labelleft=False)
    ax_hist_y.tick_params(labelbottom=False)
    
    # Add title
    fig.suptitle(title, fontsize=14, y=0.95)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, {
        'scatter': ax_scatter,
        'hist_x': ax_hist_x,
        'hist_y': ax_hist_y
    }
    
def load_catalogue(file: Path) -> pd.DataFrame:
    """
    Load the catalogue from the file.
    """
     # Define the expected column names
    columns = [
        'name', 'teff', 'logg', 'feh', 'vtur',
        'nah', 'e_nah', 'o_nah', 'nah_c',
        'mgh', 'e_mgh', 'o_mgh',
        'alh', 'e_alh', 'o_alh', 'alh_c',
        'sih', 'e_sih', 'o_sih',
        'cah', 'e_cah', 'o_cah',
        'sc1h', 'e_sc1h', 'o_sc1h', 'sc1h_c',
        'sc2h', 'e_sc2h', 'o_sc2h',
        'ti1h', 'e_ti1h', 'o_ti1h', 'ti1h_c',
        'ti2h', 'e_ti2h', 'o_ti2h',
        'vh', 'e_vh', 'o_vh', 'vh_c',
        'cr1h', 'e_cr1h', 'o_cr1h',
        'cr2h', 'e_cr2h', 'o_cr2h', 'cr2h_c',
        'mnh', 'e_mnh', 'o_mnh',
        'coh', 'e_coh', 'o_coh', 'coh_c',
        'nih', 'e_nih', 'o_nih'
    ]

    n_cols = len(columns)

    # Read and parse the file into a list of padded rows
    rows = []
    with open(file, 'r') as f:
        for i, line in enumerate(f, start=1):
            parts = line.strip().split()
            if len(parts) < n_cols:
                parts += [np.nan] * (n_cols - len(parts))  # pad missing values
            elif len(parts) > n_cols:
                parts = parts[:n_cols]  # truncate extras
            rows.append(parts)

    # Create DataFrame
    df = pd.DataFrame(rows, columns=columns)

    # Convert numeric columns
    non_numeric = ['name']
    for col in df.columns.difference(non_numeric):
        df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"Loaded {len(df)} rows with shape {df.shape}.")
    # print(df.head())

    return df

def load_probability_populations(file: Path) -> pd.DataFrame:
    """
    Load the probability populations from the file.
    """
    columns = ['name', 
               'U', 'V', 'W',
               'pDB', # probability of being a THICK disk star according to B03
               'pTDB', # probability of being a THIN disk star according to B03
               'pHB', # probability of being a HALO star according to B03
               'popB', # population of being a HALO star according to B03
               'pDR', # probability of being a TRANSITIONAL disk star according to B03
               'pTDR', # probability of being a THIN disk star according to B03
               'pHR', # probability of being a HALO star according to B03
               'popR' # population of being a TRANSITIONAL disk star according to B03
               ]
    n_cols = len(columns)

    # load txt file with columns of different types, separated by spaces
    df = pd.read_csv(file, sep='\s+', header=None, names=columns)
    # print(df.head())
    # sys.exit()
    
    return df

def main():
    """Main function to process data and create visualization."""
    # select subsample of catalogue matching the ranges
    df = load_catalogue(file)
    df_prob = load_probability_populations(file.parent / 'Adibekyan2012_table5.txt')
    # combine df and df_prob on column 'name'
    df = df.merge(df_prob, on='name', how='left')
    print(f' Merged df and df_prob on column "name" with shape {df.shape}')

    df = df[df['feh'] > -1.5]
    
    # Calculate alpha abundance
    alpha_elements = ['mg','si']
    df['alpha_fe'] = calculate_alpha_abundance(df, alpha_elements)
    fig, axes = create_scatter_with_histograms(
            x=df['feh'],
            y=df['alpha_fe'],
            xlabel='[Fe/H]',
            ylabel='[α/Fe]',
            title='All stars',
            skip_scatter=True
    )
    
    # Create visualization
    groups = ['halo', 'thin', 'thick', 'trans']
    colors = {'halo': 'green', 'thin': 'navy', 'thick': 'brown', 'trans': 'magenta'}
    markers = {'halo': 'o', 'thin': 's', 'thick': 'D', 'trans': 'd'}
    
    for i, group in enumerate(groups):
        df_group = df[df['popR'] == group]
        print(f' plotting {group} with {len(df_group)} stars')
        axes['scatter'].scatter(df_group['feh'], df_group['alpha_fe'], alpha=0.4+i*0.1, c=colors[group], s=40,
                                label=group, edgecolor='w', linewidth=0.2, marker=markers[group])
        
    # Add style elements
    axes['scatter'].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes['scatter'].axvline(x=0, color='k', linestyle='--', alpha=0.3)
    axes['scatter'].legend(loc='lower left', fontsize=12)
    
    # Set axis limits with padding
    xlim = (-1.2, 0.5)
    ylim = (-0.2, 0.4)
    axes['scatter'].set_xlim(xlim)
    axes['scatter'].set_ylim(ylim)
    
    # add text labels on the plot for each group
    for group in ['thin', 'thick']:
        df_group = df[df['popR'] == group]
        text_x = df_group['feh'].mean()
        text_y = df_group['alpha_fe'].mean()
        axes['scatter'].text(text_x, text_y, group.upper(), 
                               color=colors[group], fontweight='bold',
                               ha='center', va='center')
    # Save figure
    fig_path = path.parent / 'figures' / 'harps_alpha_enhancement.pdf'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {fig_path}")
    plt.close()

if __name__ == "__main__":
    
   
    main()
    # df_prob = load_probability_populations(file)
    
