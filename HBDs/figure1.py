import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('HBDs/my_science.mplstyle')
plt.rcParams['text.usetex'] = False



## Figure 1: Temperature vs Mass of Brown Dwarfs and Planets
df = pd.read_csv('HBDs/imaging_exoplanet.eu_catalog.csv')
print(f'Number of planets: {len(df)}')
print(f'Available columns: {df.columns}')



# check if "GQ Lup b" is in the list of planets
print('GQ Lup b' in df['name'].values)
# add temperature for GQ Lup b
df.loc[df['name'] == 'GQ Lup b', 'temp_measured'] = 2700.
# adjust mass and mass error for GQ Lup b (Stolker+2021)
df.loc[df['name'] == 'GQ Lup b', 'mass'] = 30.
df.loc[df['name'] == 'GQ Lup b', 'mass_error_min'] = 10.
df.loc[df['name'] == 'GQ Lup b', 'mass_error_max'] = 10.
df.loc[df['name'] == 'GQ Lup b', 'star_age'] = 3e-3

# adjust mass and mass error for DH Tau b (Kraus+2014, Xuan+2020)
df.loc[df['name'] == 'DH Tau b', 'mass'] = 18.
df.loc[df['name'] == 'DH Tau b', 'mass_error_min'] = 4.
df.loc[df['name'] == 'DH Tau b', 'mass_error_max'] = 4.
df.loc[df['name'] == 'DH Tau b', 'star_age'] = 2e-3
df.loc[df['name'] == 'DH Tau b', 'temp_measured'] = 2300. # Xuan+2020


# replace NaN with zeros for mass_error_min and mass_error_max
df['mass_error_min'] = df['mass_error_min'].fillna(0.)
df['mass_error_max'] = df['mass_error_max'].fillna(0.)

# discard objects with temperatures hotter than 500 K
df = df[df['temp_measured'] > 500.]

fig, ax = plt.subplots(figsize=(8,6))

# make size of points proportional to radius
size = df['radius']**2
# ax.scatter(df['mass'], df['temp_measured'], s=40., label='Measured Temperature', alpha=0.8)
# plot points with errorbars for the x-axis (different upper and lower error) 

df['star_age'] = df['star_age'] * 1e3 # convert to Myr
age = df['star_age']
# add the Hot Brown Dwarfs

bds = dict(J1200={'mass': 42., 'mass_error_min': 5., 'mass_error_max':5., 'temp': 2600., 'age': 3.7},
                      
           J0856={'mass': 14.4, 'mass_error_min': 1.4, 'mass_error_max':0.8, 'temp': 2500., 'age': 10.},
           TWA28={'mass': 20.9, 'mass_error_min': 5., 'mass_error_max':5., 'temp': 2600., 'age': 10.},
           )
df_bds = pd.DataFrame(bds).T
# add keys as column "name"
df_bds['name'] = df_bds.index

# make colorbar with age
cmap = plt.colormaps['summer_r']
# define a range of colors from min to max age in log space
from matplotlib.colors import LogNorm

log_norm = LogNorm(1., 1e3)
# norm = plt.Normalize(age.min(), age.max())
# create a ScalarMappable object with the colormap and norm
sm = plt.cm.ScalarMappable(cmap=cmap, norm=log_norm)
# set array of colors for each point
sm.set_array([])
im = ax.scatter([], [], c=[], cmap=cmap, norm=log_norm, s=40., alpha=0.8)
# add colorbar, less padding between colorbar and plot
cbar = fig.colorbar(sm, ax=ax, label='Age (Myr)', pad=0.02)
# set colorbar ticks
# cbar.set_ticks([0, 5, 10, 15, 20])
# # set colorbar tick labels
# cbar.set_ticklabels([str(x) for x in cbar.get_ticks()])
# # set colorbar tick labels to be horizontal
# cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=0)



# loop over rows, if age is nan, set color to black
for row in df.iterrows():
   
    alpha = 0.95
    color = sm.to_rgba(row[1]['star_age'])
     
    if np.isnan(row[1]['star_age']):
        color = 'gray'
        alpha=0.7
    
    temp = row[1]['temp_measured'] if not np.isnan(row[1]['temp_measured']) else row[1]['temp_calculated']
    # if GQ Lup b, add black edge
    
    mec = 'black'
    mew = 1.0
    markersize = 16.
    label = None
    zorder = 0
    if row[1]['name'] == 'GQ Lup b':
        mec = 'royalblue'
        # markersize = 18.
        label = row[1]['name'].replace(' b', ' B')
        zorder = 99
        mew=3.5
        print(f"GQ Lup b age = {row[1]['star_age']} color = {color}")
    if row[1]['name'] == 'DH Tau b':
        mec = 'indianred'
        mew = 3.5
        label = row[1]['name'].replace(' b', ' B')
        zorder = 98
        print(f"DH Tau b age = {row[1]['star_age']} color = {color}")
    
    ax.errorbar(row[1]['mass'], temp, 
                xerr=[[row[1]['mass_error_min']], [row[1]['mass_error_max']]],
                fmt='D', markersize=markersize, color=color, alpha=alpha, mec=mec, mew=mew,
                label=label, zorder=zorder)

markers = ['s', 'v', 'o']
for i, row in enumerate(df_bds.iterrows()):
    color = sm.to_rgba(row[1]['age'])
    print(f"{row[0]} age = {row[1]['age']} color = {color}")
    # make edge of marker black
    ax.errorbar(row[1]['mass'], row[1]['temp'], 
                xerr=[[row[1]['mass_error_min']], [row[1]['mass_error_max']]],
                fmt=markers[i], markersize=18., color=color, alpha=0.95, mec='k', mew=3.5,
                label=row[0])
    
# increase spacing between legend entries
# make the 3 last entries in the legend in bold
# invert order of entries in legend
handles, labels = ax.get_legend_handles_labels()
handles = handles[::-1]
labels = labels[::-1]
leg = ax.legend(handles, labels, loc='upper left', fontsize=14, frameon=True, labelspacing=1.2, borderpad=0.5, handletextpad=0.5, ncol=1)
# leg = ax.legend(loc='upper left', fontsize=14, frameon=True, labelspacing=1.2, borderpad=0.5, handletextpad=0.5, ncol=1)
for i in range(3, 6):
    leg.get_texts()[len(leg.get_texts())-i].set_fontweight("bold")


# ax.legend(loc='upper left', fontsize=14, frameon=True, labelspacing=1.2, borderpad=0.5, handletextpad=0.5, ncol=1)
# plot points with colorbar
# ax.scatter(df['mass'], df['temp_measured'], s=size, c=age, cmap=cmap, label='Measured Temperature', alpha=0.8)


# ax.errorbar(df['mass'], df['temp_measured'], xerr=[df['mass_error_min'], df['mass_error_max']], fmt='D', markersize=5, label='Measured Temperature', alpha=0.8)
# ax.scatter(df['mass'], df['temp_calculated'], s=1, label='Calculated Temperature')

ax.set_xscale('log')
# ax.set_yscale('log')
# custom ticks for mass with float formatting
ax.set_xticks([0.1, 1, 10, 30., 90.])
ax.set_xticklabels([str(x) for x in ax.get_xticks()])
ax.set(xlabel='Mass ($M_{Jup}$)', ylabel='Temperature (K)', xlim=(0.9, 100.))
ax.set_ylim(300., None)
# ax.legend()
plt.tight_layout()
plt.savefig('HBDs/figure1.pdf')
plt.show()
