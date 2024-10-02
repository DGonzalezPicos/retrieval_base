from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord, Galactic, Galactocentric
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Step 1: Define SIMBAD query to get coordinates, pm, radial velocities and parallaxes (distances)
custom_simbad = Simbad()

# also query gaia id
custom_simbad.add_votable_fields('coordinates', 'pmra', 'pmdec', 'rv_value', 'parallax', 'ids')

# List of target objects
spirou_sample = {'880': [(3720, 4.72, 0.21, 6.868), '17'],
                 '15A': [(3603, 4.86, -0.30, 3.563), None],
                # '411': (3563, 4.84, 0.12), # TODO: double check this target
                '832': [(3590, 4.70, 0.06, 4.670),None],  # Tilipman+2021
                '752A': [(3558, 4.76, 0.10, 3.522),None], # Cristofari+2022
                '849':  [(3530, 4.78, 0.37, 8.803),None], # Cristofari+2022
                '725A': [(3441, 4.87, -0.23, 3.522),None],# Cristofari+2022
                '687': [(3413, 4.80, 0.10, 4.550),None], # Cristofari+2022
                '876' : [(3366, 4.80, 0.10, 4.672),None], # Moutou+2023, no measurement for logg, Z

                '725B': [(3345, 4.96, -0.30, 3.523),None],
                '699': [(3228.0, 5.09, -0.40, 1.827),None],
                '15B': [(3218, 5.07, -0.30, 3.561),None],
                '1151': [(3178, 4.71, -0.04, 8.043),None], # Lehmann+2024, I call it `gl` but it's `gj`
                '905': [(2930, 5.04, 0.23, 3.155),None],
}


targets = ['gl'+t for t in spirou_sample.keys()]
temperature_dict = {t: spirou_sample[t[2:]][0][0] for t in targets}
# norm = plt.Normalize(min(temperature_dict.values()), max(temperature_dict.values()))
norm = plt.Normalize(min(temperature_dict.values()), 4000.0)
cmap = plt.cm.plasma
colors = [cmap(norm(t)) for t in temperature_dict.values()]

def get_gaia_dr3(IDS):
    
    s = IDS.split('|')
    assert len(s) > 1, f'No Gaia DR3 ID found in {IDS}'
    
    gaia_dr3 = [i for i in s if 'Gaia DR3' in i]
    assert len(gaia_dr3) == 1, f'Must only have one Gaia DR3 ID, not {len(gaia_dr3)}'
    # print(f' Gaia DR3 ID = {gaia_dr3[0]}')
    
    gaia_dr3_id = int(gaia_dr3[0].split('Gaia DR3')[-1])
    return gaia_dr3_id


# Query the data for the targets
result_table = custom_simbad.query_objects(targets)
gaia_dr3_list = [get_gaia_dr3(row['IDS']) for row in result_table]


# Step 2: Extract data
ra = result_table['RA']   # Right ascension in HMS
dec = result_table['DEC']  # Declination in DMS
pmra = result_table['PMRA']  # Proper motion in RA (mas/yr)
pmdec = result_table['PMDEC']  # Proper motion in DEC (mas/yr)
rv = result_table['RV_VALUE']  # Radial velocity (km/s)
parallax = result_table['PLX_VALUE']  # Parallax (mas)

# Convert RA, Dec to degrees
coords = SkyCoord(ra=ra, dec=dec, unit=(u.hourangle, u.deg))

# Step 3: Calculate distances from parallax
distance = (1000 / parallax) * u.pc  # Convert parallax to distance (parsec)

# Create SkyCoord object with proper motions, radial velocities and distances
skycoord = SkyCoord(ra=coords.ra, dec=coords.dec, 
                    distance=distance,
                    pm_ra_cosdec=pmra, pm_dec=pmdec, 
                    radial_velocity=rv)

# Now we can convert to the Galactic (U, V, W) velocities
uvw = skycoord.velocity.to_cartesian()


# Add the U, V, W velocities to the original table

v = np.array([uvw.x.value, uvw.y.value, uvw.z.value]) # vector with [U, V, W] velocities

# Step 5: Adjust to the Local Standard of Rest (LSR)
# Solar motion relative to LSR from Schönrich et al. 2010 (U_sun, V_sun, W_sun in km/s)
v_sun = np.array([-11.1, 12.24, 7.25])
# center of toomre diagram in LSR
v_sun_centre = np.sqrt(np.sum(v_sun**2))

# Extract U, V, W velocities (correcting for LSR)
U, V, W = v - v_sun[:, None]

# Step 6: Plot the Toomre diagram (V on x-axis and sqrt(U^2 + W^2) on y-axis)
UW = np.sqrt(U**2 + W**2)  # Magnitude of (U, W)


# load GAIA DR3 data
parallax_mas = 50 # mas
file_gaia = '/home/dario/phd/retrieval_base/paper/data/toomre_data_gaia_parallax_50.csv'
df_gaia = pd.read_csv(file_gaia)
# apply LSR correction for U, V, W
df_gaia['U'] = df_gaia['U'] - v_sun[0]
df_gaia['V'] = df_gaia['V'] - v_sun[1]
df_gaia['W'] = df_gaia['W'] - v_sun[2]

# match my targets with GAIA DR3
df_gaia_sample = df_gaia[df_gaia['SOURCE_ID'].isin(gaia_dr3_list)]
assert len(df_gaia_sample) == len(targets), f'Length of GAIA DR3 sample ({len(df_gaia_sample)}) does not match targets ({len(targets)})'
# Create the plot
fig, ax = plt.subplots(figsize=(6, 3), tight_layout=True)
ax.scatter(df_gaia['V'], np.sqrt(df_gaia['U']**2 + df_gaia['W']**2), s=10, c='k', alpha=0.2, lw=0, 
           label='Gaia DR3' + r' ($\varpi \leq$' + f' {parallax_mas} mas)')
im = ax.scatter(df_gaia_sample['V'], np.sqrt(df_gaia_sample['U']**2 + df_gaia_sample['W']**2), 
           s=70, c=colors,
           alpha=0.98, label='M dwarfs',
           edgecolors='k', lw=0.5)
# add colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Only needed for color bar
cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.03, aspect=15, location='right')
cbar.set_label('Temperature / K')

# add text next to the points

offset = (-8.0, 10.0)
for i, target in enumerate(targets):
    V, U, W = df_gaia_sample['V'].iloc[i], df_gaia_sample['U'].iloc[i], df_gaia_sample['W'].iloc[i]
    if np.sqrt(np.sum(np.square([V,U,W]))) > 60.0:
        U2 = U**2
        W2 = W**2
        
        ax.text(V+offset[0], np.sqrt(U2 + W2) + offset[1], target.replace('gl', 'Gl '), fontsize=12, color='k',
                ha='center', va='center')
              
# Define circles for different stellar populations
circ = np.linspace(-500, 500, 5000)

step = 50.0
velocities = np.arange(100, 250.0, step)
for disk_velocity in velocities:
    # center around the LSR
    ax.plot(circ, np.sqrt((disk_velocity - v_sun_centre)**2 - circ**2), 'k--', lw=0.5, zorder=-10)
    # ax.plot(circ, np.sqrt(disk_velocity**2 - circ**2), 'k--', lw=0.5)
    

# remove filling
# ax.scatter(V, UW, s=50, c='b', alpha=0.8, label='SIMBAD', facecolors='none', edgecolors='b')
kms = 'km s' + r'$^{-1}$'
ax.set(xlabel='$V$' + f' / {kms}', ylabel=r'$\sqrt{U^2 + W^2}$' + f' / {kms}')
ax.set_xlim(-200.0, 150.0)
ax.set_ylim(0.0, 150.0)
# ax.legend()
fig_name = '/home/dario/phd/retrieval_base/paper/figures/toomre_diagram.pdf'
fig.savefig(fig_name)
print(f' Saved {fig_name}')
plt.show()
