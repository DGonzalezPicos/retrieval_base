import numpy as np
import matplotlib.pyplot as plt
from astroquery.gaia import Gaia
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

# Define a query to select nearby M dwarfs
parallax_mas = 50.0 # parallax in milliarcseconds (mas), equivalent to a distance of <= 20 pc
query = f"""
SELECT source_id, ra, dec, parallax, pmra, pmdec, radial_velocity
FROM gaiadr3.gaia_source
WHERE parallax > {parallax_mas}  -- Select stars with parallax > 100 mas (distance < 100 pc)
AND radial_velocity IS NOT NULL  -- Only include stars with radial velocity measurements
"""

# Execute the query
job = Gaia.launch_job_async(query)
results = job.get_results()
assert len(results) > 0, 'No results found!'

# Convert the Gaia data to astropy SkyCoord format for proper velocity transformations
c = SkyCoord(ra=results['ra'],
             dec=results['dec'],
             distance=results['parallax'].to(u.pc, equivalencies=u.parallax()),
             pm_ra_cosdec=results['pmra'],
             pm_dec=results['pmdec'],
             radial_velocity=results['radial_velocity'])

# Now we can convert to the Galactic (U, V, W) velocities
uvw = c.velocity.to_cartesian()

# Add the U, V, W velocities to the original table
results['U'] = uvw.x.value
results['V'] = uvw.y.value
results['W'] = uvw.z.value

# Display the first few rows of the results with U, V, W velocities
print(results[['SOURCE_ID', 'ra', 'dec', 'U', 'V', 'W']])

# Define the Solar motion relative to the LSR (in km/s)
U_LSR = -11.1  # km/s
V_LSR = 12.24  # km/s
W_LSR = 7.25   # km/s

# Adjust the velocities to be relative to the LSR
U = results['U'] - U_LSR
V = results['V'] - V_LSR
W = results['W'] - W_LSR

# save csv with data

base_path = '/home/dario/phd/retrieval_base/'
file_csv = base_path + 'paper/data/' + f'toomre_data_gaia_parallax_{parallax_mas:.0f}.csv'
import pandas as pd
# create dataframe with source_id, ra, dec, U, V, W before LSR correction
# convert astropy.table to dataframe
df = results.to_pandas()
df.to_csv(file_csv, index=False)
print(f'Data saved as {file_csv}')


# Compute sqrt(U^2 + W^2) for the Toomre diagram
V_toomre = np.sqrt(U**2 + W**2)

plot = False

if plot:
    # Plot the Toomre diagram
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(V, V_toomre, s=5, c='k', alpha=0.5)
    ax.set(xlabel='V (km/s)', ylabel='V$_{\\rm Toomre}$ (km/s)')
    # plt.show()

    base_path = '/home/dario/phd/retrieval_base/'
    fig_name = base_path + 'paper/figures/toomre_diagram.pdf'

    fig.savefig(fig_name)
    print(f'Figure saved as {fig_name}')
    plt.close(fig)
