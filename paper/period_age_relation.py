# load periods from csv file
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def age_estimate(P_rot, spectral_type, method="segmented"):
    """
    Estimate the age (in Gyr) of an M dwarf star based on its rotation period (in days)
    and spectral type using the fitted age-rotation relationships.

    Parameters
    ----------
    P_rot : float
        Rotation period in days.
    spectral_type : str
        Spectral type of the star (e.g. "M1", "M4.5").
    method : str, optional
        Fitting method to use: "segmented" (default) or "odr" (for the older track only).
    
    Returns
    -------
    age_Gyr : float
        Stellar age in Gyr.

    Raises
    ------
    ValueError
        If the spectral type is not within the supported range or if the rotation period is 
        out of the valid range for the chosen method.

    Spectral Type Ranges and Equations
    ----------------------------------
    1. For the segmented fits (default):
       - **M0–2 dwarfs** (spectral types M0 to M2):
         - For P_rot < 23.4933 days:
           log10(Age [Gyr]) = 0.0621 * P_rot - 1.0437
         - For P_rot ≥ 23.4933 days:
           log10(Age [Gyr]) = 0.0621 * P_rot - 1.0437 - 0.0528 * (P_rot - 23.4933)
       - **M2.5–6.5 dwarfs** for stars with spectral types M2.5 to M3.5:
         - For P_rot < 24.1888 days:
           log10(Age [Gyr]) = 0.0561 * P_rot - 0.8900
         - For P_rot ≥ 24.1888 days:
           log10(Age [Gyr]) = 0.0561 * P_rot - 0.8900 - 0.0521 * (P_rot - 24.1888)
       - **M4–6.5 dwarfs** (spectral types M4 to M6.5):
         - For P_rot < 25.4500 days:
           log10(Age [Gyr]) = 0.0251 * P_rot - 0.1615
         - For P_rot ≥ 25.4500 days:
           log10(Age [Gyr]) = 0.0251 * P_rot - 0.1615 - 0.0212 * (P_rot - 25.4500)
    
    2. For the ODR (older track) fits:
       - **M0–2 dwarfs**:
         Valid for P_rot ≳ 22 days.
         log10(Age [Gyr]) = 0.0094 * P_rot + 0.1909
       - **M2.5–6.5 dwarfs** (typically for spectral types M2.5–M3.5):
         Valid for P_rot ≳ 22 days.
         log10(Age [Gyr]) = 0.0041 * P_rot + 0.3691
       - **M4–6.5 dwarfs**:
         Valid for P_rot ≳ 25 days.
         log10(Age [Gyr]) = 0.0042 * P_rot + 0.3401

    The age is calculated by converting the logarithmic age to a linear scale:
        Age (Gyr) = 10^(log10(Age))
    """
    # Parse the spectral type string (e.g. "M1" or "M4.5")
    try:
        sp_num = float(spectral_type.strip().upper().replace("M", ""))
    except ValueError:
        raise ValueError(f"Invalid spectral type format. Please use a format like 'M1' or 'M4.5' for {spectral_type}.")

    assert method.lower() == "segmented", f'Method {method} not supported. Only "segmented" is supported.'
    
    if method.lower() == "segmented":
        # Choose the appropriate segmented fit based on spectral type
        if 0 <= sp_num <= 2:
            # M0-2 dwarfs
            break_point = (23.4933, 0.7643)
            slope_1 = (0.0621, 0.0024)
            intercept_1 = (-1.0437, 0.0380)
            slope_2 = (0.0528, 0.0025)
            
            log_age = slope_1[0] * P_rot + intercept_1[0]
            log_age_err_2 = ((slope_1[1]**2 * P_rot**2) + intercept_1[1]**2)
            if P_rot > break_point[0]:
                log_age -= slope_2[0] * (P_rot - break_point[0])
                log_age_err_2 += (slope_2[1]**2 * (P_rot - break_point[0])**2)
                log_age_err_2 += (slope_1[0]**2 * break_point[1]**2)
                
            log_age_err = np.sqrt(log_age_err_2)
                                  
                
                
                
        elif 2.5 <= sp_num <= 3.5:
            # M2.5–6.5 dwarfs for stars with types M2.5–M3.5
            break_point = (24.1888, 0.4268)
            slope_1 = (0.0561, 0.0012)
            intercept_1 = (-0.8900, 0.0185)
            slope_2 = (0.0521, 0.0012)
            
            log_age = slope_1[0] * P_rot + intercept_1[0]
            log_age_err_2 = ((slope_1[1]**2 * P_rot**2) + intercept_1[1]**2)
            if P_rot > break_point[0]:
                log_age -= slope_2[0] * (P_rot - break_point[0])
                log_age_err_2 += (slope_2[1]**2 * (P_rot - break_point[0])**2)
                log_age_err_2 += (slope_1[0]**2 * break_point[1]**2)
                
            log_age_err = np.sqrt(log_age_err_2)
            
        elif 4 <= sp_num <= 6.5:
            # M4–6.5 dwarfs
            break_point = (25.4500, 1.9079)
            slope_1 = (0.0251, 0.0018)
            intercept_1 = (-0.1615, 0.0303)
            slope_2 = (0.0212, 0.0018)
            
            log_age = slope_1[0] * P_rot + intercept_1[0]
            log_age_err_2 = ((slope_1[1]**2 * P_rot**2) + intercept_1[1]**2)
            if P_rot > break_point[0]:
                log_age -= slope_2[0] * (P_rot - break_point[0])
                log_age_err_2 += (slope_2[1]**2 * (P_rot - break_point[0])**2)
                log_age_err_2 += (slope_1[0]**2 * break_point[1]**2)
                
            log_age_err = np.sqrt(log_age_err_2)

        else:
            raise ValueError("Spectral type out of supported range. Supported ranges are M0 to M6.5.")

    # Convert from logarithmic (base 10) age to linear age in Gyr.
    age_Gyr = 10 ** log_age
    age_Gyr_err = 10.0 ** log_age_err
    if age_Gyr > 14:
        age_Gyr = np.nan
    return age_Gyr, age_Gyr_err

path = Path(__file__).parent
file = path / 'data/cristofari23_table2.csv'

data = pd.read_csv(file, names=['name', 'spectral_type', 'period', 'error'])
# clean spectral type
data['spectral_type'] = data['spectral_type'].str.replace('V', '')
# print(data.head())


# Example inputs
method = "segmented"

# run for all data
for index, row in data.iterrows():
    age, age_err = age_estimate(row['period'], row['spectral_type'], method=method)
    # print(f"Spectral type {row['spectral_type']} with P_rot = {row['period']} days "
    #       f"({method} fit) gives an estimated age of {age:.2f} Gyr.")
    data.at[index, 'age'] = age
    data.at[index, 'age_err'] = age_err
    
    
# generate the different tracks to plot them
tracks = {}
tracks['early M'] = []
tracks['late M'] = []
periods = np.linspace(1, 200, 20)
for p in periods:
    tracks['early M'].append(age_estimate(p, 'M0', method=method))
    tracks['late M'].append(age_estimate(p, 'M6', method=method))
    
tracks['early M'] = np.array(tracks['early M']).T
tracks['late M'] = np.array(tracks['late M']).T

# plot age vs period
fig, ax = plt.subplots(figsize=(7, 5))
# ax.scatter(data['period'], data['age'], c=data['spectral_type'])

def plot_track_envelopes(tracks, ax, **kwargs):
    lower_track = tracks[0] - tracks[1]
    upper_track = tracks[0] + tracks[1]
    alpha = kwargs.pop('alpha', 0.2)
    label = kwargs.pop('label', None)
    ax.fill_between(periods, lower_track, upper_track, alpha=alpha, **kwargs)
    ax.plot(periods, tracks[0], label=label, **kwargs)

plot_track_envelopes(tracks['early M'], ax, label='Early M', color='navy', zorder=-1)
plot_track_envelopes(tracks['late M'], ax, label='Late M', color='darkorange', zorder=-1)

ax.scatter(data['period'], data['age'], c='k', facecolor='none', edgecolor='w', s=60,
           alpha=0.95)
ax.errorbar(data['period'], data['age'], yerr=data['age_err'], c='k', fmt='none',
            alpha=0.5, elinewidth=1, capsize=2)

ax.grid(True, alpha=0.5)

ax.set_xlabel('Rotation period (days)')

y_logscale = False
if y_logscale:
    ax.set_yscale('log')
ax.set_ylabel('Age (Gyr)')
ax.set_title('Rotation-Age relation (Engle and Guinan, 2023)')
ax.legend()
# plt.show()

nat_path = Path('/home/dario/phd/nat/figures/')
fig_name = nat_path / 'rotation_age_relation.pdf'
fig.savefig(fig_name, dpi=300, bbox_inches='tight')
# save as png too
fig_name = nat_path / 'rotation_age_relation.png'
fig.savefig(fig_name, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {fig_name}")
plt.close(fig)

# Create a nicely formatted table for display and saving
formatted_table = data[['name', 'spectral_type', 'period', 'error', 'age', 'age_err']].copy()
formatted_table.columns = ['Name', 'Spectral Type', 'Period (days)', 'Period Error (days)', 
                         'Age (Gyr)', 'Age Error (Gyr)']

# Round numerical columns to 2 decimal places
numeric_cols = ['Period (days)', 'Period Error (days)', 'Age (Gyr)', 'Age Error (Gyr)']
formatted_table[numeric_cols] = formatted_table[numeric_cols].round(2)

# Display the formatted table
print("\nRotation Periods and Age Estimates:")
print("===================================")
print(formatted_table.to_string(index=False))

# Save to CSV
output_file = path / 'data/rotation_age_estimates.csv'
formatted_table.to_csv(output_file, index=False)
print(f"\nTable saved to: {output_file}")



