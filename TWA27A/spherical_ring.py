import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm

# Constants
G = 6.67430e-8  # Gravitational constant in cgs units (cm^3 g^-1 s^-2)
Mjup = 1.898e30  # Mass of Jupiter in grams
rjup = 7.1492e9  # Radius of Jupiter in cm

# Function to create colorbars
def make_cbar(vmin, vmax, cmap, label, ax):
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(mappable, ax=ax, orientation='horizontal', fraction=0.05, pad=0.1)
    cbar.set_label(label)
    return norm, mappable

# Define inclination and position angle
i_deg = 35  # Inclination angle in degrees
i = np.radians(i_deg)
pa_deg = 165  # Position angle in degrees
pa = np.radians(pa_deg)


# Define grid in spherical coordinates (r, theta, phi)
r = np.linspace(1.0, 11.0, 100)  # Avoid zero to prevent division errors in Keplerian velocity
phi = np.linspace(0, 2 * np.pi, 100)  # Azimuthal angle
theta = np.pi - pa



# Create mesh grid
R, PHI = np.meshgrid(r, phi)

# Convert spherical to Cartesian coordinates
x = R * np.cos(PHI) * np.sin(theta)
y = R * np.sin(PHI) * np.sin(theta)

# Brightness map
brightness = np.real(np.exp(1j * PHI * np.sin(theta)))

# Keplerian velocity and projected velocity map in the x,y-plane with inclination and PA
M_g = 20.0 * Mjup  # Mass of the central object in grams
R_cm = R * rjup  # Convert radius to cm
v_keplerian = np.sqrt(G * M_g / R_cm)

# Projected velocity along line of sight considering inclination and PA
velocity = v_keplerian * (np.sin(i) * np.cos(PHI - pa)) * 1e-5  # Convert to km/s

# Weighted velocity map (weighted by brightness and emitting area)
emitting_area_factor = R / np.max(R)
# emitting_area_factor = 1
weighted_velocity = velocity * brightness * emitting_area_factor

# Plot
fig, ax = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'aspect': 'equal'})

# Brightness map (left)
ax[0].set_title("Brightness Map (Re(e^{i*phi}))")
ax[0].set(xlabel='x', ylabel='y')
norm_b, mappable_b = make_cbar(0.0, 1, 'inferno', 'Re(e^{i*phi})', ax=ax[0])
contour_brightness = ax[0].contourf(x, y, brightness, levels=100, cmap=mappable_b.cmap, norm=norm_b)

# Velocity map (center), set vmin, vmax to (-100, 100) km/s
ax[1].set_title("Keplerian Velocity Map")
ax[1].set(xlabel='x', ylabel='y')
vmax = 100.0
norm_v, mappable_v = make_cbar(-vmax, vmax, 'RdBu_r', 'Velocity (km/s)', ax=ax[1:])
contour_velocity = ax[1].contourf(x, y, velocity, levels=100, cmap=mappable_v.cmap, norm=norm_v)

# Weighted velocity map (right)
ax[2].set_title("Weighted Keplerian Velocity Map")
ax[2].set(xlabel='x', ylabel='y')
contour_weighted_velocity = ax[2].contourf(x, y, weighted_velocity, levels=100, cmap=mappable_v.cmap, norm=norm_v)

# Set figure title with inclination and position angle
fig.suptitle(f'Ring inclination = {i_deg}°, Position angle = {pa_deg}°')
plt.show()
