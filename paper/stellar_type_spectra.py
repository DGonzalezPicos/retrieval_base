import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import pathlib
import pandas as pd

path = pathlib.Path('/home/dario/phd/pRT_input/input_data/stellar_specs')

def load_stellar_params(path):
    file = 'stellar_params.dat'
    # Load stellar parameters
    spt = []
    with open(path / file, 'r') as f:
        lines = f.readlines()
        header = lines[0].replace('#','').replace('\n','').split(', ')
        
        for line in lines[1:]:
            line = line.replace('\n','').split('   ')
            line = [l.strip() for l in line if l]  # Clean up the data
            spt.append([line[-1]] + [float(l) for l in line[:-1]])
    return spt

def load_spec(file, wave_min_nm=2200.0, wave_max_nm=2500.0, normalize=True):
    wave_cm, flux = np.loadtxt(file).T
    wave_nm = wave_cm * 1e7
    mask = (wave_nm > wave_min_nm) & (wave_nm < wave_max_nm)
    wave_nm = wave_nm[mask]
    flux = flux[mask]
    if normalize:
        flux /= np.nanmedian(flux)
    return wave_nm, flux

# Initialize plot with transparent background
fig = plt.figure(facecolor='black')

fig, ax = plt.subplots(1, 1, figsize=(16, 8))
ax.set_xlim(2200, 2500)
ax.set_ylim(0.5, 1.35)
ax.set_xlabel("Wavelength (nm)", color='white', fontsize=16)
ax.set_ylabel("Normalized Flux", color='white', fontsize=16)

# Increase font size of ticks
ax.tick_params(axis='x', colors='white', labelsize=14)
ax.tick_params(axis='y', colors='white', labelsize=14)

# Plot lines for the current and previous spectra
line_current, = ax.plot([], [], lw=2)
line_previous, = ax.plot([], [], lw=2, alpha=0.5)

# Set dark background for the plot area and ticks
ax.set_facecolor('gray')

spt = load_stellar_params(path)
teff = 10.0 ** np.array([s[1] for s in spt])
files = [path / f'spec_{i:02d}.dat' for i in range(len(spt))]

# Sort by teff, high to low
spt = [x for _, x in sorted(zip(teff, spt))][::-1]
files = [x for _, x in sorted(zip(teff, files))][::-1]
teff = sorted(teff)[::-1]

# Remove teffs that are too high or too low
teff_max = 9000.0
teff_min = 2900.0
spt = [s for s, t in zip(spt, teff) if teff_min < t < teff_max]
files = [f for f, t in zip(files, teff) if teff_min < t < teff_max]
teff = [t for t in teff if teff_min < t < teff_max]

# Define color mapping
norm = plt.Normalize(min(teff), teff_max)
cmap = plt.cm.jet_r

# Placeholder for the text to be updated during animation
text_box = ax.text(0.05, 0.88, "", color='white', transform=ax.transAxes, fontsize=22, weight='bold')

def init():
    """Initialize animation."""
    line_current.set_data([], [])
    line_previous.set_data([], [])
    text_box.set_text("")
    return line_current, line_previous, text_box

def animate(i):
    """Update the plot for each frame."""
    file = files[i]
    wave, flux = load_spec(file)
    teff_i = 10 ** spt[i][1]
    
    color = cmap(norm(teff_i))
    
    # Plot current spectrum
    line_current.set_data(wave, flux)
    line_current.set_color(color)
    
    # If not the first frame, plot the previous spectrum with lower alpha
    if i > 0:
        prev_file = files[i - 1]
        prev_wave, prev_flux = load_spec(prev_file)
        prev_teff_i = 10 ** spt[i - 1][1]
        prev_color = cmap(norm(prev_teff_i))
        line_previous.set_data(prev_wave, prev_flux)
        line_previous.set_color(prev_color)
    else:
        line_previous.set_data([], [])
    
    # Update the text and its color
    text_box.set_text(f"{spt[i][0]} (T$_\mathrm{{eff}}$ = {teff_i:.0f} K)")
    text_box.set_color(color)
    
    return line_current, line_previous, text_box

# Create animation with looping
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(spt), interval=400, blit=True, repeat=False)

# Save the animation as a GIF
save_path = '/home/dario/phd/pRT_input/stellar_spectra_animation.gif'
ani.save(save_path, writer='pillow', fps=2)
print(f' Animation saved to {save_path}!')
# save last frame as png
save_path = '/home/dario/phd/pRT_input/stellar_spectra_last_frame.png'
fig.savefig(save_path, dpi=300, facecolor='darkgray')
print(f' Last frame saved to {save_path}!')

# Show the animation
# plt.close(fig)
