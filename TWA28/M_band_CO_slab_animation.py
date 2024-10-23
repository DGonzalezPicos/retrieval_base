import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import pathlib
import pickle
import json
# pdf pages
from matplotlib.backends.backend_pdf import PdfPages
from petitRADTRANS import Radtrans
import petitRADTRANS.nat_cst as nc

from retrieval_base.auxiliary_functions import get_path, pickle_load, blackbody
from retrieval_base.slab_model import Disk
from retrieval_base.slab_grid import SlabGrid
from retrieval_base.spectrum import ModelSpectrum

path = pathlib.Path(get_path())
line_species = ['CO_high_Sam', 'H2O_pokazatel_main_iso']
rayleigh_species = ['H2', 'He']
continuum_species = ['H2-H2', 'H2-He']
cloud_species = None
mode = 'lbl'
lbl_opacity_sampling = 10
wave_range = (4.5, 4.7)
grating = 'g395h'

target = 'TWA28'
run = 'lbl15_G2G3_3'
PT = pickle_load(path / target / f'retrieval_outputs/{run}/test_data/bestfit_PT.pkl')
pressure = PT.pressure
temperature = PT.temperature
mass_fractions = {'CO_high_Sam': 10**(-3.52), 'H2O_pokazatel_main_iso':10**(-3.63), 'H2': 0.74, 'He': 0.24}
mass_fractions['MMW'] = 2.33
mass_fractions = {k:v*np.ones_like(temperature) for k,v in mass_fractions.items()}
logg=3.4

d_spec = pickle_load(path / target / f'retrieval_outputs/{run}/test_data/d_spec_NIRSpec.pkl')
d_wave = d_spec.wave.flatten()
d_mask = (d_wave > wave_range[0]*1e3) & (d_wave < wave_range[1]*1e3)
assert np.sum(d_mask) > 0, f'No data in the range {wave_range}'
d_wave = d_wave[d_mask]
d_flux = d_spec.flux.flatten()[d_mask]


bestfit = json.load(open(path / target / f'retrieval_outputs/{run}/test_data/bestfit.json'))
params = bestfit['params']



atm = Radtrans(
                line_species=line_species, 
                rayleigh_species=rayleigh_species, 
                continuum_opacities=continuum_species, 
                cloud_species=cloud_species, 
                wlen_bords_micron=wave_range, 
                mode=mode, 
                lbl_opacity_sampling=lbl_opacity_sampling, 
                )

# Set up the atmospheric layers
atm.setup_opa_structure(pressure)



# define parameters for flux calculation
def get_flux(atm, temperature, mass_fractions, params):
    atm.calc_flux(
                    temperature, 
                    mass_fractions, 
                    gravity=10.0**params['log_g'], 
                    mmw=mass_fractions['MMW'], 
                    contribution=False, 
                    )
    # end_cf = time.time()
    # print(f'Order {i} took {end_cf-start_cf:.3f} s to compute the flux')
    wave = nc.c / atm.freq
    finite = np.isfinite(atm.flux)
    assert np.sum(finite) == len(finite), f'NaNs in flux ({np.sum(~finite)} non-finite values)'
    flux = np.where(np.isfinite(atm.flux), atm.flux, 0.0)        
    # [erg cm^{-2} s^{-1} Hz^{-1}] -> [erg cm^{-2} s^{-1} cm^{-1}]
    flux = atm.flux *  nc.c / (wave**2)

    # Convert [erg cm^{-2} s^{-1} cm^{-1}] -> [erg cm^{-2} s^{-1} nm^{-1}]
    # flux /= 1e7
    flux = flux * 1e-7

    # Convert [cm] -> [nm]
    wave *= 1e7
    # print(f'[pRT_model.get_model_spectrum] np.nanmean(np.diff(wave)) = {1e-3 * np.nanmean(np.diff(wave))} um')

    # Convert to observation by scaling with planetary radius
    flux *= (
        (params.get('R_p', 1.0)*nc.r_jup_mean) / \
        (1e3/params['parallax']*nc.pc)
        )**2

    m_spec = ModelSpectrum(
                    wave=wave, flux=flux, 
                    lbl_opacity_sampling=lbl_opacity_sampling
                    )
    # Apply radial-velocity shift, rotational/instrumental broadening
    # start_sbr = time.time()
    m_spec.shift_broaden_rebin(
        rv=params['rv'], 
        vsini=params['vsini'], 
        epsilon_limb=params['epsilon_limb'], 
        # out_res=d_resolution[i], # NEW 2024-05-26: resolution per order
        grating=grating, # NEW 2024-05-26: grating per order
        in_res=m_spec.resolution, 
        rebin=False, 
        instr_broad_fast=False,
        )

    rebin = False
    if rebin:
        m_spec.rebin_spectres(d_wave=d_wave[None,:], replace_wave_flux=True, numba=True)
        
    return m_spec

m_spec = get_flux(atm, temperature, mass_fractions, params)

## Disk emission model

get_slab = True
if get_slab:
    grating = 'g395h'
    T_ex_range = np.arange(300.0, 800.0+50.0, 50.0)
    N_mol_range = np.logspace(15, 20, 6*2)
    species = '12CO'

    slab = SlabGrid(species=species, grating=grating, path=path)
    slab.get_grid(T_ex_range, N_mol_range, cache=True)
    slab.load_interpolator(del_flux_grid=True) # False to plot the grid
    flux_slab = slab.interpolate(T_ex=600.0, N_mol=1e18, A_au=2e-3, d_pc=params['d_pc'])
    
R_jup = nc.r_jup_mean
wave_cm = m_spec.wave*1e-7
bb = blackbody(wave_cm, params['T_d']) * (params['R_d']*R_jup / (params['d_pc'] * nc.pc))**2


# Setup figure
fig, ax = plt.subplots(figsize=(12, 7), tight_layout=True, facecolor='none')
ax.set_facecolor('none')
# Data and initial plot
lw = 1.5
ax.plot(d_wave, d_flux, lw=lw, label='Data', color='k')
atm_line, = ax.plot(m_spec.wave, m_spec.flux, lw=lw, label='Atm.')
bb_line, = ax.plot(m_spec.wave, bb, lw=lw, label=r'BB (T$_d$)' + f' = {params["T_d"]:.0f} K', ls='--')
slab_line, = ax.plot(m_spec.wave, np.zeros_like(m_spec.wave), lw=lw, label='Slab')
total_line, = ax.plot(m_spec.wave, m_spec.flux + np.zeros_like(m_spec.wave), lw=lw)

# ax.set_title(f'{target} 12CO atmospheric model with disk')
ax.set_xlabel('Wavelength / nm')
ax.set_ylabel('Flux / erg s$^{-1}$ cm$^{-2}$ nm$^{-1}$')
ax.set_xlim(wave_range[0]*1e3, wave_range[1]*1e3)
ax.set_ylim(-1e-16, 2e-15)
ax.legend()


# T_d_range = np.arange(300, 700, 10)
log_N_mol_range = np.concatenate([np.arange(13, 16, 0.2),
                                  np.arange(16, 16.5, 0.05), 
                                  np.arange(16.5, 16.9, 0.01),
                                  np.arange(16.9, 17.4, 0.05),
                                  np.arange(17.4, 20, 0.2)])
# Normalize temperatures for colormap
norm = plt.Normalize(vmin=log_N_mol_range.min(), vmax=log_N_mol_range.max())
cmap = plt.get_cmap('gist_earth_r')

# Create colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation='vertical', label=r'$\log\ N_{\mathrm{mol}}$ (cm$^{-2}$)',
                     pad=0.01, aspect=10)

# Add a vertical bar that will move with the animation to indicate the current temperature
cbar_marker = cbar.ax.plot([0.5], [log_N_mol_range[0]], marker="o", markersize=50, color='w', lw=10)[0]

# Update function for animation
def update(frame):
    
    log_N_mol = log_N_mol_range[frame]
    
    # Update blackbody flux and total flux and label with the current temperature
    slab_flux = slab.interpolate(T_ex=600.0, N_mol=10**log_N_mol, A_au=3e-3, d_pc=params['d_pc'])
    slab_flux = np.interp(m_spec.wave, slab.wave_grid*1e3, slab_flux)
    slab_line.set_ydata(slab_flux)
    
    total_line.set_ydata(m_spec.flux + bb + slab_flux)
    # total_line.set_label(f'Atm. + BB + Slab (log N_mol = {log_N_mol:.0e} cm$^{-2}$)')
    # Update the color of the blackbody line based on temperature
    slab_line.set_color(cmap(norm(log_N_mol)))
    slab_line.set_label(f'Slab (log N_mol = {log_N_mol:.1f}' + r' cm$^{-2}$)')
    
    # update title at each frame
    # ax.set_title(f'{target} 12CO atmospheric model with disk, BB Temp = {T_d:.0f}K')
    # add text with temperature
    
    # Move the marker on the colorbar
    cbar_marker.set_ydata([log_N_mol])
    ax.get_legend().remove()
    ax.legend(loc='upper right', ncol=2, frameon=False)
    ax.set_facecolor('none')
    # Update the plot title with the current temperature
    return slab_line, total_line, cbar_marker

# Create animation
ani = FuncAnimation(fig, update, frames=len(log_N_mol_range), blit=True, interval=100)

# Show animation
plt.show()
# Save the animation as a .gif using PillowWriter
save = True
if save:
    animations_folder = path / target / 'animations'
    animations_folder.mkdir(exist_ok=True)
    gif_filename = animations_folder / 'slab_Nmol_animation.gif'
    ani.save(gif_filename, writer=PillowWriter(fps=10), dpi=300, savefig_kwargs={'transparent': True})

    print(f"Animation saved as {gif_filename}")