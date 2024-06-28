import numpy as np
import matplotlib.pyplot as plt
import pathlib
from scipy.signal import medfilt

from retrieval_base.auxiliary_functions import blackbody

path = pathlib.Path(__file__).parent.absolute()
# print(path)
file = path / '1102-3430.txt'
wave, flux, err, flag = np.loadtxt(file, skiprows=0).T
mask = flag > 0
print(f' Number of flagged points: {np.sum(mask)}')

# mask points beyond 33 micron
mask_wave = wave > 34
flux[mask_wave] = np.nan

# clip 3 sigma outliers
flux_medfilt = medfilt(flux, kernel_size=5)
mask_clip = np.abs(flux - flux_medfilt) > 3*err
flux[mask_clip] = np.nan

wave_cm = wave * 1e-4  # [microns] -> [cm]
flux_Jy = np.copy(flux)
err_Jy = np.copy(err)
# convert Jy to [erg cm^{-2} s^{-1} Hz^{-1}]
flux *= 1e-23
# convert [erg cm^{-2} s^{-1} Hz^{-1}] -> [erg cm^{-2} s^{-1} cm^{-1}]
flux *= 2.998e10 / wave_cm**2 # wave in cm
# Convert [erg cm^{-2} s^{-1} cm^{-1}] -> [erg cm^{-2} s^{-1} nm^{-1}]
flux *= 1e-7
flux_units = 'erg cm$^{-2}$ s$^{-1}$ nm$^{-1}$'

err = err * (1e-23) * (2.998e10 / wave_cm**2) * 1e-7

# flux[mask] = np.nan

fig, ax_spec = plt.subplots(4,1, figsize=(8,6))
ax = ax_spec[:3]
ax_res = ax_spec[-1]

# plot two blackbodies with temperatures T1 and T2

T = [2480, # Photosphere
     520,  # Inner disk (hot gas)
    #  250,  # Mid plane warm gas??
     80,   # Outer disk (cold dust)
     ]
bb_list = []

# rjup to cm
R_jup = np.array([2.80, 
                  18.,
                #   20.,
                  400,
                  ])
R_cm =  R_jup * 7.1492e9 # [R_jup] -> [cm]
# pc to cm
d_cm = 59.17 * 3.086e18 # [pc] -> [cm]
for i, T_i in enumerate(T):
    bb_list.append(blackbody(wave_cm, T_i) * (R_cm[i] / d_cm)**2)
    ax[-1].plot(wave, bb_list[-1], label=f'T = {T_i} K\nR= {R_jup[i]:.1f} R$_{{Jup}}$', alpha=0.4)

model = sum(bb_list)
ax[-1].plot(wave, model, color='magenta')

res = flux - model
ax_res.plot(wave, res, color='black')
ax_res.axhline(0, color='magenta', ls='-', lw=0.5)
ax_res.set(xscale='log', yscale='linear', ylabel='Residuals')

ax[0].errorbar(wave, flux_Jy, yerr=err_Jy, fmt='o', color='black', ms=1, alpha=0.8)
ax[0].set_ylabel('Flux density / Jy')
for i, axi in enumerate(ax[1:]):
    axi.errorbar(wave, flux, yerr=err, fmt='o', color='black', ms=1, alpha=0.8)
    axi.errorbar(wave[mask_clip], flux[mask_clip], yerr=err[mask_clip], fmt='o', color='red', ms=1)

    
ax[-1].set(xscale='log', yscale='log')


xlim = (np.nanmin(wave), np.nanmax(wave))
ax[0].set_xlim(xlim)
ax[1].set_xlim(xlim)
ax[-1].legend(ncol=len(T), loc='upper right', frameon=False, fontsize=8)
ax[-1].set(xlim=xlim, ylim=(1e-18, None))
ax[-1].set_xlabel(r'Wavelength / $\mu$m')
# ax[1].set_ylabel(f'Flux / {flux_units}')
# common ylabel for two last axes
fig.text(0.03, 0.45, f'Flux / {flux_units}', va='center', rotation='vertical')
fig_name = path / 'spitzer_quickstart.pdf'
fig.savefig(fig_name, bbox_inches='tight')
print(f' Figure saved to {fig_name}')

plt.close(fig)
