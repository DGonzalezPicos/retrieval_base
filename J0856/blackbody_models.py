
import numpy as np
import matplotlib.pyplot as plt
import petitRADTRANS.nat_cst as nc

def bb(T, wave_cm):
    ''' Blackbody flux in erg/s/cm^2/cm '''
    bb_lambda = 2*nc.h*nc.c**2/(wave_cm**5)  # [erg/s/cm^2/cm/sr]
    bb_lambda *= 1/(np.exp(nc.h*nc.c/(wave_cm*nc.kB*T)) - 1) # [erg/s/cm^2/cm/sr]
    bb_lambda *= np.pi # [erg/s/cm^2/cm]
    return bb_lambda

# wave = np.linspace(1.90, 2.50, 10000) * 1e-3 # [cm]
wave_nm = np.linspace(1900, 2500, 10000) # [nm]
wave_cm = wave_nm * 1e-7 # [cm]
kteff_range = np.arange(15, 18, 0.5)



fig, ax = plt.subplots()
for kteff in kteff_range:
    T = 10**kteff
    bb_T = bb(T, wave_cm) 
    
    ax.plot(wave_nm, bb_T / np.mean(bb_T), label=f'{kteff} K')

ax.set(xlabel='Wavelength [nm]', ylabel='Flux [erg/s/cm$^2$/cm]')
ax.legend()
plt.show()