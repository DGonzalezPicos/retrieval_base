import numpy as np
import matplotlib.pyplot as plt


from petitRADTRANS import Radtrans
import petitRADTRANS.poor_mans_nonequ_chem as pm


species = {'CO': 'CO_high', 
           '13CO': 'CO_36_high',
           'H2O': 'H2O_pokazatel_main_iso',
           'CN': 'CN_main_iso',
           }
atmo = Radtrans(line_species = list(species.values()), 
                            rayleigh_species = ['H2', 'He'],
                            continuum_opacities = ['H2-H2', 'H2-He'],
                            wlen_bords_micron = [1.90, 2.50],
                            mode='lbl',
                            lbl_opacity_sampling=3,
                        )

temperature = [2500.]
pressure_bar = [0.1]

temp = np.array(temperature)
pressure_bar = np.array(pressure_bar)

temp = temp.reshape(1)
pressure_bar = pressure_bar.reshape(1)
wlen_cm, opas = atmo.get_opa(temp)
wlen_nm = wlen_cm * 1e7


CO = 0.55
FeH = 0.0
ab = pm.interpol_abundances(CO * np.ones_like(temp),
                                    FeH * np.ones_like(temp),
                                    temp,
                                    pressure_bar)
ab['13CO'] = ab['CO'] / 89.
ab['CN'] = ab['CO'] / 1000.
ab = {k: v for k, v in sorted(ab.items(), key=lambda item: item[1])}

fig, ax = plt.subplots(1,1, figsize=(8,8))

for i, s in enumerate(species.keys()):
    print(f'{s}: {species[s]}')
    if s not in ab.keys():
        print(f'WARNING: {s} not in ab.keys()')
        continue
    ax.plot(wlen_nm, ab[s] * opas[species[s]], lw=2.5, label=s, alpha=0.4)

ax.set(xlabel='wavelength (nm)', ylabel=r'opacity (cm$^2$ g$^{-1}$)', yscale='log')
ax.legend()
plt.show()


               