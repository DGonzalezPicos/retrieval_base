import numpy as np
import matplotlib.pyplot as plt


from petitRADTRANS import Radtrans
import petitRADTRANS.poor_mans_nonequ_chem as pm


species = {'CO': 'CO_high', 
           'H2O': 'H2O_pokazatel_main_iso',
           'Ca': 'Ca',
           'Mg': 'Mg',
           }
atmo = Radtrans(line_species = list(species.values()), 
                            rayleigh_species = ['H2', 'He'],
                            continuum_opacities = ['H2-H2', 'H2-He'],
                            wlen_bords_micron = [1.90, 2.50],
                            mode='lbl',
                            lbl_opacity_sampling=3,
                        )

fig, ax = plt.subplots(1,1, figsize=(14,4))

temperature_range = np.arange(2000, 3100, 200)
colors = plt.cm.inferno(np.linspace(0, 1, len(temperature_range)))
for i, temperature in enumerate(temperature_range):
    temperature = [temperature]
    pressure_bar = [0.3]

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
    ab['Ca'] = ab['CO'] / 5.
    ab['Mg'] = ab['CO'] / 5.
    ab = {k: v for k, v in sorted(ab.items(), key=lambda item: item[1])}


    # for i, s in enumerate(species.keys()):
    #     print(f'{s}: {species[s]}')
    #     if s not in ab.keys():
    #         print(f'WARNING: {s} not in ab.keys()')
    #         continue
    #     # alpha = 0.9 if s == 'Ca' else 0.3
    #     alpha = 0.9 if s == 'Ca' else 0.3
    #     ax.plot(wlen_nm, ab[s] * opas[species[s]], lw=2.5, label=f'{s} ({float(ab[s]):.1e})', alpha=alpha)
    ax.plot(wlen_nm, ab['Ca'] * opas[species['Ca']], lw=2.5, color=colors[i], alpha=0.7)
    
    if i == 0:
        ax.plot(wlen_nm, ab['CO'] * opas[species['CO']], color='brown')
        ax.plot(wlen_nm, ab['H2O'] * opas[species['H2O']], color='deepskyblue')
        ax.plot(wlen_nm, ab['Mg'] * opas[species['Mg']], lw=2.5, color='limegreen', alpha=0.7)

    
im = ax.scatter([], [], c=[], cmap='inferno', vmin=temperature_range[0], vmax=temperature_range[-1])
cbar = fig.colorbar(im, ax=ax, label='temperature (K)')


ax.set(xlabel='wavelength (nm)', ylabel=r'opacity (cm$^2$ g$^{-1}$)', yscale='log')
ax.set_title(f'T = {temperature[0]} K, P = {pressure_bar[0]} bar', fontsize=22)
ax.set(ylim=(1e-4,None))
ax.legend()
plt.show()
fig.savefig('Ca_Mg_opacity_temperature.png', bbox_inches='tight', dpi=300)

               