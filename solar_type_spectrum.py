import matplotlib.pyplot as plt
import numpy as np
from expecto import get_spectrum
from scipy.ndimage import gaussian_filter

def instr_broadening(wave, flux, out_res=1e6, in_res=1e6):

    # Delta lambda of resolution element is FWHM of the LSF's standard deviation
    sigma_LSF = np.sqrt(1/out_res**2 - 1/in_res**2) / \
                (2*np.sqrt(2*np.log(2)))

    spacing = np.mean(2*np.diff(wave) / (wave[1:] + wave[:-1]))

    # Calculate the sigma to be used in the gauss filter in pixels
    sigma_LSF_gauss_filter = sigma_LSF / spacing
    
    # Apply gaussian filter to broaden with the spectral resolution
    flux_LSF = gaussian_filter(flux, sigma=sigma_LSF_gauss_filter, 
                                mode='nearest'
                                )
    return flux_LSF

settings = {
    'K2166': np.array([
        [[1921.318,1934.583], [1935.543,1948.213], [1949.097,1961.128]],
        [[1989.978,2003.709], [2004.701,2017.816], [2018.708,2031.165]],
        [[2063.711,2077.942], [2078.967,2092.559], [2093.479,2106.392]],
        [[2143.087,2157.855], [2158.914,2173.020], [2173.983,2187.386]],
        [[2228.786,2244.133], [2245.229,2259.888], [2260.904,2274.835]],
        [[2321.596,2337.568], [2338.704,2353.961], [2355.035,2369.534]],
        [[2422.415,2439.061], [2440.243,2456.145], [2457.275,2472.388]],
        ]), 
    }

def plot_settings(ax, setting='K2166', wave_units='um', color='k', alpha=0.1, **kwargs):
    n_order, n_det, _ = settings[setting].shape
    for i in range(n_order):
        for j in range(n_det):
            wave_min, wave_max = settings[setting][i][j]
            wave_factor = 1e-3 if wave_units == 'um' else 1.0
            label = kwargs.pop('label', None) if (i+j)==0 else None
            ax.axvspan(wave_min * wave_factor, wave_max * wave_factor, color=color, alpha=alpha, label=label, **kwargs)
    return ax

# teff_range = np.arange(5200, 5600+100, 100)
teff_range = np.arange(3000, 9000, 1000)
teff_colors = plt.cm.coolwarm_r(np.linspace(0, 1, len(teff_range)))
wave_range = [1900, 2490]
fig, (ax, ax_zoom) = plt.subplots(2,1,figsize=(12, 4))
zoom_wave_range = [2351, 2353.9]
for i, teff in enumerate(teff_range):
    spectrum = get_spectrum(
        T_eff=teff, log_g=4.5, cache=True
    )

    wave = spectrum.wavelength.value * 1e-1 # A -> nm
    flux = spectrum.flux.value

    wave_mask = (wave > wave_range[0]) & (wave < wave_range[1])
    flux_LSF = instr_broadening(wave[wave_mask], flux[wave_mask], out_res=1e5, in_res=1e6)
    wave = wave[wave_mask]
    for j, axx in enumerate([ax, ax_zoom]):
        
        
        if i == 0 and j == 0:
            plot_settings(axx, setting='K2166', wave_units='nm', color='k', alpha=0.1, label='CRIRES+ K2166')
            
        mask_j = np.ones_like(wave, dtype=bool) if j == 0 else (wave > zoom_wave_range[0]) & (wave < zoom_wave_range[1])
        axx.plot(wave[mask_j], flux_LSF[mask_j] / np.nanmedian(flux_LSF[mask_j]), label=f'{teff} K', color=teff_colors[i], alpha=0.9)

ax.legend(ncol=2)
# ax_zoom.legend()

ax_zoom.set_xlim(zoom_wave_range)
ax.axvspan(zoom_wave_range[0], zoom_wave_range[1], color='blue', alpha=0.1)
ax.set_xlim(wave_range)
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Flux [normalized]')
ax.set_title('K-band spectrum (R=100,000)')
ax_zoom.set_xlabel('Wavelength (nm)')
ax_zoom.set_ylabel('Flux [normalized]')
# ax_zoom.set_title('Zoomed in on the H2O line at 2352.9 nm')

# plt.show()
png = True
fig_name = 'stellar_spectrum.pdf'
if png:
    fig_name = fig_name.replace('.pdf', '.png')
    fig.savefig(fig_name, dpi=300, bbox_inches='tight')
else:
    fig.savefig(fig_name, bbox_inches='tight')
print(f'Saved figure to {fig_name}')
plt.close()