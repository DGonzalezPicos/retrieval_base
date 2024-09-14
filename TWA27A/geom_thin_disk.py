import numpy as np
from scipy.constants import h, c, k
from scipy.integrate import quad

from retrieval_base.auxiliary_functions import blackbody
# Constants
pi = np.pi

# Planck's constant, speed of light, and Boltzmann constant in CGS units
h_cgs = h * 1e7  # Planck's constant in erg s
c_cgs = c * 1e2  # speed of light in cm/s
k_cgs = k * 1e7  # Boltzmann constant in erg/K

# Vectorized Planck's law for blackbody radiation
# def blackbody(wave, T):
#     """Planck's law for blackbody radiation. 
#        Takes wave (wavelength in cm) and T (temperature in K) as inputs, both can be arrays
       
#          Returns the blackbody flux density in erg/s/cm^2/cm."""
#     wave = np.asarray(wave)
#     return (2 * h_cgs * c_cgs**2 / wave**5) / (np.exp(h_cgs * c_cgs / (wave * k_cgs * T)) - 1)

# Disk temperature profile Td(r)
def T_disk(r, T_star, Rp, q=0.75):
    """Temperature profile for the disk, returns the temperature at each radius."""
    return T_star * (2 / (3 * pi))**0.25 * (r / Rp)**-q # TODO: check this eq.

# Emission from the disk (fully vectorized)
def disk_emission(wave_nm, T_star, Rp, Rcav, Rout, i, D, q=0.75, n_rings=100):
    """Calculate the total flux emission from the disk at an array of wavelengths (nm).
       Uses full vectorization to avoid looping over wavelengths."""
    
    # Convert wavelength from nm to cm (1 nm = 1e-7 cm)
    wave_cm = wave_nm * 1e-7
    
    # Create an array of radii for the disk (using n_rings for integration precision)
    r_values = np.linspace(Rcav, Rout, n_rings)
    
    # Compute the temperature profile for the disk at all radii
    T_values = T_disk(r_values, T_star, Rp, q=q)
    # mask T < 10 K, ignore this region --> negligible contribution
    # speed up computation
    mask = T_values > 60.0
    # print(f' mask: {mask.sum()} / {mask.size}')
    
    r_values = r_values[mask]
    T_values = T_values[mask]
    
    # Compute the Planck function (blackbody) for all wavelengths and all radii
    # Shape of blackbody_grid: (n_rings, len(wave_cm))
    blackbody_grid = blackbody(wave_cm[:, np.newaxis], T_values)
    print(f' blackbody_grid: {blackbody_grid.shape}')
    # Integrate over the disk radii for all wavelengths
    integrand = 2 * pi * r_values * blackbody_grid * np.cos(i)
    
    # Perform trapezoidal integration over the radius (axis=1)
    flux = np.trapz(integrand, r_values, axis=1) / D**2
    print(f' mean flux: {np.mean(flux)}')
    return flux


# Example usage
if __name__ == "__main__":
    # Parameters (example values)
    T_star = 2300.0  # Temperature of the central object in Kelvin
    # 1 jupiter radius in cm
    Rp = 7.1492e9  # Radius of the central object in cm
    Rcav = 2 * Rp    # Inner disk radius in cm
    Rout = 30 * Rp    # Outer disk radius in cm
    i = np.radians(45)  # Inclination angle in radians
    q = 0.75  # Temperature profile exponent
    
    pc_cm = 3.086e18  # 1 parsec in cm
    D = 50.0 * pc_cm  # Distance to the object in cm
    
    # Wavelength array (in nm)
    wavelength_nm = np.linspace(1600, 5300, 10000)  # 500 nm to 2000 nm
    
    # Calculate disk flux at the given wavelength array in erg/s/cm^2/nm
    # flux = disk_emission(wavelength_nm, T_star, Rp, Rcav, Rout, i, D, q=q, n_rings=100)
    m_wave, m_flux = np.load("/home/dario/phd/retrieval_base/TWA27A/retrieval_outputs/lbl15_KM_6/test_data/bestfit_model_no_disk.npy")
    m_wave = m_wave.flatten()
    m_flux = m_flux.flatten()
    

    import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1,2, figsize=(14,6), gridspec_kw={'width_ratios': [3, 2]})
    # create gridspec with 6 columns and 2 rows
    from matplotlib import gridspec
    fig = plt.figure(figsize=(14,6))
    gs = fig.add_gridspec(nrows=3, ncols=6)
    ax_disk = fig.add_subplot(gs[0, :4])
    ax_disk_temp = fig.add_subplot(gs[0, 4:])
    
    ax_spec = fig.add_subplot(gs[1, :])
    ax_12CO = fig.add_subplot(gs[2, :])
    
    
    
    r_disk = np.linspace(Rcav, Rout, 100)

    for q in np.arange(0.50, 1.2, 0.15):
        f_i = disk_emission(wavelength_nm, T_star, Rp, Rcav, Rout, i, D, q=q, n_rings=100)
        f_i = np.interp(m_wave, wavelength_nm, f_i)
        f_i_norm = f_i / np.nanmax(f_i)
        # ax[0].plot(wavelength_nm, f_i, label=f'q={q:.2f}')
        ax_disk.plot(m_wave, f_i_norm, label=f'q={q:.2f}')
        
        temp_disk = T_disk(r_disk, T_star, Rp,q=q)
        ax_disk_temp.plot(r_disk / Rp, temp_disk)
        
        # m_flux_i = m_flux
        # ax_spec.plot(m_wave, m_flux_i, label=f'q={q:.2f}')
        xlim_12CO = (2170, 2380)
        for axi in [ax_spec, ax_12CO]:
            
            mask = np.ones_like(m_wave, dtype=bool)
            if axi==ax_12CO:
                mask = (m_wave > xlim_12CO[0]) & (m_wave < xlim_12CO[1])
            
            axi.plot(m_wave[mask], m_flux[mask]+f_i[mask], label=f'q={q:.2f}')
    
        
    ax_12CO.set(xlabel='Wavelength (nm)', ylabel='Flux (erg/s/cm$^2$/nm)', xlim=xlim_12CO)  
    ax_disk.set_xlabel('Wavelength (nm)')
    ax_disk.set_ylabel('Flux'+r' (erg/s/cm$^2$/nm)')
    ax_disk.set_title('Disk Emission Spectrum')
    ax_disk.legend()
    
    # plot temperature profile of the disk
    
    
    ax_disk_temp.set(xlabel='r/Rp', ylabel='Temperature (K)',
           title='Disk Temperature Profile')
    
    # compare to simple blackbody at T_BB = 500.0 K and size R_BB = np.sqrt(R_out**2 - R_cav**2)
    bb_flux = blackbody(wavelength_nm * 1e-7, 600.0)
    bb_flux *= (pi * (Rout**2 - Rcav**2) / D**2)
    ax_disk.plot(wavelength_nm, bb_flux / bb_flux.max(), 'k--', label='T=500 K')
    
    
    plt.show()