import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
import time

c = 3e5  # Speed of light in km/s
def calculate_dv(wave):
    dv = c * np.diff(wave).mean() / wave.mean()
    return dv

def gaussian_variable_kernel(fwhm, wave, truncate=4.0):
    spacing = np.gradient(wave)  # Wavelength spacing at each point
    sigma = (fwhm / c) * wave / (2.355 * spacing)  # Convert FWHM to Gaussian sigma in wavelength units
    lw = int(np.ceil(truncate * sigma.max()))  # Half-width of kernel in index units
    x = np.arange(-lw, lw + 1)  # Kernel grid in index units
    kernels = np.exp(-0.5 * (x[None, :] / sigma[:, None]) ** 2)  # 2D Gaussian kernel array
    kernels /= kernels.sum(axis=1, keepdims=True)  # Normalize each kernel
    return kernels, lw

def instrumental_broaden_spatial(wave, flux, fwhm_array):
    kernels, lw = gaussian_variable_kernel(fwhm_array, wave)  # Generate Gaussian kernels
    flux_padded = np.pad(flux, (lw, lw), mode='reflect')  # Reflect padding for edge handling
    # Perform convolution with sliding window and kernels
    flux_broadened = np.array([
        np.sum(flux_padded[i:i + 2 * lw + 1] * kernels[i])
        for i in range(len(flux))
    ])
    return flux_broadened

def instrumental_broaden_variable_fwhm(wave, flux, fwhm_array):
    """
    Broadens the given flux by convolving it with a wavelength-dependent 
    Gaussian kernel, representing the spectrograph's resolving power at each
    wavelength. The convolution is performed in the Fourier domain.

    Parameters
    ----------
    wave : array_like
        The current wavelength grid (1D).
    flux : array_like
        The current flux (1D).
    fwhm_array : array_like
        An array of full-width half-maximum (FWHM) values (in km/s) 
        corresponding to each wavelength point.

    Raises
    ------
    ValueError
        If any value in `fwhm_array` is negative or the lengths of inputs do not match.

    Returns
    -------
    numpy.ndarray
        The broadened flux with the same shape as the input flux.
    """

    if len(wave) != len(flux) or len(wave) != len(fwhm_array):
        raise ValueError("Lengths of wave, flux, and fwhm_array must be equal.")
    if np.any(fwhm_array < 0):
        raise ValueError("FWHM values must be non-negative.")

    # Compute velocity spacing (dv) between adjacent wavelength points
    dv = calculate_dv(wave)  # Assumes this function exists or is provided

    # Fourier frequencies
    freq = np.fft.rfftfreq(flux.shape[-1], d=dv)

    # Fourier transform of flux
    flux_ff = np.fft.rfft(flux)

    # Convert FWHM to sigma
    sigma_array = fwhm_array / 2.355

    # Build the Gaussian kernel directly in Fourier space
    # Using broadcasting to handle sigma_array without loops
    kernel_matrix = np.exp(-2 * (np.pi * freq[:, np.newaxis] * sigma_array[np.newaxis, :]) ** 2)

    # Average the kernel across all sigma values (efficiently combine them)
    broadening_kernel = kernel_matrix.mean(axis=1)

    # Apply the broadening kernel in Fourier space
    flux_ff *= broadening_kernel

    # Inverse Fourier transform to return to real space
    flux_final = np.fft.irfft(flux_ff, n=flux.shape[-1])
    return flux_final

# Generate a toy model spectrum
def generate_toy_spectrum(wave):
    """
    Generate a toy spectrum with Gaussian emission lines and a continuum.

    Parameters
    ----------
    wave : array_like
        Wavelength grid.

    Returns
    -------
    flux : numpy.ndarray
        Toy spectrum flux values.
    """
    # Continuum
    flux = 0.5 + 0.01 * np.sin(0.01 * wave)

    # Add Gaussian lines
    line_centers = [4500, 4550, 4580]
    line_amplitudes = [1.0, 0.8, 1.2]
    line_widths = 0.5 * np.ones(len(line_centers))  # Gaussian sigma in wavelength units

    for center, amplitude, width in zip(line_centers, line_amplitudes, line_widths):
        flux += amplitude * np.exp(-0.5 * ((wave - center) / width) ** 2)

    return flux

# Define test parameters
wave = np.linspace(4430, 4700, 2000)  # Wavelength grid
flux = generate_toy_spectrum(wave)
fwhm_min, fwhm_max = 100, 70  # Min and max FWHM in km/s
fwhm_array = np.linspace(fwhm_min, fwhm_max, len(wave))

def time_broadening(func):
    start_time = time.time()
    result = func(wave, flux, fwhm_array)
    end_time = time.time()
    print(f"{func.__name__} time: {end_time - start_time:.4f} seconds")
    return result

flux_broadened_kernel = time_broadening(instrumental_broaden_spatial)
flux_broadened_fft = time_broadening(instrumental_broaden_variable_fwhm)
flux_broadened_fft_2 = time_broadening(instrumental_broaden_variable_fwhm)
flux_broadened_kernel = time_broadening(instrumental_broaden_spatial)

# Plot the results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

# Plot the spectra
ax1.plot(wave, flux, label="Original Spectrum", lw=2)
ax1.plot(wave, flux_broadened_kernel, label="Spatial-Domain Broadening", linestyle="--", lw=2)
ax1.plot(wave, flux_broadened_fft, label="Fourier-Domain Broadening", linestyle=":", lw=2)
ax1.set_ylabel("Flux", fontsize=14)
ax1.set_title("Comparison of Broadening Methods", fontsize=16)
ax1.legend(fontsize=12)
ax1.grid(alpha=0.3)

# Plot the residuals
residuals = flux_broadened_kernel - flux_broadened_fft
ax2.plot(wave, residuals, label="Residuals (Spatial - Fourier)", lw=2, color='r')
ax2.set_xlabel("Wavelength (Å)", fontsize=14)
ax2.set_ylabel("Residuals", fontsize=14)
ax2.legend(fontsize=12)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()
