import pickle
import os
import shutil
import wget

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d, generic_filter, median_filter, gaussian_filter
from scipy.signal import savgol_filter

from astropy.io import fits

import petitRADTRANS.nat_cst as nc
pi = 3.14159265358979323846
rjup_cm = 7.1492e9 # 1 Jupiter radius in cm

def get_path():
    
    cwd = os.getcwd()
    if 'dgonzalezpi' in cwd:
        path = '/home/dgonzalezpi/retrieval_base/'
        import matplotlib
        matplotlib.use('Agg') # disable interactive plots
    if 'dario' in cwd:
        path = '/home/dario/phd/retrieval_base/'
        
    return path
    

def pickle_save(file, object_to_pickle):

    with open(file, 'wb') as f:
        pickle.dump(object_to_pickle, f)
    print(f'--> Saved {file}')
        
def pickle_load(file):
    
    with open(file, 'rb') as f:
        pickled_object = pickle.load(f)

    return pickled_object

def create_output_dir(prefix, file_params):
    
    # Make the output directory
    if not os.path.exists('/'.join(prefix.split('/')[:-1])):
        os.makedirs('/'.join(prefix.split('/')[:-1]))

    # Make the plots directory
    if not os.path.exists(prefix+'plots'):
        os.makedirs(prefix+'plots')

    # Make the data directory
    if not os.path.exists(prefix+'data'):
        os.makedirs(prefix+'data')

    # Make a copy of the parameters file
    shutil.copy(file_params, prefix+'data/'+file_params)

def quantiles(x, q, weights=None, axis=-1):
    '''
    Compute (weighted) quantiles from an input set of samples.
    Parameters
    ----------
    x : `~numpy.ndarray` with shape (nsamps,)
        Input samples.
    q : `~numpy.ndarray` with shape (nquantiles,)
       The list of quantiles to compute from `[0., 1.]`.
    weights : `~numpy.ndarray` with shape (nsamps,), optional
        The associated weight from each sample.
    Returns
    -------
    quantiles : `~numpy.ndarray` with shape (nquantiles,)
        The weighted sample quantiles computed at `q`.
    '''

    # Initial check.
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    # Quantile check.
    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError('Quantiles must be between 0. and 1.')

    if weights is None:
        # If no weights provided, this simply calls `np.percentile`.
        return np.percentile(x, list(100.0 * q), axis=axis)
    else:
        # If weights are provided, compute the weighted quantiles.
        weights = np.atleast_1d(weights)
        if len(x) != len(weights):
            raise ValueError('Dimension mismatch: len(weights) != len(x).')
        idx = np.argsort(x)  # sort samples
        sw = weights[idx]  # sort weights
        cdf = np.cumsum(sw)[:-1]  # compute CDF
        cdf /= cdf[-1]  # normalize CDF
        cdf = np.append(0, cdf)  # ensure proper span
        quantiles = np.interp(q, cdf, x[idx]).tolist()
        return quantiles

def CCF_to_SNR(rv, CCF, ACF=None, rv_to_exclude=(-100,100)):

    # Select samples from outside the expected peak
    rv_mask = (rv < rv_to_exclude[0]) | (rv > rv_to_exclude[1])

    # Correct the offset
    CCF -= np.nanmean(CCF[rv_mask])

    if ACF is not None:
        # Correct the offset
        ACF -= np.nanmean(ACF[rv_mask])

        # Subtract the (scaled) ACF before computing the standard-deviation
        excluded_CCF = (CCF - ACF*(CCF/ACF)[rv==0])[rv_mask]
    else:
        excluded_CCF = CCF[rv_mask]

    # Standard-deviation of the cross-correlation function
    std_CCF = np.nanstd(excluded_CCF)
    
    # Convert to signal-to-noise
    CCF_SNR = CCF / std_CCF
    if ACF is not None:
        ACF_SNR = ACF*(CCF/ACF)[rv==0]/std_CCF
    else:
        ACF_SNR = None
    
    return CCF_SNR, ACF_SNR, (CCF - ACF*(CCF/ACF)[rv==0])/std_CCF
    
def CCF(d_spec, 
        m_spec, 
        m_wave_pRT_grid, 
        m_flux_pRT_grid, 
        m_spec_wo_species=None, 
        m_flux_wo_species_pRT_grid=None, 
        LogLike=None, 
        Cov=None, 
        rv=np.arange(-500,500+1e-6,1), 
        apply_high_pass_filter=True, 
        ):

    CCF = np.zeros((d_spec.n_orders, d_spec.n_dets, len(rv)))
    d_ACF = np.zeros_like(CCF)
    m_ACF = np.zeros_like(CCF)

    # Loop over all orders and detectors
    for i in range(d_spec.n_orders):

        if m_wave_pRT_grid is not None:
            m_wave_i = m_wave_pRT_grid[i]
            m_flux_i = m_flux_pRT_grid[i].copy()

            if m_flux_wo_species_pRT_grid is not None:
                # Perform the cross-correlation on the residuals
                m_flux_i -= m_flux_wo_species_pRT_grid[i].copy()
                #m_flux_i = m_flux_wo_species_pRT_grid[i].copy()

            # Function to interpolate the model spectrum
            m_interp_func = interp1d(
                m_wave_i, m_flux_i, bounds_error=False, fill_value=np.nan
                )
        
        for j in range(d_spec.n_dets):

            if m_wave_pRT_grid is None:
                m_wave_i = m_spec.wave[i,j]
                m_flux_i = m_spec.flux[i,j]

                # Function to interpolate the model spectrum
                m_interp_func = interp1d(
                    m_wave_i[np.isfinite(m_flux_i)], 
                    m_flux_i[np.isfinite(m_flux_i)], 
                    bounds_error=False, fill_value=np.nan
                    )
                    
            # Select only the pixels within this order
            mask_ij = d_spec.mask_isfinite[i,j,:]

            d_wave_ij = d_spec.wave[i,j,mask_ij]
            d_flux_ij = d_spec.flux[i,j,mask_ij]

            # if LogLike is not None:
            #     # Scale the data instead of the models
            #     d_flux_ij /= LogLike.f[i,j]
            
            if Cov is not None:
                # Use the covariance matrix to weigh 
                # the cross-correlation coefficients
                cov_ij = Cov[i,j]

            if m_spec_wo_species is not None:
                # Perform the cross-correlation on the residuals
                d_flux_ij -= LogLike.f[:,i,j] @ m_spec_wo_species.flux[:,i,j,mask_ij]
                # d_flux_ij -= np.sum(LogLike.f[:,i,j,None] * m_spec_wo_species.flux[:,i,j,mask_ij],axis=0)

            # Function to interpolate the observed spectrum
            d_interp_func = interp1d(
                d_wave_ij, d_flux_ij, bounds_error=False, 
                fill_value=np.nan
                )
            
            # Create a static model template
            m_flux_ij_static = m_interp_func(d_wave_ij)

            if apply_high_pass_filter:
                # Apply high-pass filter
                d_flux_ij -= gaussian_filter1d(d_flux_ij, sigma=300, mode='reflect')
                m_flux_ij_static -= gaussian_filter1d(
                    m_flux_ij_static, sigma=300, mode='reflect'
                    )
            
            
            for k, rv_k in enumerate(rv):

                # Apply Doppler shift
                d_wave_ij_shifted = d_wave_ij * (1 + rv_k/(nc.c*1e-5))

                # Interpolate the spectra onto the new wavelength grid
                d_flux_ij_shifted = d_interp_func(d_wave_ij_shifted)
                m_flux_ij_shifted = m_interp_func(d_wave_ij_shifted)

                if apply_high_pass_filter:
                    # Apply high-pass filter
                    d_flux_ij_shifted -= gaussian_filter1d(
                        d_flux_ij_shifted, sigma=300, mode='reflect'
                        )
                    m_flux_ij_shifted -= gaussian_filter1d(
                        m_flux_ij_shifted, sigma=300, mode='reflect'
                        )

                # Compute the cross-correlation coefficients, weighted 
                # by the covariance matrix
                if Cov is None:
                    CCF[i,j,k]   = np.nansum(m_flux_ij_shifted*d_flux_ij)# / np.isfinite((m_flux_ij_shifted*d_flux_ij)).sum()
                    m_ACF[i,j,k] = np.nansum(m_flux_ij_shifted*m_flux_ij_static)# / np.isfinite((m_flux_ij_shifted*m_flux_ij_static)).sum()
                    d_ACF[i,j,k] = np.nansum(d_flux_ij_shifted*d_flux_ij)# / np.isfinite((d_flux_ij_shifted*d_flux_ij)).sum()
                else:
                    CCF[i,j,k] = np.dot(
                        m_flux_ij_shifted, cov_ij.solve(d_flux_ij)
                        )
                    # Auto-correlation coefficients
                    m_ACF[i,j,k] = np.dot(
                        m_flux_ij_shifted, cov_ij.solve(m_flux_ij_static)
                        )
                    d_ACF[i,j,k] = np.dot(
                        d_flux_ij_shifted, cov_ij.solve(d_flux_ij)
                        )

            # Scale the correlation coefficients
            if LogLike is not None:
                CCF[i,j,:]   *= LogLike.f[i,j]/LogLike.beta[i,j]**2
                m_ACF[i,j,:] *= LogLike.f[i,j]/LogLike.beta[i,j]**2
                d_ACF[i,j,:] *= LogLike.f[i,j]/LogLike.beta[i,j]**2

    return rv, CCF, d_ACF, m_ACF

def get_PHOENIX_model(T, log_g, FeH=0, wave_range=(500,3000), PHOENIX_path='./data/PHOENIX/'):

    # Only certain combinations of parameters are allowed
    T_round     = int(200*np.round(T/200))
    log_g_round = 0.5*np.round(log_g/0.5)
    FeH_round   = 0.5*np.round(FeH/0.5)
    FeH_sign    = '+' if FeH>0 else '-'

    file_name = 'lte{0:05d}-{1:.2f}{2}{3:.1f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
    file_name = file_name.format(
        T_round, log_g_round, FeH_sign, np.abs(FeH_round)
        )
    
    # Join the file name to the url
    url = 'ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/' + file_name
    wave_url = 'ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS//WAVE_PHOENIX-ACES-AGSS-COND-2011.fits'

    # Make the directory
    if not os.path.exists(PHOENIX_path):
        os.mkdir(PHOENIX_path)

    file_path = os.path.join(PHOENIX_path, file_name)
    wave_path = os.path.join(PHOENIX_path, 'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')

    if not os.path.exists(wave_path):
        print(f'Downloading wavelength grid from {wave_url}')
        wget.download(wave_url, wave_path)

    if not os.path.exists(file_path):
        print(f'Downloading PHOENIX spectrum from {url}')
        wget.download(url, file_path)

    # Load the spectrum
    hdu = fits.open(file_path)
    PHOENIX_flux = hdu[0].data.astype(float) # erg s^-1 cm^-2 cm^-1

    hdu = fits.open(wave_path)
    PHOENIX_wave = hdu[0].data.astype(float) / 10 # Convert from A to nm

    mask_wave = (PHOENIX_wave > wave_range[0]) & \
        (PHOENIX_wave < wave_range[1])
    
    return PHOENIX_wave[mask_wave], PHOENIX_flux[mask_wave]

def read_results(prefix, n_params):

    import json
    import pymultinest

    # Set-up analyzer object
    analyzer = pymultinest.Analyzer(
        n_params=n_params, 
        outputfiles_basename=prefix
        )
    stats = analyzer.get_stats()

    # Load the equally-weighted posterior distribution
    posterior = analyzer.get_equal_weighted_posterior()
    posterior = posterior[:,:-1]

    # Read the parameters of the best-fitting model
    bestfit = np.array(stats['modes'][0]['maximum a posterior'])

    PT = pickle_load(f'{prefix}data/bestfit_PT.pkl')
    Chem = pickle_load(f'{prefix}data/bestfit_Chem.pkl')

    m_spec = pickle_load(f'{prefix}data/bestfit_m_spec.pkl')
    d_spec = pickle_load(f'{prefix}data/d_spec.pkl')

    LogLike = pickle_load(f'{prefix}data/bestfit_LogLike.pkl')

    try:
        Cov = pickle_load(f'{prefix}data/bestfit_Cov.pkl')
    except:
        Cov = None

    int_contr_em           = np.load(f'{prefix}data/bestfit_int_contr_em.npy')
    int_contr_em_per_order = np.load(f'{prefix}data/bestfit_int_contr_em_per_order.npy')
    int_opa_cloud          = np.load(f'{prefix}data/bestfit_int_opa_cloud.npy')

    f = open(prefix+'data/bestfit.json')
    bestfit_params = json.load(f)
    f.close()

    res = (
        posterior, 
        bestfit, 
        PT, 
        Chem, 
        int_contr_em, 
        int_contr_em_per_order, 
        int_opa_cloud, 
        m_spec, 
        d_spec, 
        LogLike, 
        Cov, 
        bestfit_params
        )

    return res

def blackbody(wave_cm, T):
    '''Calculate Blackbody spectrum in erg/s/cm^2/nm
    '''
    # nc = natural constants [cgs]
    bb = 2*nc.h*nc.c**2/wave_cm**5 * 1/(np.exp(nc.h*nc.c/(wave_cm*nc.kB*T))-1)
    # [erg/s/cm^2/cm/steradian] -> [erg/s/cm^2/cm]
    bb *= np.pi
    # [erg/s/cm^2/cm] -> [erg/s/cm^2/nm]
    bb *= 1e-7
    return bb

def sigma_clip(y, sigma=3, width=10, max_iter=5, fun='median', replace=False, replace_w_fun=False):
    '''Sigma clipping algorithm. If replace=True, the function will replace the
    clipped values with np.nan. If replace=False, the function will return a
    boolean mask with the same shape as y.
    '''
    
    assert fun in ['median', 'gaussian', 'savgol'], 'fun must be either "median" or "gaussian"'
    
    mask_clip = np.isnan(y)
    print(f' Initial number of clipped points: {np.sum(mask_clip)}')
    clip_0 = 0
    for i in range(max_iter):
        std_y = np.nanstd(np.where(mask_clip, np.nan, y))
        # mean_y = np.nanmean(y[~mask_clip])
        # use median filter 
        
        if fun == 'median':
            mean_y = median_filter(y, width, mode='nearest')
        elif fun == 'gaussian':
            mean_y = gaussian_filter1d(y, width / 2.355, 
                                       mode='reflect',
                                       ) 
        elif fun == 'savgol':
            mean_y = savgol_filter(y, width, 2, mode='nearest')       
        
        clip = np.abs(y - mean_y) > sigma * std_y
        print(f' Iteration {i}: {np.sum(clip)} points clipped')
        
        mask_clip |= clip
        
        if (np.sum(clip) - clip_0) == 0 and i > 0:
            print(f'--> Converged after {i} iterations')
            break
        clip_0 = np.sum(clip)

        
        if replace:
            y[mask_clip] = np.nan
            # return y
        if replace_w_fun:
            # use the values from the function to replace the clipped values
            # y[mask_clip] = mean_y[mask_clip] * sigma / 2
            # interpolate over nans
            # print(f' Interpolated number of clipped points: {np.sum(mask_clip)}')
            y = np.interp(np.arange(len(y)), np.arange(len(y))[~mask_clip], y[~mask_clip])
    return y
    
    
    # return mask_clip

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

def rebin(wave, flux, nbin, err=None, debug=False):
    # Create new bins by averaging every `nbin` points
    nans = np.isnan(flux) | np.isnan(wave)
    wave, flux = wave[~nans], flux[~nans]
    
    size = len(wave)
    trimmed_size = size - (size % nbin)
    if debug:
        print(f'Original size: {size}, Trimmed size: {trimmed_size}')
    
    # Trim the arrays to be a multiple of nbin
    trimmed_wave = wave[:trimmed_size]
    trimmed_flux = flux[:trimmed_size]
    # trimmed_err = err[:trimmed_size]
    
    new_wave = trimmed_wave.reshape(-1, nbin).mean(axis=1)
    new_flux = trimmed_flux.reshape(-1, nbin).mean(axis=1)
    
    if err is not None:
        trimmed_err = err[~nans][:trimmed_size]
        new_err = np.sqrt(np.sum(trimmed_err.reshape(-1, nbin)**2, axis=1)) / nbin
        return new_wave, new_flux, new_err
    
    return new_wave, new_flux, np.std(trimmed_flux.reshape(-1, nbin), axis=1)


def make_array(arrays):
    # Determine the length of the longest array
    max_length = max(len(arr) for arr in arrays)
    
    # Pad all arrays to the max_length
    padded_arrays = [np.pad(arr, (0, max_length - len(arr)), 'constant', constant_values=np.nan) for arr in arrays]
    
    # Convert to a single numpy array
    result_array = np.array(padded_arrays)
    
    return result_array

def profile_attributes(obj, max_depth=3, depth=0, min_size_mb=0.1):
    from pympler import asizeof
    indent = " " * (depth * 4)
    if depth > max_depth:
        return
    for attr_name in dir(obj):
        print(f"{indent}{attr_name}")
        try:
            if attr_name.startswith('__') and attr_name.endswith('__'):
                continue
            attr_value = getattr(obj, attr_name)
            if callable(attr_value):
                continue
            size_bytes = asizeof.asizeof(attr_value)
            size_mb = size_bytes / (1024 ** 2)  # Convert bytes to megabytes
            if size_mb < min_size_mb:
                continue
            print(f"{indent}{attr_name}: {size_mb:.1f} MB")
            if hasattr(attr_value, '__dict__'):
                profile_attributes(attr_value, max_depth, depth + 1)
        except Exception as e:
            print(f"{indent}{attr_name}: {e}")
            pass
        
def ism_extinction(
    av_mag, rv_red, wave):
    """
    Author: Tomas Stolker
    
    Function for calculating the optical and IR extinction
    with the empirical relation from `Cardelli et al. (1989)
    <https://ui.adsabs.harvard.edu/abs/1989ApJ...345..245C/abstract>`_.

    Parameters
    ----------
    av_mag : float
        Extinction (mag) in the $V$ band.
    rv_red : float
        Reddening in the $V$ band, ``R_V = A_V / E(B-V)``. Standard diffuse ISM = 3.1.
    wave : np.ndarray, list(float), float
        Array or list with the wave (um) for which
        the extinction is calculated. It is also possible
        to provide a single value as float.

    Returns
    -------
    np.ndarray
        Extinction (mag) at ``wave``.
    """

    if isinstance(wave, float):
        wave = np.array([wave])

    elif isinstance(wave, list):
        wave = np.array(wave)

    x_wavel = 1.0 / wave
    y_wavel = x_wavel - 1.82

    a_coeff = np.zeros(x_wavel.size)
    b_coeff = np.zeros(x_wavel.size)

    indices = np.where(x_wavel < 1.1)[0]

    if len(indices) > 0:
        a_coeff[indices] = 0.574 * x_wavel[indices] ** 1.61
        b_coeff[indices] = -0.527 * x_wavel[indices] ** 1.61

    indices = np.where(x_wavel >= 1.1)[0]

    if len(indices) > 0:
        a_coeff[indices] = (
            1.0
            + 0.17699 * y_wavel[indices]
            - 0.50447 * y_wavel[indices] ** 2
            - 0.02427 * y_wavel[indices] ** 3
            + 0.72085 * y_wavel[indices] ** 4
            + 0.01979 * y_wavel[indices] ** 5
            - 0.77530 * y_wavel[indices] ** 6
            + 0.32999 * y_wavel[indices] ** 7
        )

        b_coeff[indices] = (
            1.41338 * y_wavel[indices]
            + 2.28305 * y_wavel[indices] ** 2
            + 1.07233 * y_wavel[indices] ** 3
            - 5.38434 * y_wavel[indices] ** 4
            - 0.62251 * y_wavel[indices] ** 5
            + 5.30260 * y_wavel[indices] ** 6
            - 2.09002 * y_wavel[indices] ** 7
        )

    return av_mag * (a_coeff + b_coeff / rv_red)

def apply_extinction(
    flux, wave, av_mag, rv_red=3.1):
    """
    Apply extinction to a spectrum. The extinction is calculated with
    the empirical relation from `Cardelli et al. (1989)
    <https://ui.adsabs.harvard.edu/abs/1989ApJ...345..245C/abstract>`_.

    Parameters
    ----------
    flux : np.ndarray
        Spectrum to which the extinction is applied.
    wave : np.ndarray
        Wavelengths (um) of the spectrum.

    av_mag : float  
        Extinction (mag) in the $V$ band.

    rv_red : float
        Reddening in the $V$ band, ``R_V = A_V / E(B-V)``. Standard diffuse ISM = 3.1.

    Returns
    -------
    np.ndarray
        Spectrum with extinction applied.
    """
    
    # extinction = ism_extinction(av_mag, rv_red, wave)
    return flux * 10 ** (-0.4 * ism_extinction(av_mag, rv_red, wave))


# Disk temperature profile Td(r)
def T_disk(r, T_star, R_p, q=0.75):
    """Temperature profile for the disk, returns the temperature at each radius."""
    return T_star * (2 / (3 * pi))**0.25 * (r / R_p)**-q # TODO: check this eq.

# Emission from the disk (fully vectorized)
def geom_thin_disk_emission(wave_nm, T_star, R_p, R_cav, R_out, i=0.785, d_pc=1.0, q=0.75, n_rings=100):
    """Calculate the total flux emission from the disk at an array of wavelengths (nm).
    
       Uses full vectorization to avoid looping over wavelengths.
       
       Parameters
         ----------
            wave_nm : np.ndarray
                Array of wavelengths in nm.
            T_star : float
                Temperature of the star in K.
            R_p : float
                Radius of the planet in R_jup.
            R_cav : float
                Inner disk radius in R_jup.
            R_out : float
                Outer disk radius in R_jup.
            i : float
                Inclination angle in radians.
            d_pc : float
                Distance to the system in parsecs.
            q : float
                Temperature profile exponent.
            n_rings : int
                Number of rings used for the integration.

        Returns
        -------
            np.ndarray
                Array of fluxes units of erg/s/cm^2/nm.
    """
                
    
    # Convert wavelength from nm to cm (1 nm = 1e-7 cm)
    wave_cm = wave_nm * 1e-7
    
    # Create an array of radii for the disk (using n_rings for integration precision)
    r_values = np.linspace(R_cav, R_out, n_rings) * rjup_cm
    
    # Compute the temperature profile for the disk at all radii
    T_values = T_disk(r_values, T_star, R_p * rjup_cm, q=q)
    # mask T < 10 K, ignore this region --> negligible contribution
    # speed up computation
    mask = T_values > 60.0
    # print(f' mask: {mask.sum()} / {mask.size}')
    r_values = r_values[mask]
    T_values = T_values[mask]
    
    # Compute the Planck function (blackbody) for all wavelengths and all radii
    # Shape of blackbody_grid: (n_rings, len(wave_cm))
    blackbody_grid = blackbody(wave_cm[:, np.newaxis], T_values)
    # print(f' blackbody_grid: {blackbody_grid.shape}')
    # Integrate over the disk radii for all wavelengths
    integrand = 2 * pi * r_values * blackbody_grid * np.cos(i)
    
    # Perform trapezoidal integration over the radius (axis=1)
    flux = np.trapz(integrand, r_values, axis=1) / (d_pc * 3.086e18)**2
    # print(f' mean flux: {np.mean(flux)}')
    return flux

def apply_PT_cutoff(atm, T_min, T_max, P_min=1e-4, P_max=1e2):
    """ Apply a cutoff to the PT grid of the custom line opacities to reduce memory usage and speed up the retrieval.
    
    `atm` is a petitRADTRANS.Radtrans object
    Pressure in bars.
    """
   
    # convert P from bar to cgs
    P_min_cgs = P_min * 1e6
    P_max_cgs = P_max * 1e6
    
    for i, species in enumerate(atm.line_species):
        if atm.custom_grid[species]:
            new_custom_line_TP_grid = [] # old has shape (152, 2) for each PT pair
            new_custom_line_paths = []
            new_line_grid_kappas_custom_PT = []
            for j, (T_j, P_j) in enumerate(atm.custom_line_TP_grid[species]):
                # print(f' T_j = {T_j}, P_j = {P_j}')
                if T_j >= T_min and T_j <= T_max and P_j >= P_min_cgs and P_j <= P_max_cgs:
                    new_custom_line_TP_grid.append([T_j, P_j])
                    new_custom_line_paths.append(atm.custom_line_paths[species][j])
                    new_line_grid_kappas_custom_PT.append(atm.line_grid_kappas_custom_PT[species][:,:,j])
                
            print(f' Number of PT pairs {species}:\n -> before = {len(atm.custom_line_TP_grid[species])} \n -> after = {len(new_custom_line_TP_grid)}')
            # save new values
            atm.custom_line_TP_grid[species] = np.array(new_custom_line_TP_grid)
            atm.custom_line_paths[species] = np.array(new_custom_line_paths)
            atm.line_grid_kappas_custom_PT[species] = np.moveaxis(np.array(new_line_grid_kappas_custom_PT), 0, 2)
            
            
            Ts = np.unique(np.array(new_custom_line_TP_grid)[:,0])
            atm.custom_diffTs[species] = len(Ts)
            
            Ps = np.unique(np.array(new_custom_line_TP_grid)[:,1])
            atm.custom_diffPs[species] = len(Ps)
    return atm