import pickle
import os
import shutil
import wget
import pathlib

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d, generic_filter

from astropy.io import fits
import pandas as pd

import petitRADTRANS.nat_cst as nc

spirou_sample = {
                '338B': [(3952, 4.71, -0.08, 4.14), None],
                '880': [(3720, 4.72, 0.21, 6.868), '17'],
                 '15A': [(3603, 4.86, -0.30, 3.563), None],
                '411':  [(3563, 4.84, 0.12, 2.547),None], # TODO: double check this target
                '832':  [(3590, 4.70, 0.06, 4.670),None],  # Tilipman+2021
                '752A': [(3558, 4.76, 0.10, 3.522),None], # Cristofari+2022
                '849':  [(3530, 4.78, 0.37, 8.803),None], # Cristofari+2022
                '436':  [(3479, 4.78, 0.01, 9.756),None], # Cristofari+2022
                '725A': [(3441, 4.87, -0.23, 3.522),None],# Cristofari+2022
                '687':  [(3413, 4.80, 0.10, 4.550),None], # Cristofari+2022
                '876' : [(3366, 4.80, 0.10, 4.672),None], # Moutou+2023, no measurement for logg, Z

                '725B': [(3345, 4.96, -0.30, 3.523),None],
                '699':  [(3228.0, 5.09, -0.40, 1.827),'1'], # SPHINX2 seems noisier? changed RV aligment to 0.5 km/s and decreased n_edge
                '15B':  [(3218, 5.07, -0.30, 3.561),None],
                '1151': [(3178, 4.71, -0.04, 8.043),None], # Lehmann+2024, I call it `gl` but it's `gj`
                '905':  [(2930, 5.04, 0.23, 3.155),None],
}

def read_spirou_sample_csv():
    return pd.read_csv('/home/dario/phd/retrieval_base/paper/data/fundamental_parameters.csv')
    


def get_path(return_pathlib=False):
    
    cwd = os.getcwd()
    if 'dgonzalezpi' in cwd:
        path = '/home/dgonzalezpi/retrieval_base/'
        import matplotlib
        matplotlib.use('Agg') # disable interactive plots
    if 'dario' in cwd:
        path = '/home/dario/phd/retrieval_base/'
        
    if return_pathlib:
        return pathlib.Path(path)
    return path

def pickle_save(file, object_to_pickle):

    with open(file, 'wb') as f:
        pickle.dump(object_to_pickle, f)
        
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
        
    if not os.path.exists(prefix+'output'):
        os.makedirs(prefix+'output')

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
                
                # finite = np.isfinite(m_flux_i)

                # Function to interpolate the model spectrum
                m_interp_func = interp1d(
                    m_wave_i[np.isfinite(m_flux_i)], 
                    m_flux_i[np.isfinite(m_flux_i)], 
                    bounds_error=False, fill_value=np.nan
                    )
                    
            # Select only the pixels within this order
            mask_ij = d_spec.mask_isfinite[i,j,:]
            if np.sum(mask_ij) == 0:
                continue

            d_wave_ij = d_spec.wave[i,j,mask_ij]
            d_flux_ij = d_spec.flux[i,j,mask_ij]

            if LogLike is not None:
                # Scale the data instead of the models
                d_flux_ij /= LogLike.f[i,j]
            
            if Cov is not None:
                # Use the covariance matrix to weigh 
                # the cross-correlation coefficients
                cov_ij = Cov[i,j]

            if m_spec_wo_species is not None:
                # Perform the cross-correlation on the residuals
                d_flux_ij -= m_spec_wo_species.flux[i,j,mask_ij]

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


def weigh_alpha(contr_em, pressure, temperature, ax, alpha_min=0.8, 
                T_max=None, plot=False, n_layers=300):
    ''' Overplot white areas on the temperature-pressure diagram to indicate
    where the emission contribution is low. This is done by weighing the
    opacity by the emission contribution and then setting a minimum opacity
    value.
    '''

    contr_em_weigh = contr_em / contr_em.max()
    contr_em_weigh_interp = interp1d(pressure, contr_em_weigh)

    # extended vector (oversampled)
    p = np.logspace(np.log10(pressure.min()), np.log10(pressure.max()), n_layers)
    
    if T_max is None:
        T_max = np.max(temperature)
    t = np.linspace(0, T_max, p.size)
    if isinstance(alpha_min, float):
        alpha_min_vec = np.ones_like(p) * alpha_min
    else:
        alpha_min_vec = np.array(alpha_min)
    
    alpha_list = []
    for i_p in range(len(p)-1):
        mean_press = np.mean([p[i_p], p[i_p+1]])
        # print(f'{i_p}: alpha_min = {alpha_min_vec[i_p]}')
        alpha = min(1. - contr_em_weigh_interp(mean_press), alpha_min_vec[i_p])
        # print(f'{i_p}: alpha = {alpha}')
        if plot:
            ax.fill_between(t, p[i_p+1], p[i_p], color='white',alpha=alpha,
                            lw=0, 
                            rasterized=True, 
                            zorder=4,
                            )
        alpha_list.append(alpha)
    return alpha_list

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


def compare_evidence(ln_Z_A, ln_Z_B):
    '''Convert log-evidences of two models to a sigma confidence level
    
    Adapted from samderegt/retrieval_base'''

    from scipy.special import lambertw as W
    from scipy.special import erfcinv

    ln_list = [ln_Z_B, ln_Z_A]
    
    for i in range(2):
        ln_list = ln_list[::-1] if i == 1 else ln_list
        labels = ['A', 'B'] if i == 1 else ['B', 'A']
        ln_B = ln_list[0] - ln_list[1]
        B = np.exp(ln_B)
        p = np.real(np.exp(W((-1.0/(B*np.exp(1))),-1)))
        sigma = np.sqrt(2)*erfcinv(p)
        
        print(f'{labels[0]} vs. {labels[1]}: ln(B)={ln_B:.2f} | sigma={sigma:.2f}')
    return B, sigma

def load_romano_models(mass_range='1_8', Z_min=None):
    
    mass_range_files = {'1_8': 'abunda.bncmrkTHIN18_300',
                        '3_8': 'abunda.bncmrkTHIN_300'}
    assert mass_range in ['1_8', '3_8'], f'Invalid mass range: {mass_range}'
    
    file = get_path(return_pathlib=True) / 'paper/data' / mass_range_files[mass_range]
    assert file.exists(), f'File {file} does not exist'
    data = np.loadtxt(file, skiprows=0)
    
    Z = data[:, 1]
    c12 = data[:, 5] / 12
    c13 = data[:, 6] / 13
    c12c13 = c12 / c13

    o16 = data[:, 12] / 16
    o18 = data[:, 13] / 18
    o16o18 = o16 / o18
    if Z_min is not None:
        mask = (Z > Z_min)
        Z = Z[mask]
        c12c13 = c12c13[mask]
        o16o18 = o16o18[mask]
    return Z, c12c13, o16o18
