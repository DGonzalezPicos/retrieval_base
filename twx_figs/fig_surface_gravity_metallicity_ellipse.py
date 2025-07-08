import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
from retrieval_base.retrieval import Retrieval
import retrieval_base.auxiliary_functions as af
from retrieval_base.config import Config
import seaborn as sns
from matplotlib.patches import Rectangle, Ellipse
import h5py
from scipy.stats import chi2
import matplotlib.patheffects as PathEffects
pe_white = PathEffects.withStroke(linewidth=2, foreground="white")

fontsize = 14
plt.rcParams['font.size'] = fontsize
plt.rcParams['axes.linewidth'] = 2.0

# Configuration - choose x-axis parameter, for now only work with '12CO'
X_AXIS_PARAM = "[C/H]"  # Options: "C/O", "[C/H]", "12C/13C", "12CO", "H2O"
calculate_metallicity_from_carbon_monoxide = True

def metallicity_from_carbon_monoxide(volume_mixing_ratio_CO=1e-3, H2=0.85):
    """
    Calculate metallicity from carbon monoxide volume mixing ratio.
    H2/He atmosphere (85%, 15% He, <1% metals)
    """
    ratios = {'C/H_solar': 10.0**(8.46-12)}
    
    ratios['C/H'] = volume_mixing_ratio_CO / H2
    ratios['[C/H]'] = np.log10(ratios['C/H']) - np.log10(ratios['C/H_solar'])
    
    return ratios['[C/H]']

def setup_paths():
    path = af.get_path(return_pathlib=True)
    path_figures = pathlib.Path('/home/dario/phd/twa2x_paper/figures')
    return path, path_figures

def define_runs_and_colors():
    runs = {
        'TWA27A': [
            ('freeslab_lbl10_G1G2G3_1', 'G1+G2+G3'),
            ('freeslab_lbl10_G2G3_2', 'G2+G3'), # update to index 2
        ],
        'TWA28': [
            ('freeslab_lbl10_G1G2G3_1', 'G1+G2+G3'),
            ('freeslab_lbl10_G2G3_1', 'G2+G3'), # update to index 1

        ]
    }
    
    # Publication-quality colorblind-friendly palette with better contrast
    # colors = {
    #     'TWA28': {
    #         'data': '#2C2C2C',  # Dark gray for data
    #         'model': ['#FF6B35', '#1F77B4'],  # Orange, Blue
    #         'crires': '#2E8B57',  # Sea green
    #         'object_color': '#FF6B35'  # Main color for TWA28
    #     },
    #     'TWA27A': {
    #         'data': '#733b27',
    #         'model': ['#9467BD', '#737373'],  # Purple, Grey
    #         'zhang2025': 'black',
    #         'object_color': '#9467BD'  # Main color for TWA27A
    #     }
    # }
    colors = {
        'TWA28': {
            # 'data': '#2C2C2C',  # Dark gray for data
            'model': ['navy', 'purple'],  # Orange, Blue
            # 'crires': '#2E8B57',  # Sea green
            'crires':'#507356',
            'object_color': '#FF6B35'  # Main color for TWA28
        },
        'TWA27A': {
            # 'data': '#733b27',
            'model': ['royalblue', 'plum'],  # Purple, Grey
            # 'zhang2025': 'black',
            'object_color': '#9467BD'  # Main color for TWA27A
        }
    }
    
    # IGNORE CMAPS
    cmaps = {
        # 'TWA28': ['Oranges', 'Blues', 'BuGn'],
        'TWA28': ['PuBu', 'OrRd', 'BuGn'],
        'TWA27A': ['Purples', 'RdPu']
    }
    
    return runs, colors, cmaps

def load_data(path, target, run, cache=True):
    cwd = os.getcwd()
    if target not in cwd:
        os.chdir(f'{path}/{target}')
        print(f'Changed directory to {target}')
    
    config_file = 'config_jwst.txt'    
    conf = Config(path=path, target=target, run=run)(config_file)
    
    log_g_posterior_file = f'{conf.prefix}data/log_g_posterior.npy'
    chem_posterior_file = f'{conf.prefix}data/chem_posterior.h5'
    files = [log_g_posterior_file, chem_posterior_file]
    
    posterior = None
    
    # Check if we need to regenerate data
    need_regenerate = not cache or not all(os.path.exists(file) for file in files)
    
    # Also check if HDF5 file exists but is incomplete
    if not need_regenerate and os.path.exists(chem_posterior_file):
        try:
            with h5py.File(chem_posterior_file, 'r') as f:
                # Check if required groups exist
                if 'COH_posterior' not in f or 'VMRs_posterior' not in f:
                    need_regenerate = True
                    print("HDF5 file incomplete, regenerating...")
        except Exception as e:
            need_regenerate = True
            print(f"Error reading HDF5 file, regenerating: {e}")
    
    if need_regenerate:
        ret = Retrieval(conf=conf, evaluation=False)

        _, posterior = ret.PMN_analyze()
        log_g_index = list(ret.Param.param_keys).index('log_g')
        log_g_posterior = posterior[:,log_g_index]
        np.save(log_g_posterior_file, log_g_posterior)
        print(f'Saved {log_g_posterior_file}')
        
        _ = ret.get_PT_mf_envelopes(posterior=posterior, n_samples=None, cache=cache)
        
        # Save chemistry posteriors as separate datasets in HDF5
        with h5py.File(chem_posterior_file, 'w') as f:
            # Save COH_posterior dictionary
            coh_grp = f.create_group('COH_posterior')
            for key, value in ret.Chem.COH_posterior.items():
                coh_grp.create_dataset(key, data=value, compression='gzip', compression_opts=9)
            
            # Save VMRs_posterior dictionary
            vmrs_grp = f.create_group('VMRs_posterior')
            for key, value in ret.Chem.VMRs_posterior.items():
                vmrs_grp.create_dataset(key, data=value, compression='gzip', compression_opts=9)
            
            # Save metadata
            f.attrs['target'] = target.encode('utf-8')
            f.attrs['run'] = run.encode('utf-8')
            f.attrs['param_keys'] = [key.encode('utf-8') for key in ret.Param.param_keys]
            
        print(f'Saved chemistry posteriors to {chem_posterior_file}')
        
        # Return the chemistry objects
        COH_posterior = ret.Chem.COH_posterior
        VMRs_posterior = ret.Chem.VMRs_posterior
        
    else:
        # Load chemistry posteriors from HDF5
        with h5py.File(chem_posterior_file, 'r') as f:
            # Load COH_posterior dictionary
            COH_posterior = {}
            for key in f['COH_posterior'].keys():
                COH_posterior[key] = f['COH_posterior'][key][:]
            
            # Load VMRs_posterior dictionary
            VMRs_posterior = {}
            for key in f['VMRs_posterior'].keys():
                VMRs_posterior[key] = f['VMRs_posterior'][key][:]
        
        print(f'Loaded chemistry posteriors from {chem_posterior_file}')
             
    log_g_posterior = np.load(log_g_posterior_file)
    
    return COH_posterior, VMRs_posterior, log_g_posterior

def get_posteriors(COH_posterior, VMRs_posterior):
    CO_posterior = np.mean(COH_posterior['C'] / COH_posterior['O'], axis=-1)
    
    # Choose metallicity calculation method
    if calculate_metallicity_from_carbon_monoxide:
        # Use custom function to calculate metallicity from CO
        CO_vmr = np.mean(VMRs_posterior['12CO'], axis=-1)
        H2_fraction = 0.85  # Assumed H2 fraction
        CH_posterior = np.array([metallicity_from_carbon_monoxide(co, H2_fraction) 
                                for co in CO_vmr])
    else:
        # Use standard method from auxiliary_functions
        CH_posterior = af.solar_metallicity(
            np.mean(COH_posterior['C'], axis=-1),
            np.mean(COH_posterior['H'], axis=-1)
        )
    
    isotope_ratios = {
        '12C/13C': np.mean(VMRs_posterior['12CO'] / VMRs_posterior['13CO'], axis=-1)
    }
    
    my_posteriors = {
        'C/O': CO_posterior,
        '[C/H]': CH_posterior,
        '12C/13C': isotope_ratios['12C/13C'],
        '12CO': VMRs_posterior['12CO'],
        'H2O': VMRs_posterior['H2O']
    }
    
    return my_posteriors

def get_parameter_label(param):
    """Get LaTeX formatted label for parameter"""
    labels = {
        'C/O': r'C/O',
        '[C/H]': r'[C/H]',
        '12C/13C': r'$^{12}$C/$^{13}$C',
        '12CO': r'log $^{12}$CO',
        'H2O': r'log H$_2$O'
    }
    return labels.get(param, param)

def get_axis_limits(param, data_values):
    """Get appropriate axis limits for different parameters"""
    if param == 'C/O':
        return (0.3, 1.2)
    elif param == '[C/H]':
        return (-1.0, 1.0)
    elif param == '12C/13C':
        return (20, 200)
    elif param == '12CO':
        return (-6, -2)
    elif param == 'H2O':
        return (-6, -2)
    else:
        # Default: use data range with 10% padding
        data_min, data_max = np.nanmin(data_values), np.nanmax(data_values)
        padding = 0.1 * (data_max - data_min)
        return (data_min - padding, data_max + padding)

def get_filename_suffix():
    """Get filename suffix based on metallicity calculation method"""
    if calculate_metallicity_from_carbon_monoxide:
        return "_co_derived"
    else:
        return "_standard"

def load_crires_data(path, target):
    """Load CRIRES data for TWA28"""
    file_crires = path / target / f'retrieval_outputs/final_full/test_data/bestfit_Chem.pkl'
    chem_crires = af.pickle_load(file_crires)
    
    log_g_crires_file = path / target / f'retrieval_outputs/final_full/test_data/log_g_posterior.npy'
    if os.path.exists(log_g_crires_file):
        log_g_crires = np.load(log_g_crires_file)
    else:
        import pymultinest
        conf = Config(path=path, target=target, run='final_full')('config_freechem.txt')
        analyzer = pymultinest.Analyzer(
            n_params=len(conf.free_params),
            outputfiles_basename=conf.prefix
        )
        posterior = analyzer.get_equal_weighted_posterior()
        posterior = posterior[:,:-1]
        log_g_index = list(conf.free_params).index('log_g')
        log_g_crires = posterior[:,log_g_index]
        np.save(log_g_crires_file, log_g_crires)
    
    # Handle metallicity calculation based on the flag
    if calculate_metallicity_from_carbon_monoxide:
        # Calculate metallicity from CO VMR using the custom function
        CO_vmr = chem_crires.VMRs_posterior['12CO']
        H2_fraction = 0.85
        crires_metallicity = np.array([metallicity_from_carbon_monoxide(co, H2_fraction) 
                                      for co in CO_vmr])
    else:
        # Use standard Fe/H from CRIRES
        crires_metallicity = chem_crires.VMRs_posterior['Fe/H']
    
    crires_data = {
        'C/O': chem_crires.VMRs_posterior['C/O'],
        '[C/H]': crires_metallicity,
        '12C/13C': chem_crires.VMRs_posterior['12_13CO'],
        'log_g': log_g_crires,
        '12CO': np.log10(chem_crires.VMRs_posterior['12CO']),
        'H2O': np.log10(chem_crires.VMRs_posterior['H2O'])
    }
    # get quantiles for crires log_g
    q = [0.16, 0.5, 0.84]
    log_g_quantiles = np.quantile(log_g_crires, q)
    print(f"CRIRES log_g {log_g_quantiles[1]:.2f} (+{log_g_quantiles[1]-log_g_quantiles[0]:.2f} -{log_g_quantiles[2]-log_g_quantiles[1]:.2f})")
    
    return crires_data

def clean_data(log_g_posterior, x_posterior):
    """Clean the data by removing infinities and NaNs."""
    # Handle infinities and NaNs
    log_g_posterior = np.where(np.isinf(log_g_posterior), np.nan, log_g_posterior)
    x_posterior = np.where(np.isinf(x_posterior), np.nan, x_posterior)
    
    # Create mask for valid data
    mask = ~np.isnan(log_g_posterior) & ~np.isnan(x_posterior)
    
    # Apply mask and ensure we have enough data points
    log_g_clean = log_g_posterior[mask]
    x_clean = x_posterior[mask]
    
    if len(log_g_clean) < 10:
        print(f"Warning: Only {len(log_g_clean)} valid data points after cleaning")
    
    return log_g_clean, x_clean

def compute_covariance_ellipse(x_data, y_data, confidence_level=0.68):
    """
    Compute confidence ellipse using eigendecomposition of covariance matrix.
    
    Parameters:
    -----------
    x_data, y_data : array-like
        Data points
    confidence_level : float
        Confidence level (default: 0.68 for 1-sigma)
    
    Returns:
    --------
    center : tuple
        (x_center, y_center) of ellipse
    width, height : float
        Full width and height of ellipse (2 * semi-axes)
    angle : float
        Rotation angle in degrees
    semi_major, semi_minor : float
        Semi-major and semi-minor axes lengths
    """
    # Stack data
    data = np.vstack([x_data, y_data])
    
    # Compute covariance matrix
    cov = np.cov(data)
    
    # Eigendecomposition
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    
    # Sort by eigenvalue (largest first)
    order = eigenvals.argsort()[::-1]
    eigenvals = eigenvals[order]
    eigenvecs = eigenvecs[:, order]
    
    # Compute ellipse parameters
    # Chi-squared value for confidence level (2 DOF)
    chi2_val = chi2.ppf(confidence_level, df=2)
    
    # Semi-axes lengths
    semi_major = np.sqrt(eigenvals[0] * chi2_val)
    semi_minor = np.sqrt(eigenvals[1] * chi2_val)
    
    # Rotation angle (in degrees)
    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
    if angle > 90:
        angle = angle - 180
    if angle < -90:
        angle = angle + 180
    
    # Center
    center = (np.mean(x_data), np.mean(y_data))
    
    # Full width and height for matplotlib Ellipse
    width = 2 * semi_major
    height = 2 * semi_minor
    
    return center, width, height, angle, semi_major, semi_minor

def create_correlation_plot(ax, log_g_clean, x_clean, color, label, cmap, alpha=0.4):
    """Create a correlation plot with scatter points, regression line, and multiple confidence ellipses."""
    
    # Get slope and intercept of the regression line
    slope, intercept = np.polyfit(x_clean, log_g_clean, 1)
    correlation = np.corrcoef(log_g_clean, x_clean)[0, 1]
    
    # Create scatter plot with smaller round markers
    one_every = 5
    scatter = ax.scatter(x_clean[::one_every], log_g_clean[::one_every], c=color, alpha=alpha, s=2, 
                        edgecolors='none', rasterized=True, zorder=-1)
    
    # Add regression line with improved styling
    if abs(correlation) > 1.0:  # Only show regression line for moderate correlations
        sns.regplot(x=x_clean, y=log_g_clean, ax=ax, scatter=False, 
                    line_kws={'color': color, 'linewidth': 1.5, 'alpha': 0.7})
    
    # Define confidence levels and their visual properties
    confidence_levels = [0.68, 0.95, 0.99]
    line_styles = ['-', '--', '-.']
    alphas = [0.8, 0.6, 0.4]
    linewidths = [1.5, 1.25, 1.25]
    
    # Compute and draw multiple confidence ellipses
    ellipse_params = []
    ellipse_center = None
    for i, (conf_level, linestyle, alpha_val, linewidth) in enumerate(zip(confidence_levels, line_styles, alphas, linewidths)):
        center, width, height, angle, semi_major, semi_minor = compute_covariance_ellipse(
            x_clean, log_g_clean, confidence_level=conf_level
        )
        
        ellipse = Ellipse(center, width, height, angle=angle, 
                         facecolor='none', edgecolor=color, linewidth=linewidth, 
                         alpha=alpha_val, linestyle=linestyle, path_effects=[pe_white])
        ax.add_patch(ellipse)
        
        # Store parameters for the 68% ellipse (first one)
        if i == 0:
            ellipse_params = [angle, semi_major, semi_minor]
            ellipse_center = center
    
    # Print ellipse parameters (using 68% confidence ellipse)
    print(f"\n{label}:")
    print(f"  Correlation: {correlation:.3f}")
    print(f"  68% confidence ellipse:")
    print(f"    Semi-major axis: {ellipse_params[1]:.4f}")
    print(f"    Semi-minor axis: {ellipse_params[2]:.4f}")
    print(f"    Axis ratio: {ellipse_params[1]/ellipse_params[2]:.2f}")
    print(f"    Rotation angle: {ellipse_params[0]:.1f}°")
    
    # Mark intercept with improved styling (only for metallicity)
    if X_AXIS_PARAM == '[C/H]':
        ax.scatter(0, intercept, color=color, marker='s', s=20, 
                  edgecolor='white', linewidth=1, alpha=0.8, zorder=100)
    
    return correlation, slope, intercept, ellipse_params[0], ellipse_params[1], ellipse_params[2], ellipse_center

def setup_plot():
    """Create and setup the plot figure and axis with improved styling."""
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Enhanced axis styling
    ax.set_xlabel(get_parameter_label(X_AXIS_PARAM), fontsize=fontsize)
    ax.set_ylabel(r'$\log(g)$', fontsize=fontsize)
    
    ax.set_xlim(-4.05, -2.8)
    
    # Improve tick styling with increased width
    ax.tick_params(axis='both', which='major', labelsize=fontsize, width=2.0, length=6)
    ax.tick_params(axis='both', which='minor', width=1.5, length=3, labelsize=fontsize)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    return fig, ax

def create_custom_legend(ax, correlations, colors, runs):
    """Create a custom legend structure with object groupings and ellipse angles."""
    
    # Create legend elements manually for better control
    legend_elements = []
    
    # TWA28 section - just text title
    dummy_patch = plt.Line2D([0], [0], color='none')
    legend_elements.append((dummy_patch, 'TWA 28 (ρ, θ°)'))
    
    # TWA28 entries
    for i, (run, label) in enumerate(runs['TWA28']):
        color = colors['TWA28']['model'][i]
        corr = correlations.get(f"TWA28 {label}", 0)
        angle = correlations.get(f"TWA28 {label}_angle", 0)
        
        line_patch = plt.Line2D([0], [0], color=color, linewidth=2.5, alpha=0.9)
        label_text = f"  {label}: ({corr:.2f}, {angle:.0f}°)"
        legend_elements.append((line_patch, label_text))
    
    # CRIRES entry (only for TWA28 and certain parameters)
    if X_AXIS_PARAM in ['C/O', '[C/H]', '12C/13C', '12CO', 'H2O']:
        crires_corr = correlations.get('CRIRES', 0)
        crires_angle = correlations.get('CRIRES_angle', 0)
        crires_patch = plt.Line2D([0], [0], color=colors['TWA28']['crires'], 
                                 linewidth=2.5, alpha=0.9)
        crires_text = f"  {'CRIRES' + r'$^{+}$'}: ({crires_corr:.2f}, {crires_angle:.0f}°)"
        legend_elements.append((crires_patch, crires_text))
    
    # TWA27A section - just text title
    dummy_patch2 = plt.Line2D([0], [0], color='none')
    legend_elements.append((dummy_patch2, 'TWA 27A (ρ, θ°)'))
    
    # TWA27A entries
    for i, (run, label) in enumerate(runs['TWA27A']):
        color = colors['TWA27A']['model'][i]
        corr = correlations.get(f"TWA27A {label}", 0)
        angle = correlations.get(f"TWA27A {label}_angle", 0)
        
        line_patch = plt.Line2D([0], [0], color=color, linewidth=2.5, alpha=0.9)
        label_text = f"  {label}: ({corr:.2f}, {angle:.0f}°)"
        legend_elements.append((line_patch, label_text))
    
    # Create the legend above the plot without frame
    handles, labels = zip(*legend_elements)
    legend = ax.legend(handles, labels, 
                      bbox_to_anchor=(0.5, 1.02), 
                      loc='lower center',
                      ncol=2, 
                      frameon=False,
                      fontsize=fontsize*0.8,
                      columnspacing=2.0,
                      handlelength=1.2,
                      handletextpad=0.4)
    
    return legend

def add_information_box(ax):
    """Add information box with correlation and ellipse definitions."""
    info_text = (
        "ρ: Pearson coefficient\n"
        "θ: Ellipse rotation angle"
        # "Ellipses: 68% (--), 95% (-.·), 99% (···)"
    )
    
    text_bbox = dict(facecolor='white', alpha=0.8, edgecolor='gray', 
                     linewidth=1,
                     boxstyle='round,pad=0.3')
    ax.text(0.56, 0.14, info_text,
            fontsize=fontsize*0.75,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=text_bbox,
            zorder=15)

def add_grating_annotations(ax, ellipse_centers, colors):
    """Add annotations with arrows pointing to ellipse centers to indicate grating combinations."""
    
    # Find ellipse centers for each grating combination
    g2g3_data = []
    g1g2g3_data = []
    
    for key, center in ellipse_centers.items():
        if center is not None and 'CRIRES' not in key:
            target = key.split()[0]
            # Check for exact grating combination matches
            if 'G1+G2+G3' in key:
                g1g2g3_data.append({
                    'center': center,
                    'color': colors[target]['model'][0],  # Use first color for G1+G2+G3
                    'target': target
                })
            elif 'G2+G3' in key:
                g2g3_data.append({
                    'center': center,
                    'color': colors[target]['model'][1],  # Use second color for G2+G3
                    'target': target
                })
    
    if g2g3_data and g1g2g3_data:
        # Get plot limits for positioning
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        
        # Calculate centers of mass for each group
        g2g3_center_x = np.mean([d['center'][0] for d in g2g3_data])
        g2g3_center_y = np.mean([d['center'][1] for d in g2g3_data])
        g1g2g3_center_x = np.mean([d['center'][0] for d in g1g2g3_data])
        g1g2g3_center_y = np.mean([d['center'][1] for d in g1g2g3_data])
        
        # Position text boxes strategically
        # G2+G3 text position
        g2g3_text_x = xlim[0] + 0.2 * x_range
        g2g3_text_y = ylim[1] - 0.3 * y_range
        
        # G1+G2+G3 text position
        g1g2g3_text_x = xlim[1] - 0.17 * x_range
        g1g2g3_text_y = ylim[1] - 0.45 * y_range
        
        # Use representative colors for text boxes
        g2g3_text_color = g2g3_data[0]['color']  # Use first G2+G3 color
        g1g2g3_text_color = g1g2g3_data[0]['color']  # Use first G1+G2+G3 color
        
        # Add G2+G3 annotation with consistent color
        ax.text(g2g3_text_x, g2g3_text_y, 'G2+G3\n1.66-5.27 μm', 
                fontsize=fontsize*0.8, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='k', linewidth=1.0, alpha=0.95,
                         pad=0.05),
                zorder=25)
        
        # Add arrows from G2+G3 text to each G2+G3 ellipse center
        for data in g2g3_data:
            ax.annotate('', xy=data['center'], xytext=(g2g3_text_x, g2g3_text_y),
                       arrowprops=dict(arrowstyle='->', color=data['color'], 
                                     lw=2.0, alpha=0.9, shrinkA=5, shrinkB=5),
                       zorder=24)
        
        # Add G1+G2+G3 annotation with consistent color
        ax.text(g1g2g3_text_x, g1g2g3_text_y, 'G1+G2+G3\n0.97-5.27 μm', 
                fontsize=fontsize*0.8, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='k', linewidth=1.0, alpha=0.95,
                         pad=0.05),
                zorder=25)
        
        # Add arrows from G1+G2+G3 text to each G1+G2+G3 ellipse center
        for data in g1g2g3_data:
            ax.annotate('', xy=data['center'], xytext=(g1g2g3_text_x, g1g2g3_text_y),
                       arrowprops=dict(arrowstyle='->', color=data['color'], 
                                     lw=2.0, alpha=0.9, shrinkA=5, shrinkB=5),
                       zorder=24)
        
        print(f"\nAdded grating annotations:")
        print(f"  G2+G3 ellipses: {len(g2g3_data)} (text at {g2g3_text_x:.2f}, {g2g3_text_y:.2f})")
        print(f"  G1+G2+G3 ellipses: {len(g1g2g3_data)} (text at {g1g2g3_text_x:.2f}, {g1g2g3_text_y:.2f})")
        print(f"  G2+G3 text color: {g2g3_text_color}")
        print(f"  G1+G2+G3 text color: {g1g2g3_text_color}")
        for i, data in enumerate(g2g3_data):
            print(f"    G2+G3 #{i+1}: center at {data['center'][0]:.3f}, {data['center'][1]:.3f}, color: {data['color']}")
        for i, data in enumerate(g1g2g3_data):
            print(f"    G1+G2+G3 #{i+1}: center at {data['center'][0]:.3f}, {data['center'][1]:.3f}, color: {data['color']}")
    else:
        print("No ellipse centers found for grating annotations")

def finalize_plot(fig, ax, correlations, colors, ellipse_centers, runs):
    """Add final touches to the plot with improved styling."""
    
    # Create custom legend
    legend = create_custom_legend(ax, correlations, colors, runs)
    
    # Add information box
    add_information_box(ax)
    
    # Add grating annotations
    add_grating_annotations(ax, ellipse_centers, colors)
    
    # Style improvements
    ax.set_ylim(3.0, 4.55)
    
    # Add reference line only for metallicity
    if X_AXIS_PARAM == '[C/H]':
        ax.axvline(0, color='black', linestyle='--', alpha=0.6, linewidth=1.5, zorder=0)
    
    # Improve spine styling with increased width
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
        # spine.set_color('gray')
    
    # Adjust layout to accommodate legend
    plt.subplots_adjust(top=0.85, bottom=0.12, left=0.12, right=0.95)

def main():
    print(f"Creating correlation plot with {X_AXIS_PARAM} on x-axis")
    if calculate_metallicity_from_carbon_monoxide:
        print("Using CO-derived metallicity calculation")
    else:
        print("Using standard metallicity calculation")
    print("=" * 60)
    
    path, path_figures = setup_paths()
    runs, colors, cmaps = define_runs_and_colors()
    
    fig, ax = setup_plot()
    correlations = {}
    ellipse_centers = {}
    all_x_data = []  # Collect all x-axis data for scaling
    
    for target in runs.keys():
        for i, (run, label) in enumerate(runs[target]):
            # Load and process data
            COH_posterior, VMRs_posterior, log_g_posterior = load_data(path, target, run)
            my_posteriors = get_posteriors(COH_posterior, VMRs_posterior)
            
            # Get the selected x-axis parameter
            if X_AXIS_PARAM not in my_posteriors:
                print(f"Warning: Parameter {X_AXIS_PARAM} not available for {target} {run}")
                continue
                
            x_posterior = my_posteriors[X_AXIS_PARAM]
            
            # Handle VMR parameters (take mean across pressure levels and convert to log)
            if X_AXIS_PARAM in ['12CO', 'H2O']:
                x_posterior = np.mean(x_posterior, axis=-1)
                # Convert to log10 scale, handling potential negative/zero values
                x_posterior = np.where(x_posterior > 0, np.log10(x_posterior), np.nan)
                print(f"VMR range for {X_AXIS_PARAM}: {np.nanmin(x_posterior):.3f} to {np.nanmax(x_posterior):.3f}")
            
            log_g_clean, x_clean = clean_data(log_g_posterior, x_posterior)
            all_x_data.extend(x_clean)  # Collect for axis scaling
            
            # Create plot with enhanced styling
            color = colors[target]['model'][i]
            plot_label = f"{target} {label}"
            correlation, slope, intercept, angle, semi_major, semi_minor, ellipse_center = create_correlation_plot(
                ax, log_g_clean, x_clean, color, plot_label, cmaps[target][i]
            )
            
            # Store all statistics
            correlations[plot_label] = correlation
            correlations[f"{plot_label}_slope"] = slope
            correlations[f"{plot_label}_intercept"] = intercept
            correlations[f"{plot_label}_angle"] = angle
            correlations[f"{plot_label}_semi_major"] = semi_major
            correlations[f"{plot_label}_semi_minor"] = semi_minor
            ellipse_centers[plot_label] = ellipse_center
    
    # Add CRIRES data for TWA28 (only if parameter is available)
    if X_AXIS_PARAM in ['C/O', '[C/H]', '12C/13C', '12CO', 'H2O']:
        crires_data = load_crires_data(path, 'TWA28')
        if X_AXIS_PARAM in crires_data:
            all_x_data.extend(crires_data[X_AXIS_PARAM])  # Collect for axis scaling
            crires_correlation, crires_slope, crires_intercept, crires_angle, crires_semi_major, crires_semi_minor, crires_center = create_correlation_plot(
                ax, crires_data['log_g'], crires_data[X_AXIS_PARAM], 
                colors['TWA28']['crires'], 'TWA 28 (CRIRES)', cmaps['TWA28'][-1]
            )
            
            # Store CRIRES statistics
            correlations['CRIRES'] = crires_correlation
            correlations['CRIRES_slope'] = crires_slope
            correlations['CRIRES_intercept'] = crires_intercept
            correlations['CRIRES_angle'] = crires_angle
            correlations['CRIRES_semi_major'] = crires_semi_major
            correlations['CRIRES_semi_minor'] = crires_semi_minor
            ellipse_centers['CRIRES'] = crires_center
    
    # Apply proper axis scaling (but keep the current hardcoded limits for 12CO if desired)
    if all_x_data and X_AXIS_PARAM != '12CO':  # Skip auto-scaling for 12CO to keep current behavior
        x_limits = get_axis_limits(X_AXIS_PARAM, np.array(all_x_data))
        ax.set_xlim(x_limits)
        print(f"Set x-axis limits for {X_AXIS_PARAM}: {x_limits}")
    
    finalize_plot(fig, ax, correlations, colors, ellipse_centers, runs)
    
    # Save the figure with parameter-specific filename
    param_name = X_AXIS_PARAM.replace('/', '_').replace('[', '').replace(']', '')
    output_path = path_figures / f'logg_{param_name}{get_filename_suffix()}_correlation_ellipse.pdf'
    plt.savefig(output_path, bbox_inches='tight', dpi=300, 
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"\nPublication-ready figure saved to {output_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()

