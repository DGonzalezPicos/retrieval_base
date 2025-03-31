import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
from retrieval_base.retrieval import Retrieval
import retrieval_base.auxiliary_functions as af
from retrieval_base.config import Config
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import imageio.v3 as iio
from PIL import Image
from tqdm import tqdm
path = af.get_path(return_pathlib=True)
path_figures = pathlib.Path('/home/dario/phd/twa2x_paper/figures')

config_file = 'config_jwst.txt'
w_set='NIRSpec'


def load_data(target, run):
    cwd = os.getcwd()
    if target not in cwd:
        os.chdir(f'{path}/{target}')
        print(f'Changed directory to {target}')

    conf = Config(path=path, target=target, run=run)(config_file)        
        
    m_spec = af.pickle_load(f'{conf.prefix}data/bestfit_m_spec_NIRSpec.pkl')
    m_spec.flux = m_spec.flux.squeeze()
    d_spec = af.pickle_load(f'{conf.prefix}data/d_spec_NIRSpec.pkl')
    
    Cov = af.pickle_load(f'{conf.prefix}data/bestfit_Cov_NIRSpec.pkl')
    LogLike = af.pickle_load(f'{conf.prefix}data/bestfit_LogLike_NIRSpec.pkl')
    err = np.nan * np.ones_like(d_spec.flux)
    
    for i in range(d_spec.n_orders):
        for j in range(d_spec.n_dets):
            # mask_i = d_spec.mask_isfinite[i,]
            mask_ij = d_spec.mask_isfinite[i,j]

            if Cov is not None:
                err_ij = Cov[i,j].get_err(mask=mask_ij)
            else:
                err_ij = d_spec.err[i,j]
                    
            beta_ij = LogLike.beta[i,j]
            err_ij *= beta_ij # optimal uncertainty scaling
            err[i,j,:] = err_ij
        
    flux_factor = conf.config_data['NIRSpec'].get('flux_unit_factor', 1.0)
    
    d_spec.err = err
    d_spec.squeeze()
    m_spec.flux.squeeze()

    m_spec.flux /= flux_factor
    d_spec.flux /= flux_factor
    d_spec.err /= flux_factor
    return d_spec, m_spec

runs = dict(
            # TWA28='no_psf_corr_lbl10_G2G3_newGP_0',
            # TWA27A='freeslab_lbl10_G1G2G3_0',
            # TWA28='lbl11_G1G2G3_fastchem_GP_1',
            TWA28='freeslab_lbl10_G1G2G3_0',
            TWA27A='freeslab_lbl10_G1G2G3_0',
            )

d_specs, m_specs = {}, {}
for target in runs.keys():
    
    try:
        d_specs[target], m_specs[target] = load_data(target, runs[target])
    except Exception as e:
        print(f'Error loading data for {target}: {e}')
        continue
    
runs = {k:v for k,v in runs.items() if k in d_specs.keys()}

def fig_ax():
    fig, ax = plt.subplots(2,1, figsize=(12,7), sharex=True, gridspec_kw={'height_ratios':[3,1]},
                           tight_layout=True)
    return fig, ax

def plot_chunk(d_spec, m_spec, ax=None, idx=0, relative_residuals=False, colors=None, offset=0.0, ls='-', lw=1.4, new_fig=False,
               color_residuals=None, ylim_p=None, inset=None, inset_wave_range=None):
    
    new_ax = (ax is None)
    if new_ax:
        fig, ax = plt.subplots(2,1, figsize=(10,5), sharex=True)
    else:
        assert len(ax) == 2, f'ax must be a list of 2 elements, not {len(ax)}'
        
    wave = d_spec.wave[idx]
    flux = d_spec.flux[idx] + offset
    err = d_spec.err[idx]
    nans = np.isnan(flux)
    large_err = err > np.nanquantile(err, 0.99)
    nans = nans | large_err
    flux = np.where(nans, np.nan, flux)
    err = np.where(nans, np.nan, err)
    
    m_flux = m_spec.flux[idx] + offset
    m_flux_nans = np.where(~nans, np.nan, m_flux)
    m_flux[nans] = np.nan
    
    ax[0].plot(wave, flux, color=colors['data'], lw=lw, alpha=0.8, ls=ls)
    ax[0].plot(wave, flux, color=colors['data'], ls='none', marker='o', ms=2, alpha=0.8)
    ax[0].fill_between(wave, flux - err, flux + err, color=colors['data'], alpha=0.2, lw=0.0)
    ax[0].plot(wave, m_flux, color=colors['model'], lw=lw, alpha=0.8, ls=ls, label=target)
    # ax[0].plot(wave, m_flux_nans, color='red', lw=lw, alpha=0.8, ls=ls)
    if inset is not None:
        inset_mask = (wave >= inset_wave_range[0]) & (wave <= inset_wave_range[1])
        inset.plot(wave[inset_mask], flux[inset_mask], color=colors['data'], lw=lw, alpha=0.8, ls=ls)
        inset.plot(wave[inset_mask], flux[inset_mask], color=colors['data'], ls='none', marker='o', ms=2, alpha=0.8)
        inset.fill_between(wave[inset_mask], flux[inset_mask] - err[inset_mask], flux[inset_mask] + err[inset_mask], color=colors['data'], alpha=0.2, lw=0.0)
        inset.plot(wave[inset_mask], m_flux[inset_mask], color=colors['model'], lw=lw, alpha=0.8, ls=ls, label=target)
        # Add highlighting for the inset region
        ax[0].axvspan(inset_wave_range[0], inset_wave_range[1], color='k', alpha=0.01, zorder=0, lw=0.0)
        ax[1].axvspan(inset_wave_range[0], inset_wave_range[1], color='k', alpha=0.01, zorder=0, lw=0.0)
    res = flux - m_flux
    if relative_residuals:
        res_norm = res / flux
        err_norm = err / flux
    else:
        res_norm = res
        err_norm = err
        
    MAD = np.nanmedian(np.abs(res_norm))
    
    color_residuals = colors['model'] if color_residuals is None else color_residuals
    ax[1].plot(wave, res_norm, color=color_residuals, lw=lw, alpha=0.4)
    ax[1].scatter(wave, res_norm, color=color_residuals, s=1.0, alpha=0.7)
    ax[1].fill_between(wave, -err_norm, err_norm, color=colors['model'], alpha=0.3, lw=0.0)
    ax[1].axhline(0.0,color=colors['data'], lw=0.7)
    # if new_ax:
    
    ax[1].set_xlabel('Wavelength / nm')
    ax[0].set_ylabel(r'$F_{\lambda}$ / erg s$^{-1}$ cm$^{-2}$ nm$^{-1}$')
    if ylim_p is not None:
        p = np.nanpercentile(flux, ylim_p)
        # ax[0].set_ylim(p[0], p[1])
        ax[0].set_ylim(0.0, p[1])
    
    res_label = r'$\Delta F_{\lambda} / F_{\lambda}$' if relative_residuals else r'$\Delta F_{\lambda} / erg s$^{-1}$ cm$^{-2}$ nm$^{-1}$'
    ax[1].set_ylabel(res_label)
    
    if new_fig:
        
    
        ylim = - MAD*6.0, MAD*6.0
        ax[1].set_ylim(ylim)
        ax[1].text(0.02, 0.95, f'MAD = {MAD*100.0:.1f} %', transform=ax[1].transAxes,
                ha='left', va='top', fontsize=12)
        xlim = np.nanmin(wave[~nans])-2.0, np.nanmax(wave[~nans])+2.0
        ax[0].set_xlim(xlim)
        ax[1].set_xlim(xlim)
    
    # ax[1].axhline(0.0,color=colors['model'], lw=0.7)
        # ax.set_title(f'Chunk {idx}')
        # plt.show()
    return ax, wave, flux, err

colors = dict(TWA28={'data':'k', 'model':'dodgerblue'},
              TWA27A={'data':'k', 'model':'green'})

def plot_idx(idx, fig=None, ax=None, ylim_p=None, ylim=None, targets=None, inset=None, inset_wave_range=None):  
    new_fig = False
    if fig is None or ax is None:
        fig, ax = fig_ax()
        new_fig = True
        ax[0].set_ylim(ylim[0], ylim[1])
        
    if targets is None:
        targets = runs.keys()
    for target in targets:
        d_spec, m_spec = d_specs[target], m_specs[target]
    # assert len(ax) == 2, f'ax must be a list of 2 elements, not {len(ax)}'
        _, wave, flux, err = plot_chunk(d_spec, m_spec, ax=ax, relative_residuals=True, 
                                        idx=idx,
                                        colors=colors[target],
                                        new_fig=new_fig,
                                        color_residuals=colors[target]['model'],
                                        ylim_p=ylim_p,
                                        inset=inset,
                                        inset_wave_range=inset_wave_range)
        
    return wave, flux, err
         
# Create output directory for frames if it doesn't exist
transparent = False

gslides_background_color = (246,178,107,100)
gslides_background_color_norm = np.array(gslides_background_color) / 255.0
gslides_background_color_norm[-1] = 1.0 # make the background color opaque
frames_dir = path_figures / f'frames{"_transparent" if transparent else ""}'
frames_dir.mkdir(exist_ok=True)

frames_dir_high_res = path_figures / f'frames_high_res{"_transparent" if transparent else ""}'
frames_dir_high_res.mkdir(exist_ok=True)

# Get the full wavelength range
xlim = np.nanmin(d_specs['TWA27A'].wave), np.nanmax(d_specs['TWA27A'].wave)
ymin = np.nanmin([d_specs[t].flux for t in runs.keys()])
ymax = np.nanmax([d_specs[t].flux for t in runs.keys()])

# Create wavelength bins with fixed step size
step_size = 8  # nm
window_size = 180  # nm
dpi = 150
wavelength_bins = []
current_start = xlim[0]
while current_start + window_size <= xlim[1]:
    wavelength_bins.append((current_start, current_start + window_size))
    current_start += step_size

# Generate frames
frame_files = []
# save some bins in high resolution (300 dpi)
high_res_bins = [(980, 1140),
                 (2200, 2400)]

# use tqdm to show progress
for i, (bin_start, bin_end) in tqdm(enumerate(wavelength_bins), total=len(wavelength_bins)):
    # Create a completely new figure for each frame
    plt.close('all')  # Close all existing figures
    fig, ax = fig_ax()
    fig.patch.set_facecolor(gslides_background_color_norm)
    ax[0].patch.set_facecolor(gslides_background_color_norm)
    ax[1].patch.set_facecolor(gslides_background_color_norm)
    # Create inset
    inset = inset_axes(ax[0], width="70%", height="60%", loc=1, borderpad=1.5)
    inset.patch.set_facecolor(gslides_background_color_norm)
    inset_wave_range = (bin_start, bin_end)
    inset.set_xlim(inset_wave_range)
    
    # Set main plot limits
    ax[0].set_xlim(xlim)
    ax[0].set_ylim(ymin, ymax)
    ax[1].set_xlim(xlim)
    ax[1].set_ylim(-0.1, 0.1)
    
    # Plot data for all indices
    for idx in range(d_specs['TWA27A'].flux.shape[0]):
        plot_idx(idx, fig=fig, ax=ax, inset=inset, inset_wave_range=inset_wave_range)
    
    # Save frame with transparent background
    frame_file = frames_dir / f'frame_{i:03d}.png'
    plt.savefig(frame_file, dpi=dpi, bbox_inches='tight', transparent=transparent, facecolor='none', edgecolor='none')
    
    # check if wavelength bin is in high resolution bins
    if (i%50) == 0:
        # print(f'Saving high resolution frame {i} with wavelength range {bin_start:.0f} - {bin_end:.0f} nm')
        frame_file_high_res = frames_dir_high_res / f'frame_{i:03d}.png'
        plt.savefig(frame_file_high_res, dpi=dpi*2, bbox_inches='tight', transparent=transparent, facecolor='none', edgecolor='none')
    plt.close(fig)  # Close the current figure
    frame_files.append(frame_file)

def create_gif(frame_folder, output_gif, background_color=(246,178,107,100), duration=100, loop=0):
    """
    Create a GIF from a series of PNG images in a folder.
    
    Parameters:
        frame_folder (str): Path to the folder containing PNG frames.
        output_gif (str): Path to save the output GIF.
        duration (int): Duration of each frame in milliseconds.
        loop (int): Number of times the GIF loops (0 = infinite).
    """
    frames = sorted(
        [os.path.join(frame_folder, f) for f in os.listdir(frame_folder) if f.endswith(".png")]
    )
    
    if not frames:
        raise ValueError("No PNG files found in the specified folder.")
    
    # Create a white background image
    first_image = Image.open(frames[0])
    background = Image.new('RGBA', first_image.size, background_color)
    
    # Process each frame
    processed_images = []
    for frame in frames:
        # Open the transparent frame
        frame_image = Image.open(frame)
        # Composite the frame onto the white background
        composite = Image.alpha_composite(background, frame_image)
        processed_images.append(composite)
    
    # Save the GIF
    processed_images[0].save(
        output_gif,
        save_all=True,
        append_images=processed_images[1:],
        duration=duration,
        loop=loop
    )
    print(f"GIF saved at {output_gif}")

# Create GIF
gif_path = path_figures / f'spectrum_inset_{step_size}nm_{window_size}nm_{"transparent" if transparent else "opaque"}_dpi{dpi}.gif'
create_gif(frames_dir, gif_path, background_color=gslides_background_color, duration=400, loop=0)