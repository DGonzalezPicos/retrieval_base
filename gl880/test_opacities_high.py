import argparse
from retrieval_base.retrieval import Retrieval
import retrieval_base.figures as figs
from retrieval_base.config import Config

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

import os

base_path = '/home/dario/phd/retrieval_base/'

target = 'gl880'
if target not in os.getcwd():
    os.chdir(base_path + target)

run = 'sphinx_4'
config_file = 'config_freechem.txt'
w_set = 'spirou'
def get_model(run):
    conf = Config(path=base_path, target=target, run=run)(config_file)


    ret = Retrieval(
                conf=conf, 
                evaluation=False,
                )
    return ret.d_spec[w_set], ret.bestfit_model

runs = ['sphinx_4', 'sphinx_8']
d_spec, (w4, m4) = get_model(runs[0])
_, (w8, m8) = get_model(runs[1])
n_orders = m4.shape[0]


# use PDF pages to plot all orders, each order on a different row, each row with two plots: spec and residuals
pdf_name = base_path + target + f'/retrieval_outputs/{runs[-1]}/test_plots/' + f'{runs[0]}_{runs[1]}_comparison.pdf'

with PdfPages(pdf_name) as pdf:
    for i in range(n_orders):
        fig, axs = plt.subplots(2,1, figsize=(14,4), gridspec_kw={'height_ratios': [3, 1]})
        d_wave, d_flux = np.squeeze(d_spec.wave[i]), np.squeeze(d_spec.flux[i])
        # plot spectra
        axs[0].plot(w4[i], m4[i], label=runs[0], alpha=0.75)
        axs[0].plot(w8[i], m8[i], label=runs[1], alpha=0.75)
        axs[0].plot(d_wave, d_flux, label='Data', alpha=0.5)
        axs[0].legend()

        # plot residuals
        axs[1].plot(w4[i], d_flux-m4[i], label=runs[0], alpha=0.65)
        axs[1].plot(w8[i], d_flux-m8[i], label=runs[1], alpha=0.65)
        
        print(f' MAD {i} models: {np.nanmean(np.abs(m4[i]-m8[i])):.3f}')
        pdf.savefig(fig)
        plt.close(fig)
print(f'PDF saved to {pdf_name}')
        


