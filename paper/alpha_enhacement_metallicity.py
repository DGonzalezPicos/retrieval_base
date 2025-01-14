from retrieval_base.retrieval import Retrieval
import retrieval_base.figures as figs
from retrieval_base.config import Config
from retrieval_base.auxiliary_functions import spirou_sample, read_spirou_sample_csv, find_run, load_romano_models
# import config_freechem as conf
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import matplotlib.patheffects as pe
import scienceplots

# reset to default
plt.style.use('default')
# plt.style.use(['latex-sans'])
plt.style.use(['sans'])
# enable latex
# plt.rcParams['text.usetex'] = True
plt.rcParams.update({
    "font.size": 8,
})

base_path = '/home/dario/phd/retrieval_base/'
nat_path = '/home/dario/phd/nat/figures/'

norm = plt.Normalize(3000.0, 3900.0)
cmap = plt.cm.coolwarm_r

df = read_spirou_sample_csv()
names = df['Star'].to_list()
teff =  dict(zip(names, [float(t.split('+-')[0]) for t in df['Teff (K)'].to_list()]))

table_id=3 # fixed for now...
c23 = np.loadtxt(f'{base_path}paper/data/c23_table{table_id}_alpha_fe.txt', dtype=object)
c23_names = ['Gl '+n[2:] for n in c23[:,0]]
y = dict(zip(c23_names, c23[:,1].astype(float)))
y_err = dict(zip(c23_names, c23[:,2].astype(float)))

c23_mh = np.loadtxt(f'{base_path}paper/data/c23_table{table_id}_mh.txt', dtype=object)
c23_mh_names = ['Gl '+n[2:] for n in c23_mh[:,0]]
x_mh = dict(zip(c23_mh_names, c23_mh[:,1].astype(float)))
x_err_mh = dict(zip(c23_mh_names, c23_mh[:,2].astype(float)))

fig, ax = plt.subplots(1,1, figsize=(5,4))

for target in names:
    ax.errorbar(x_mh[target], y[target], xerr=x_err_mh[target], yerr=y_err[target], fmt='o', color=cmap(norm(teff[target])))
ax.set_xlabel('metallicity [M/H]')
ax.set_ylabel(r'[$\alpha$/Fe]')

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Only needed for color bar

# define cbar_ax for colorbar
top = 0.95
b = 0.00
cbar_ax = fig.add_axes([0.94, b, 0.030, top-b])
cbar = plt.colorbar(sm, cax=cbar_ax, orientation='vertical', aspect=1)
cbar.set_label(r'T$_{\mathrm{eff}}$ (K)')
        
fig.savefig(nat_path + f'alpha_enhacement_metallicity.pdf', bbox_inches='tight')
print('Saved figure to', nat_path + f'alpha_enhacement_metallicity.pdf')
