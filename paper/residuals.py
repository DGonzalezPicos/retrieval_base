from retrieval_base.retrieval import Retrieval
import retrieval_base.figures as figs
from retrieval_base.config import Config
# import config_freechem as conf
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
from retrieval_base.auxiliary_functions import spirou_sample

base_path = '/home/dario/phd/retrieval_base/'

# add another column with distance in pc

def mas_to_pc(mas):
    # distance in pc
    return 1/mas * 1e3

print(mas_to_pc(201.3252))

targets = ['gl'+t for t in spirou_sample.keys()]
temperature_dict = {t: spirou_sample[t[2:]][0][0] for t in targets}
# norm = plt.Normalize(min(temperature_dict.values()), max(temperature_dict.values()))
norm = plt.Normalize(min(temperature_dict.values()), 4000.0)
cmap = plt.cm.plasma

def main(target, ax=None, fig=None, offset=0, order=0, run=None, lw=1.0, color=None, text_x=0.05):
    
    if target not in os.getcwd():
        os.chdir(base_path + target)

    outputs = pathlib.Path(base_path) / target / 'retrieval_outputs'
    # find dirs in outputs
    print(f' outputs = {outputs}')
    dirs = [d for d in outputs.iterdir() if d.is_dir() and 'sphinx' in d.name and '_' not in d.name]
    runs = [int(d.name.split('sphinx')[-1]) for d in dirs]
    # run = 'sphinx'+str(max(runs))
    if run is None:
        run = 'sphinx'+str(max(runs))
    else:
        run = 'sphinx'+str(run)
        assert run in [d.name for d in dirs], f'Run {run} not found in {dirs}'
    print('Run with largest number:', run)

    config_file = 'config_freechem.txt'
    conf = Config(path=base_path, target=target, run=run)(config_file)
    
    output_dir = conf.prefix + 'output'
    print(f'output_dir = {output_dir}')
    
    wave, residuals, err = np.load(output_dir + f'/test_residuals_{order}.npy')
    print(f'wave.shape = {wave.shape}')
    
    ax.plot(wave, residuals + offset, color=color, lw=lw)
    ax.fill_between(wave, offset - err, offset + err, alpha=0.2, color=color, lw=0)
    ax.axhline(offset, color='k', lw=0.5)
    
    mask = np.isfinite(residuals)
    
    text_x = np.nanmin(wave) if text_x is None else text_x
    text_pos = (text_x[0], offset+0.01)
    # add white box around text
    ax.text(*text_pos, target.replace('gl','Gl'), color='k', fontsize=12, weight='bold', transform=ax.transData)
    
    # add another text with MAD
    mad = np.nanmedian(np.abs(residuals[mask]))
    print(f' Target, MAD = {target}, {mad*100} %')
    ax.text(text_x[1], offset+0.01, f'MAD = {mad*100:.1f} %', color='k', fontsize=12, transform=ax.transData)
    
    
        
fig, ax = plt.subplots(1,1, figsize=(14,8), tight_layout=True)

text_x = [(2287.0, 2364.),
          (2358.0, 2438.),
          (2435.0, 2510.0),
]
          
order = 2
for i, target in enumerate(targets):
    temperature = temperature_dict[target]
    color = cmap(norm(temperature))
    main(target, ax=ax, fig=fig, offset=0.42*(len(targets)-i), order=order, run=spirou_sample[target[2:]][1], lw=1.0, color=color, text_x=text_x[order])
    

ax.set(xlabel='Wavelength [nm]', ylabel='Residuals', xlim=(text_x[order][0]-1, text_x[order][1]+8))
ax.grid()

fig_name = base_path + f'paper/latex/figures/residuals_order{order}.pdf'
fig.savefig(fig_name)
plt.show()

    
    