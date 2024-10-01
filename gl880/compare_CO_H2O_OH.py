from retrieval_base.retrieval import Retrieval
import retrieval_base.figures as figs
from retrieval_base.config import Config
# import config_freechem as conf
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import corner
plt.style.use('/home/dario/phd/retrieval_base/HBDs/my_science.mplstyle')

base_path = '/home/dario/phd/retrieval_base/'
target = 'gl880'

if target not in os.getcwd():
    os.chdir(base_path + target)
    
outputs = pathlib.Path(base_path) / target / 'retrieval_outputs'
config_file = 'config_freechem.txt'

run_id = 17
run = f'sphinx{run_id}'


def main(target, run=None, fig=None, ax=None, ax_pt=None, **kwargs):
    
    
    if target not in os.getcwd():
        os.chdir(base_path + target)

    outputs = pathlib.Path(base_path) / target / 'retrieval_outputs'
    # find dirs in outputs
    print(f' outputs = {outputs}')
    dirs = [d for d in outputs.iterdir() if d.is_dir() and 'sphinx' in d.name and '_' not in d.name]
    runs = [int(d.name.split('sphinx')[-1]) for d in dirs]
    if run is None:
        run = 'sphinx'+str(max(runs))
    else:
        run = 'sphinx'+str(run)
        assert run in [d.name for d in dirs], f'Run {run} not found in {dirs}'
    print('Run:', run)
    
    
    conf = Config(path=base_path, target=target, run=run)(config_file)

    ret = Retrieval(
                conf=conf, 
                evaluation=False,
                )
    
    file_envelopes = ret.conf_output + 'envelopes.npy'
    file_labels = ret.conf_output + 'labels.npy'
    if os.path.exists(file_envelopes) and os.path.exists(file_labels):
        
        envelopes = np.load(file_envelopes)
        all_labels = np.load(file_labels)
        ret.Chem.VMRs_envelopes = dict(zip(all_labels, envelopes))
        
    else:

        bestfit_params, posterior = ret.PMN_analyze()

        ret.get_PT_mf_envelopes(posterior)
        ret.Chem.get_VMRs_posterior(save_to=ret.conf_output)
        
    file_pt_envelopes = ret.conf_output + 'PT_envelopes.npy'
    if os.path.exists(file_pt_envelopes):
        temperature_envelopes = np.load(file_pt_envelopes)
        
    else:
        
        bestfit_params, posterior = ret.PMN_analyze(return_dict=False)
        print(f' Shape of posterior = {posterior.shape}')
        
        # samples = np.array([dict(zip(ret.Param.param_keys, sample))
        temperature = []
        for sample in posterior:
            # sample_dict = dict(zip(ret.Param.param_keys, sample))
            # sample_dict.update(conf.constant_params)
            # print(f' sample_dict = {sample_dict}')
            ret.evaluate_model(sample)
            temperature.append(ret.PT(ret.Param.params))
        
        temperature_posterior = np.array(temperature)
        temperature_envelopes = np.quantile(temperature_posterior, [0.16, 0.5, 0.84], axis=0)
        print(f' Shape of temperature_envelopes = {temperature_envelopes.shape}')
        np.save(file_pt_envelopes, temperature_envelopes)
        print(f' Saved {file_pt_envelopes}')

    labels = ['12CO', 'H2O', 'OH']
    CO, H2O, OH = (ret.Chem.VMRs_envelopes[k] for k in labels)

    p = ret.PT.pressure
    alpha_envelope = kwargs.get('alpha_envelope', 0.5)
    color = kwargs.get('color', 'k')
    for i, (molecule, vmr) in enumerate(zip(labels, [CO, H2O, OH])):
        ax[i].plot(vmr[1], p, color=color)
        ax[i].fill_betweenx(p, vmr[0], vmr[2], alpha=alpha_envelope, color=color, lw=0)
        ax[i].set_title(molecule)
        ax[i].set(xscale='log', yscale='log', ylim=(np.max(p), np.min(p)))
        
    if ax_pt is not None:
        ax_pt.plot(temperature_envelopes[1], p, color=color, lw=2, alpha=0.8)
        ax_pt.fill_betweenx(p, temperature_envelopes[0], temperature_envelopes[2], alpha=alpha_envelope, color=color, lw=0)
        
    return fig, ax
        
spirou_sample = {
                'skip': [(3800, 4.72, 0.21), 'skip'], # to extend colorbar
                '880': [(3720, 4.72, 0.21), '17'],
                '15A': [(3603, 4.86, -0.30), None],
            # '411': (3563, 4.84, 0.12), # TODO: double check this target
            # '752A': [(3558, 4.76, 0.10),None],
            '725A': [(3441, 4.87, -0.23),None],
            '725B': [(3345, 4.96, -0.30),None],
            '15B': [(3218, 5.07, -0.30),None],
            '905': [(2930, 5.04, 0.23),None],
}   
targets = ['gl'+t for t in spirou_sample.keys()]
teffs = np.array([spirou_sample[t][0][0] for t in spirou_sample.keys()])
fig, ax_all = plt.subplots(1, 4, figsize=(12, 4), sharey=True, sharex=False,
                       gridspec_kw={'wspace': 0.10, 'hspace': 0.05, 'bottom': 0.13, 'top': 0.93, 'left': 0.08, 'right': 0.96})
ax_pt = ax_all[0]
ax = ax_all[1:]


cmap = 'plasma'
colors = getattr(plt.cm, cmap)(np.linspace(0, 1.0, len(targets)))[::-1]

sm = plt.cm.ScalarMappable(cmap=getattr(plt.cm, cmap), norm=plt.Normalize(vmin=teffs.min(), vmax=teffs.max()))
cbar = plt.colorbar(sm, ax=ax[-1], orientation='vertical', pad=0.03, label='Teff / K', aspect=20, location='right')


for t, target in enumerate(targets):
    # if t>2:
    #     break
    if target == 'glskip':
        continue
    fig, ax = main(target, run=spirou_sample[target[2:]][1], fig=fig, ax=ax, color=colors[t],
                   ax_pt=ax_pt)
        

[axi.set_xlim(1e-8, 5e-4) for axi in ax]

# add common xlabel for all subplots
fig.text(0.60, 0.02, 'Volume mixing ratio', ha='center', fontsize=14)
ax_pt.set_ylabel('Pressure / bar', fontsize=14)
ax_pt.set_xlim(1200, 7000)
ax_pt.set_xlabel('Temperature / K', fontsize=14)

ax_pt.axhspan(10, 0.1, color='k', alpha=0.1, lw=0)
# plt.show()
fig_name = base_path + 'paper/figures/CO_H2O_OH.pdf'
fig.savefig(fig_name)
print(f' Saved {fig_name}')
plt.close(fig)