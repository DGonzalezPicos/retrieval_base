import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import numpy as np

import os
import time
import json
import copy
import corner

import retrieval_base.auxiliary_functions as af
import retrieval_base.figures as figs

class CallBack:

    plot_histograms = False
    plot_cov_matrix = False
    plot_residual_ACF = False
    plot_ccf = False
    plot_summary = True
    
    def __init__(self, 
                 d_spec, 
                 evaluation=False, 
                 n_samples_to_use=2000, 
                 prefix=None, 
                 posterior_color='C0', 
                 bestfit_color='C1', 
                 PT_color='orangered',
                 species_to_plot_VMR=['12CO', 'H2O', '13CO', 'CH4', 'NH3', 'C18O'], 
                 species_to_plot_CCF=['12CO', 'H2O', '13CO', 'CH4'], 
                 ):
        
        self.elapsed_times = []
        self.active = False
        self.return_PT_mf = False

        self.evaluation = evaluation
        self.cb_count = 0
        if self.evaluation:
            self.cb_count = -2

        self.n_samples_to_use = n_samples_to_use

        self.d_spec = d_spec
        self.prefix = prefix

        self.posterior_color = posterior_color
        self.bestfit_color = bestfit_color
        self.PT_color = PT_color

        self.envelope_cmap = mpl.colors.LinearSegmentedColormap.from_list(
            name='envelope_cmap', colors=['w', self.posterior_color], 
            )
        self.PT_envelope_cmap = mpl.colors.LinearSegmentedColormap.from_list(
            name='PT_envelope_cmap', colors=['w', self.PT_color], 
            )
        self.envelope_colors = self.envelope_cmap([0.0,0.2,0.4,0.6,0.8])
        self.envelope_colors[0,-1] = 0.0
        
        self.PT_envelope_colors = self.PT_envelope_cmap([0.0,0.2,0.4,0.6,0.8])
        self.PT_envelope_colors[0,-1] = 0.0

        self.species_to_plot_VMR = species_to_plot_VMR
        self.species_to_plot_CCF = species_to_plot_CCF

    def __call__(self, 
                 Param, 
                 LogLike, 
                 Cov, 
                 PT, 
                 Chem, 
                 m_spec, 
                 pRT_atm, 
                 posterior, 
                 species_to_plot_VMR=[],
                 m_spec_species=None, 
                 pRT_atm_species=None, 
                 ):

        time_A = time.time()
        self.cb_count += 1

        # Make attributes for convenience
        self.Param   = Param
        self.LogLike = LogLike
        self.Cov     = Cov
        self.PT      = PT
        self.Chem    = Chem
        
        assert hasattr(self.Chem, 'mass_fractions_posterior'), 'Chemistry object does not have mass_fractions_posterior'
        if not hasattr(self.Chem, 'VMRs_posterior') and self.evaluation:
            self.Chem.get_VMRs_posterior()
        self.m_spec  = m_spec
        self.pRT_atm = pRT_atm

        self.m_spec_species  = m_spec_species
        self.pRT_atm_species = pRT_atm_species

        if not self.evaluation:
            # Use only the last n samples to plot the posterior
            n_samples = min([len(posterior), self.n_samples_to_use])
            self.posterior = posterior[-n_samples:]
        else:
            self.posterior = posterior

        # Display the mean elapsed time per lnL evaluation
        print('\n\nElapsed time per evaluation: {:.2f} seconds'.format(np.mean(self.elapsed_times)))
        self.elapsed_times.clear()

        # Create the labels
        self.param_labels = np.array(list(self.Param.param_mathtext.values()))

        chi_squared_tot, n_dof = 0, 0
        for w_set in self.LogLike.keys():
            print(f'\n--- {w_set} -------------------------')
            print('Reduced chi-squared (w/o uncertainty-model) = {:.2f}\n(chi-squared={:.2f}, n_dof={:.0f})'.format(
                self.LogLike[w_set].chi_squared_red, self.LogLike[w_set].chi_squared, self.LogLike[w_set].n_dof
                ))
            
            chi_squared_tot += self.LogLike[w_set].chi_squared
            #n_dof += self.LogLike[w_set].n_dof
            n_dof += self.d_spec[w_set].mask_isfinite.sum()

        n_dof -= self.LogLike[w_set].n_params

        print(f'\n--- Total -------------------------')
        print('Reduced chi-squared (w/o uncertainty-model) = {:.2f}\n(chi-squared={:.2f}, n_dof={:.0f})'.format(
            chi_squared_tot/n_dof, chi_squared_tot, n_dof
            ))

        # Read the best-fitting free parameters
        self.bestfit_params = []
        print('\nBest-fitting free parameters:')
        for key_i in self.Param.param_keys:
            if isinstance(self.Param.params[key_i], np.ndarray):
                print('{} = {:.2f}'.format(key_i, self.Param.params[key_i].flatten()[0]))
                self.bestfit_params.append(self.Param.params[key_i].flatten()[0])
            else:
                print('{} = {:.2f}'.format(key_i, self.Param.params[key_i]))
                self.bestfit_params.append(self.Param.params[key_i])

        for w_set in self.LogLike.keys():
            print(f'\n--- {w_set} -------------------------')
            # if self.LogLike[w_set].scale_flux:
            if hasattr(self.LogLike[w_set], 'phi'):
                print('\nOptimal flux-scaling parameters:')
                # print(self.LogLike[w_set].phi.round(2))
                print(f' phi = {self.LogLike[w_set].phi}')
            if self.LogLike[w_set].scale_err:
                print('\nOptimal uncertainty-scaling parameters:')
                print(self.LogLike[w_set].s.round(2))
                
            if hasattr(self.m_spec[w_set], 'int_contr_em'):
                # print(f' int_contr_em = {self.m_spec[w_set].int_contr_em}')
                self.PT.int_contr_em[w_set] = np.copy(self.m_spec[w_set].int_contr_em)
        
        self.bestfit_params = np.array(self.bestfit_params)
        
        # Save the bestfit parameters in a .json file
        # and the ModelSpectrum instance as .pkl
        if self.evaluation:
            self.save_bestfit()
            
        # Save a separate figure of the PT profile
        fig, ax = plt.subplots(1, 2, 
                                   figsize=(10,6), 
                                   tight_layout=True,
                                   sharey=True,)
        
        figs.fig_PT(
            PT=self.PT, 
            ax=ax[0], 
            ax_grad=ax[1] if hasattr(self.PT, 'dlnT_dlnP_array') else None,
            fig=fig,
            bestfit_color='C0',
            envelopes_color='C0',
            int_contr_em_color='red',
            text_color='gray',
            # weigh_alpha=True,
            show_photosphere=True,
            show_knots=True,
            show_text=True,
            fig_name=self.prefix+f'plots/PT_grad_profile.pdf',
            xlim=(1000, 8000), # fix view
        )

            
        # for i, w_set in enumerate(list(self.d_spec.keys())):
            # Plot the best-fitting spectrum
            # figs.fig_bestfit_model(
            #     d_spec=self.d_spec[w_set], 
            #     m_spec=self.m_spec[w_set], 
            #     LogLike=self.LogLike[w_set], 
            #     Cov=self.Cov[w_set], 
            #     bestfit_color=self.bestfit_color, 
            #     # ax_spec=ax_spec[i], 
            #     # ax_res=ax_res[i], 
            #     prefix=self.prefix, 
            #     xlabel=['Wavelength (nm)', None][i]
            #     )

        if self.evaluation:
            fig_chem, ax_chem = plt.subplots(1, 1, figsize=(6,6))
            fig_name_chem = self.prefix+f'plots/bestfit_VMRs.pdf'
            # Chem.get_VMRs_envelopes() # deprecated, VMRs_envelopes already computed with posterior
            figs.fig_VMR(Chem,
                    # ax=ax_chem,
                    # fig=fig_chem,
                    ax=ax[1],
                    fig=fig,
                    species_to_plot=species_to_plot_VMR,
                    pressure=PT.pressure,
                    showlegend=True,
                    ls='-',
                    fig_name=fig_name_chem,
                    )
            
            for w_set in self.d_spec.keys():

                # Plot the CCFs + spectra of species' contributions
                if self.plot_ccf:
                    figs.fig_species_contribution(
                        d_spec=self.d_spec[w_set], 
                        m_spec=self.m_spec[w_set], 
                        m_spec_species=self.m_spec_species[w_set], 
                        pRT_atm=self.pRT_atm[w_set], 
                        pRT_atm_species=self.pRT_atm_species[w_set], 
                        Chem=self.Chem, 
                        LogLike=self.LogLike[w_set], 
                        Cov=self.Cov[w_set], 
                        species_to_plot=self.species_to_plot_CCF, 
                        rv_CCF=np.arange(-1000,1000+1e-6,1.), 
                        prefix=self.prefix, 
                        w_set=w_set, 
                        )
            
                # Plot the auto-correlation of the residuals
                if self.plot_residual_ACF:
                    figs.fig_residual_ACF(
                        d_spec=self.d_spec[w_set], 
                        m_spec=self.m_spec[w_set], 
                        LogLike=self.LogLike[w_set], 
                        Cov=self.Cov[w_set], 
                        bestfit_color=self.bestfit_color, 
                        prefix=self.prefix, 
                        w_set=w_set, 
                        )
                if self.plot_cov_matrix:
                    # Plot the covariance matrices
                    all_cov = figs.fig_cov(
                        LogLike=self.LogLike[w_set], 
                        Cov=self.Cov[w_set], 
                        d_spec=self.d_spec[w_set], 
                        cmap=self.envelope_cmap, 
                        prefix=self.prefix, 
                        w_set=w_set, 
                        )
                    
                if self.m_spec[w_set].N_veiling > 0:
                    # Plot the veiling spectrum
                    figs.fig_veiling_factor(
                        d_spec=self.d_spec[w_set],
                        LogLike=self.LogLike[w_set], 
                        color=self.bestfit_color,
                        fig_name=self.prefix+f'plots/veiling_factors.pdf',
                        )
                if self.LogLike[w_set].N_knots > 1:
                    
                    figs.fig_spline_model(
                        d_spec=self.d_spec[w_set], 
                        m_spec=self.m_spec[w_set], 
                        LogLike=self.LogLike[w_set], 
                        Cov=self.Cov[w_set], 
                        bestfit_color=self.bestfit_color, 
                        # ax_spec=ax_spec[i], 
                        # ax_res=ax_res[i], 
                        prefix=self.prefix, 
                        xlabel='Wavelength / nm',
                        )

            # Plot the abundances in a corner-plot
            # self.fig_abundances_corner()
            figs.fig_chemistry(Chem=self.Chem,
                                    fig=None,
                                    species_to_plot=None,
                                    color=self.bestfit_color,
                                    smooth=None,
                                    fontsize=14,
                                    fig_name=self.prefix+f'plots/chemistry.pdf'
            )
                            
        # Make a summary figure
        if self.plot_summary:
            self.fig_summary()
            
            
        # Remove attributes from memory
        del self.Param, self.LogLike, self.PT, self.Chem, self.m_spec, self.pRT_atm, self.posterior

        time_B = time.time()
        print('\nPlotting took {:.0f} seconds\n'.format(time_B-time_A))

    def save_bestfit(self):
        
        # Save the best-fitting parameters
        params_to_save = {}
        for key_i, val_i in self.Param.params.items():
            if isinstance(val_i, np.ndarray):
                val_i = val_i.tolist()
            params_to_save[key_i] = val_i

        dict_to_save = {
            'params': params_to_save, 
            #'f': self.LogLike.f.tolist(), 
            #'beta': self.LogLike.beta.tolist(), 
            'temperature': self.PT.temperature.tolist(), 
            'pressure': self.PT.pressure.tolist(), 
        }
        for w_set in self.LogLike.keys():
            dict_to_save[f'f_{w_set}'] = self.LogLike[w_set].phi.tolist()
            dict_to_save[f'beta_{w_set}'] = self.LogLike[w_set].s.tolist()

        with open(self.prefix+'data/bestfit.json', 'w') as fp:
            json.dump(dict_to_save, fp, indent=4)

        # Save some of the objects
        af.pickle_save(self.prefix+'data/bestfit_PT.pkl', self.PT)

        Chem_to_save = copy.copy(self.Chem)
        if hasattr(Chem_to_save, 'fastchem'):
            del Chem_to_save.fastchem
        if hasattr(Chem_to_save, 'output'):
            del Chem_to_save.output
        if hasattr(Chem_to_save, 'input'):
            del Chem_to_save.input
        af.pickle_save(self.prefix+'data/bestfit_Chem.pkl', Chem_to_save)

        for w_set in self.LogLike.keys():
            af.pickle_save(self.prefix+f'data/bestfit_m_spec_{w_set}.pkl', self.m_spec[w_set])

            # Save the best-fitting log-likelihood
            LogLike_to_save = copy.deepcopy(self.LogLike[w_set])
            del LogLike_to_save.d_spec
            af.pickle_save(self.prefix+f'data/bestfit_LogLike_{w_set}.pkl', LogLike_to_save)

            # Save the best-fitting covariance matrix
            af.pickle_save(self.prefix+f'data/bestfit_Cov_{w_set}.pkl', self.Cov[w_set])

            # Save the contribution functions and cloud opacities
            np.save(
                self.prefix+f'data/bestfit_int_contr_em_{w_set}.npy', 
                self.pRT_atm[w_set].int_contr_em
                )
            np.save(
                self.prefix+f'data/bestfit_int_contr_em_per_order_{w_set}.npy', 
                self.pRT_atm[w_set].int_contr_em_per_order
                )
            np.save(
                self.prefix+f'data/bestfit_int_opa_cloud_{w_set}.npy', 
                self.pRT_atm[w_set].int_opa_cloud
                )

    def fig_abundances_corner(self):

        included_params = []

        # Plot the abundances
        if self.Param.chem_mode == 'free':

            for species_i, (line_species_i, _, mass_i, COH_i) in self.Chem.species_info.items():
                # print(f'line_species_i = {line_species_i}', f'mass_i = {mass_i}', f'COH_i = {COH_i}')
                # Add to the parameters to be plotted
                if (line_species_i in self.Chem.line_species) and \
                    (f'log_{species_i}' in self.Param.param_keys):
                    included_params.append(f'log_{species_i}')
                for j in range(3):
                    if (line_species_i in self.Chem.line_species) and \
                        (f'log_{species_i}_{j}' in self.Param.param_keys):
                        included_params.append(f'log_{species_i}_{j}')

            if 'log_C_ratio' in self.Param.param_keys:
                included_params.append('log_C_ratio')

            # Add C/O and Fe/H to the parameters to be plotted
            if self.evaluation:
                                
                # Add to the posterior
                self.posterior = np.concatenate(
                    (self.posterior, 
                     self.Chem.CO_posterior[:,None], 
                     self.Chem.FeH_posterior[:,None],
                     iso_posteriors[:,None]), axis=1
                    )
                # Add to the parameter keys
                self.Param.param_keys = np.concatenate(
                    (self.Param.param_keys, ['C/O', 'C/H'])
                )
                self.param_labels = np.concatenate(
                    (self.param_labels, ['C/O', '[C/H]'])
                    )
                
                self.bestfit_params = np.concatenate(
                    (self.bestfit_params, [self.Chem.CO, self.Chem.CH])
                )
                included_params.extend(['C/O', 'C/H'])

        elif self.Param.chem_mode in ['eqchem', 'fastchem', 'SONORAchem']:
            
            for key in self.Param.param_keys:
                if key.startswith('log_C_ratio'):
                    included_params.append(key)
                
                elif key.startswith('log_C13_12_ratio'):
                    included_params.append(key)
                elif key.startswith('log_O18_16_ratio'):
                    included_params.append(key)
                elif key.startswith('log_O17_16_ratio'):
                    included_params.append(key)

                elif key.startswith('log_P_quench'):
                    included_params.append(key)

            included_params.extend(['C/O', 'Fe/H'])

        figsize = (
            4/3*len(included_params), 4/3*len(included_params)
            )
        fig, ax = self.fig_corner(
            included_params=included_params, 
            fig=plt.figure(figsize=figsize), 
            smooth=False, ann_fs=10
            )

        # Plot the VMR per species
        ax_VMR = fig.add_axes([0.6,0.6,0.32,0.32])
        ax_VMR = figs.fig_VMR(
            ax_VMR=ax_VMR, 
            Chem=self.Chem, 
            species_to_plot=self.species_to_plot_VMR, 
            pressure=self.PT.pressure, 
            )

        plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08, wspace=0, hspace=0)
        fig.savefig(self.prefix+'plots/abundances.pdf')
        plt.close(fig)

        if self.evaluation and self.Param.chem_mode == 'free':
            # Remove the changes
            self.posterior        = self.posterior[:,:-2]
            self.Param.param_keys = self.Param.param_keys[:-2]
            self.param_labels     = self.param_labels[:-2]
            self.bestfit_params   = self.bestfit_params[:-2]
        
    def fig_corner(self, included_params=None, fig=None, smooth=False, ann_fs=9):
        
        # Compute the 0.16, 0.5, and 0.84 quantiles
        self.param_quantiles = np.array(
            [af.quantiles(self.posterior[:,i], q=[0.16,0.5,0.84]) \
             for i in range(self.posterior.shape[1])]
             )
        # Base the axes-limits off of the quantiles
        self.param_range = np.array(
            [(4*(q_i[0]-q_i[1])+q_i[1], 4*(q_i[2]-q_i[1])+q_i[1]) \
             for q_i in self.param_quantiles]
            )

        self.median_params = np.array(list(self.param_quantiles[:,1]))

        if fig is None:
            fig = plt.figure(figsize=(20,20)) # was (15,15) before...
        
        # Only select the included parameters
        mask_params = np.ones(len(self.Param.param_keys), dtype=bool)
        # if included_params is not None:
        #     mask_params = np.isin(
        #         self.Param.param_keys, test_elements=included_params
        #         )

        # Number of parameters
        n_params = mask_params.sum()
        labels = self.param_labels[mask_params]
        def test_labels(labels):
            bad_labels = []
            for label in labels:
                try:
                    plt.figure()
                    plt.text(0.5, 0.5, label, fontsize=12)
                    plt.title(f'Test: {label}')
                    # plt.show()
                    plt.close()
                    print(f"Label {label} rendered successfully")
                except Exception as e:
                    print(f"Error with label: {label}\n{e}")
                    bad_labels.append((label, str(e)))
            return bad_labels

        # Test each label and capture bad labels
        check_labels = False
        if check_labels:
            bad_labels = test_labels(labels)
            for i, (label, error) in enumerate(bad_labels):
                print(f"Error with label: {label}\n{error}")
                # replace label with index
                labels = labels.replace(label, str(i))
                
        # remove following characters from labels: [$, \, _, mathrm, {, }. ^]
        # for i, label in enumerate(labels):
        #     labels[i] = label.replace('$', '').replace('\\', '').replace('_', '').replace('mathrm', '').replace('{', '').replace('}', '').replace('^', '')
        
        assert len(labels) == n_params, f'len(labels)={len(labels)} != n_params={n_params}'
        # print(labels)
        # replace labels
        replace = {'$\\log\\ P_\\mathrm{OH}_0$' : '$\\log \\mathrm{P(OH)}$',
                    '$\\log\\ P_\\mathrm{H_2O}_0$' : '$\\log \\mathrm{P(H_2O)}$',
                    '$\\log\\ P_\\mathrm{H2O}_0$' : '$\\log \\mathrm{P(H_2O)}$',
                    '$\\log\\ \\mathrm{H2O}_0$' : '$\\log \\mathrm{H_2O} (0)$',
                    '$\\log\\ \\mathrm{H2O}_1$' : '$\\log \\mathrm{H_2O} (1)$',
                    '$\\log\\ \\mathrm{H2O}_2$' : '$\\log \\mathrm{H_2O} (2)$',
                    '$\\log\\ \\mathrm{OH}_0$' : '$\\log \\mathrm{OH} (0)$',
                    '$\\log\\ \\mathrm{OH}_1$' : '$\\log \\mathrm{OH} (1)$',
                    '$\\log\\ \\mathrm{OH}_2$' : '$\\log \\mathrm{OH} (2)$',
                    # add the 12CO
                    '$\\log\\ \\mathrm{12CO}_0$' : '$\\log \\mathrm{12CO} (0)$',
                    '$\\log\\ \\mathrm{12CO}_1$' : '$\\log \\mathrm{12CO} (1)$',
                    '$\\log\\ \\mathrm{12CO}_2$' : '$\\log \\mathrm{12CO} (2)$',
                    '$\\log\\ P_\\mathrm{12CO}_0$' : '$\\log \\mathrm{P(12CO)}$',
        }
        labels = [replace.get(label, label) for label in labels]
        # # replace last 10 labels
        # labels[-20:] = [i for i in range(20)]
        # labels = [str(i) for i in range(n_params)]
        # print(labels)

        fig = corner.corner(
            self.posterior[:,mask_params], 
            fig=fig, 
            quiet=True, 
            # labels=self.param_labels[mask_params], 
            labels=labels,
            show_titles=True, 
            use_math_text=True, 
            title_fmt='.2f', 
            title_kwargs={'fontsize':9}, 
            labelpad=0.25*n_params/17, 
            range=self.param_range[mask_params], 
            bins=20, 
            max_n_ticks=3, 

            quantiles=[0.16,0.84], 
            title_quantiles=[0.16, 0.5, 0.84],  # Add this line

            color=self.posterior_color, 
            linewidths=0.5, 
            hist_kwargs={'color':self.posterior_color}, 

            #levels=(1-np.exp(-0.5),),
            fill_contours=True, 
            plot_datapoints=self.evaluation, 

            contourf_kwargs={'colors':self.envelope_colors}, 
            smooth=smooth, 

            contour_kwargs={'linewidths':0.5}, 
            )

        # Add the best-fit and median values as lines
        corner.overplot_lines(fig, self.bestfit_params[mask_params], c=self.bestfit_color, lw=0.5)
        corner.overplot_lines(fig, self.median_params[mask_params], c=self.posterior_color, lw=0.5)

        # Reshape the axes to a square matrix
        ax = np.array(fig.axes)
        for ax_i in ax:
            ax_i.tick_params(axis='both', direction='inout')

        ax = ax.reshape((int(np.sqrt(len(ax))), 
                         int(np.sqrt(len(ax))))
                        )

        for i in range(ax.shape[0]):
            # Change linestyle of 16/84th percentile in histograms
            ax[i,i].get_lines()[0].set(linewidth=0.5, linestyle=(5,(5,5)))
            ax[i,i].get_lines()[1].set(linewidth=0.5, linestyle=(5,(5,5)))

            # Show the best-fitting value in histograms
            ax[i,i].annotate(
                r'$'+'{:.2f}'.format(self.bestfit_params[mask_params][i])+'$', 
                xy=(0.95,0.95), xycoords=ax[i,i].transAxes, 
                color=self.bestfit_color, rotation=0, ha='right', va='top', 
                fontsize=ann_fs
                )
            # Adjust the axis-limits
            for j in range(i):
                ax[i,j].set(ylim=self.param_range[mask_params][i])
            for h in range(ax.shape[0]):
                ax[h,i].set(xlim=self.param_range[mask_params][i])
        
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0, hspace=0)

        return fig, ax
    
    def plot_PT(ax=None, **kwargs):
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7, 7))
            
        ax.plot()
        

    def fig_summary(self):

        fig, ax = self.fig_corner()

        n_w_set = len(self.d_spec)
        
        # --> fig = corner.corner()
        l, b, w, h = [0.4,0.70,0.57,0.25] # left, bottom, width, height
        # --> ax_PT = fig.add_axes([l,b,w,h])
        # --> plot_PT(ax=ax_PT)
        
        ax_spec, ax_res = [], []        
        #ax_res_dims, ax_spec_dims = [], []
        for i, w_set in enumerate(list(self.d_spec.keys())):
            ax_res_dim_i  = [l, b+i*(h+0.03)/(n_w_set), w, 0.97*h/(5*n_w_set)]
            ax_spec_dim_i = [l, ax_res_dim_i[1]+ax_res_dim_i[3], w, 4*0.97*h/(5*n_w_set)]

            #ax_res_dims.append(ax_res_dim_i)
            #ax_spec_dims.append(ax_spec_dim_i)
            ax_spec.append(fig.add_axes(ax_spec_dim_i))
            ax_res.append(fig.add_axes(ax_res_dim_i))

        ax_spec = np.array(ax_spec)[::-1]
        ax_res  = np.array(ax_res)[::-1]
        for i, w_set in enumerate(list(self.d_spec.keys())):
            # Plot the best-fitting spectrum
            ax_spec[i], ax_res[i] = figs.fig_bestfit_model(
                d_spec=self.d_spec[w_set], 
                m_spec=self.m_spec[w_set], 
                LogLike=self.LogLike[w_set], 
                Cov=self.Cov[w_set], 
                bestfit_color=self.bestfit_color, 
                ax_spec=ax_spec[i], 
                ax_res=ax_res[i], 
                prefix=self.prefix, 
                xlabel=['Wavelength (nm)', None][i]
                )

        # ax_VMR = fig.add_axes([0.65,0.43,0.1,0.22])
        # l, b, w, h = ax_VMR.get_position().bounds
        ax_PT = fig.add_axes([0.66,0.44,0.2,0.22])
        l, b, w, h = ax_PT.get_position().bounds
        ax_grad = fig.add_axes([l+w+0.01,b,w*0.52,h])
        ax_grad.grid()

        
        figs.fig_PT(
            PT=self.PT, 
            ax=ax_PT, 
            ax_grad=ax_grad,
            fig=fig,
            bestfit_color='C0',
            envelopes_color='C0',
            int_contr_em_color='red',
            # text_color='gray',
            # weigh_alpha=True,
            show_photosphere=True,
            show_knots=True,
            show_text=True,
            xlim=(1000, 7000), # fix view
            xlim_grad=(-0.02, 0.34),
            # fig_name=self.prefix+f'plots/PT_grad_profile.pdf',
        )

        label = 'final' if self.evaluation else f'live_{self.cb_count}'
        if self.evaluation:
            if self.plot_histograms:
                for i in range(ax.shape[0]):
                    # Plot the histograms separately
                    figs.fig_hist_posterior(
                        posterior_i=self.posterior[:,i], 
                        param_range_i=self.param_range[i], 
                        param_quantiles_i=self.param_quantiles[i], 
                        param_key_i=self.Param.param_keys[i], 
                        posterior_color=self.posterior_color, 
                        title=self.param_labels[i], 
                        bins=20, 
                        prefix=self.prefix
                        )
                
        #     fig.savefig(self.prefix+'plots/final_summary.pdf')
        #     # fig.savefig(self.prefix+f'plots/final_summary.png', dpi=100)

        # else:
        #     fig.savefig(self.prefix+'plots/live_summary_{self.cb_count}.pdf')
            
            # fig.savefig(self.prefix+f'plots/live_summary_{self.cb_count}.png', dpi=100)
        fig_name = self.prefix+f'plots/{label}_summary.pdf'
        print(f' Saving... {fig_name}')
        fig.savefig(fig_name)
        print(f'- Saved {fig_name}')

        plt.close(fig)

        for w_set in self.d_spec.keys():
            # Plot the best-fit spectrum in subplots
            figs.fig_bestfit_model(
                d_spec=self.d_spec[w_set], 
                m_spec=self.m_spec[w_set], 
                LogLike=self.LogLike[w_set], 
                Cov=self.Cov[w_set], 
                bestfit_color=self.bestfit_color, 
                ax_spec=None, 
                ax_res=None, 
                prefix=self.prefix, 
                w_set=w_set
                )
            