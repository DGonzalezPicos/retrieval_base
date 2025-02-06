import numpy as np
from scipy.optimize import nnls
import copy
from retrieval_base.local_covariance_kernel import LocalCovarianceKernel

class LogLikelihood:

    def __init__(self, 
                 d_spec, 
                 n_params, 
                 scale_flux=False, 
                 scale_err=False, 
                 scale_flux_eps=0.05,
                 lck_kwargs={},
                 ):

        # Observed spectrum is constant
        self.d_spec   = d_spec
        self.n_params = n_params

        # Number of degrees of freedom
        self.n_dof = self.d_spec.mask_isfinite.sum() - self.n_params

        self.scale_flux   = scale_flux
        self.scale_flux_eps = scale_flux_eps
        self.scale_err    = scale_err
        
        self.scale_flux_all = False # WARNING: this overrides the previous setting
        
        self.use_lck = (len(lck_kwargs) > 0)
        self.lck_kwargs = lck_kwargs
        
    def __call__(self, 
                 m_spec, 
                 Cov, 
                 evaluation=False):
        '''
        Evaluate the total log-likelihood given the model spectrum and parameters.

        Input
        -----
        m_spec : ModelSpectrum class
            Instance of the ModelSpectrum class.
        Cov : Covariance class
            Instance of the GaussianProcesses or Covariance class. 
        ln_L_penalty : float
            Penalty term to be added to the total log-likelihood. Default is 0.       
        '''
        # Cov = copy.deepcopy(Cov)
        self.ln_L = 0
        self.chi_squared = 0

        # Array to store the uncertainty-scaling terms
        self.beta = np.ones((self.d_spec.n_orders, self.d_spec.n_dets))
        
        self.m_flux = np.nan * np.ones_like(self.d_spec.flux)

        # Loop over all orders and detectors
        for i in range(self.d_spec.n_orders):
            for j in range(self.d_spec.n_dets):

                # Apply mask to model and data, calculate residuals
                mask_ij = self.d_spec.mask_isfinite[i,j,:]

                # Number of data points
                N_ij = mask_ij.sum()
                if N_ij == 0:
                    print(f'No data points in order {i}, detector {j}')
                    continue
                
                m_flux_ij = m_spec.flux[i,j,mask_ij] # shape must be (n_orders, n_dets, n_pixels)
                d_flux_ij = self.d_spec.flux[i,j,mask_ij]
                d_err_ij  = Cov[i,j].err
            
                res_ij = (d_flux_ij - m_flux_ij)
                                   
                if np.sum(np.isnan(res_ij)) > 0: 
                    print(f'NaNs in residuals: {np.sum(np.isnan(res_ij))}')
                    self.ln_L = -np.inf
                    return self.ln_L
                
                if self.use_lck:
                    # print(f' [LogLikelihood.__call__]: using LCK')
                    debug_lck = False
                    lck = LocalCovarianceKernel(self.d_spec.wave[i,j,mask_ij],
                                                d_flux_ij,
                                                d_err_ij,
                                                lck_width=self.lck_kwargs.get('lck_width', 4))
                    lck.s = lck(m_flux_ij, 
                        sigma_threshold=self.lck_kwargs.get('sigma_threshold', 5.0),
                        n_max_regions=self.lck_kwargs.get('n_max_regions', 5))
                   
                    if len(getattr(lck, 'chi2_regions', [])) > 0:
                        if debug_lck:
                            print(f' [LogLikelihood.__call__]: lck.s.shape {lck.s.shape}')
                            print(f' [LogLikelihood.__call__]: lck.s.min() {lck.s.min():.2e} lck.s.max() {lck.s.max():.2e} lck.s.mean() {lck.s.mean():.2e}')
                            print(f' [LogLikelihood.__call__]: lck.regions {lck.regions}')
                            
                        kernel = lck.correlated_kernel(trunc_dist=self.lck_kwargs.get('trunc_dist', 4)) # a_k**2
                        a_k = np.sqrt(Cov[i,j].get_banded(kernel)[:Cov[i,j].separation.shape[0]])
                        del lck

                        if a_k.shape[0] < Cov[i,j].separation.shape[0]:
                            # fill with zeros along axis 0
                            a_k = np.concatenate((a_k, np.zeros((Cov[i,j].separation.shape[0] - a_k.shape[0], a_k.shape[1]))), axis=0)
                        assert a_k.shape[0] == Cov[i,j].separation.shape[0], f'a_k.shape {a_k.shape} != Cov[i,j].separation.shape {Cov[i,j].separation.shape}'
                        Cov[i,j].add_RBF_kernel(a=a_k,
                                                l=self.lck_kwargs.get('lck_width', 4),
                                                array=Cov[i,j].err_eff,
                                                scale=self.lck_kwargs.get('scale_GP_amp', True),
                                                trunc_dist=self.lck_kwargs.get('trunc_dist', 4))
                        del kernel, a_k
                    
                if Cov[i,j].is_matrix:
                    # Retrieve a Cholesky decomposition
                    Cov[i,j].get_cholesky()
                    if np.all(Cov[i,j].cov_cholesky == 0):
                        print(f' [LogLikelihood.__call__]: Cholesky decomposition failed for order {i}, detector {j}')
                        self.ln_L = -np.inf
                        return self.ln_L
                    # print(f' Cholesky shape {Cov[i,j].cov_cholesky.shape}')

                # Get the log of the determinant (log prevents over/under-flow)
                Cov[i,j].get_logdet()
                # check logdet
                # print(f' logdet {Cov[i,j].logdet:.2e}')
                
                # Chi-squared for the optimal linear scaling
                inv_cov_ij_res_ij = Cov[i,j].solve(res_ij)
                # check there's no inf or nan
                assert np.all(np.isfinite(inv_cov_ij_res_ij)), 'There are inf or nan in the inverse covariance matrix'
                # print(f' [LogLikelihood.__call__]: mean(res_ij) {np.mean(res_ij):.2e}')
                # print(f' [LogLikelihood.__call__]: std(res_ij) {np.std(res_ij):.2e}')
                # print(f' [LogLikelihood.__call__]: mean(inv_cov_ij_res_ij) {np.mean(inv_cov_ij_res_ij):.2e}')
                # print(f' [LogLikelihood.__call__]: std(inv_cov_ij_res_ij) {np.std(inv_cov_ij_res_ij):.2e}')
                
                chi_squared_ij_scaled = np.dot(res_ij, inv_cov_ij_res_ij)
                # print(f' chi_squared_ij_scaled {chi_squared_ij_scaled:.2e}')
                if self.scale_err:
                    # Scale the flux uncertainty that maximizes the log-likelihood
                    beta_ij = self.get_err_scaling(chi_squared_ij_scaled, N_ij)
                else:
                    # No additional uncertainty scaling
                    beta_ij = 1

                # Chi-squared for optimal linear scaling and uncertainty scaling
                chi_squared_ij = 1/beta_ij**2 * chi_squared_ij_scaled

                # Add chi-squared and optimal uncertainty scaling terms to log-likelihood
                ln_L_ij = -(N_ij/2*np.log(2*np.pi) + 1/2*Cov[i,j].logdet)
                ln_L_ij += -0.5 * N_ij*np.log(beta_ij**2) 
                ln_L_ij += -0.5 * chi_squared_ij

                # Add to the total log-likelihood and chi-squared
                self.ln_L += ln_L_ij
                #self.chi_squared += chi_squared_ij
                self.chi_squared += np.nansum((res_ij/d_err_ij)**2)
            
                self.beta[i,j] = beta_ij
                self.m_flux[i,j,mask_ij] = m_flux_ij

        # Reduced chi-squared
        self.chi_squared_red = self.chi_squared / self.n_dof

        return self.ln_L

    def get_flux_scaling(self, d_flux_ij, m_flux_ij, cov_ij):
        '''
        Following Ruffio et al. (2019). Find the optimal linear 
        scaling parameter to minimize the chi-squared error. 

        Solve for the linear scaling parameter f in:
        (M^T * cov^-1 * M) * f = M^T * cov^-1 * d

        Input
        -----
        d_flux_ij : np.ndarray
            Flux of the observed spectrum.
        m_flux_ij : np.ndarray
            Flux of the model spectrum.
        cov_ij : Covariance class
            Instance of the Covariance class. Should have a 
            solve() method to avoid matrix-inversion.

        Returns
        -------
        m_flux_ij*f_ij : np.ndarray
            Scaled model flux.
        f_ij : 
            Optimal linear scaling factor.
        '''
        
        # Left-hand side
        lhs = np.dot(m_flux_ij, cov_ij.solve(m_flux_ij))
        # Right-hand side
        rhs = np.dot(m_flux_ij, cov_ij.solve(d_flux_ij))
        
        # Return the scaled model flux
        f_ij = rhs / lhs
        return m_flux_ij * f_ij, f_ij

    def get_err_scaling(self, chi_squared_ij_scaled, N_ij):
        '''
        Following Ruffio et al. (2019). Find the optimal uncertainty
        scaling parameter to maximize the log-likelihood. 

        Input
        -----
        chi_squared_ij_scaled : float
            Chi-squared error of the optimally-scaled model spectrum.
        N_ij : int
            Number of datapoints/pixels in spectrum.

        Returns
        -------
        beta_ij : float
            Optimal uncertainty scaling factor.
        '''

        # Find uncertainty scaling that maximizes log-likelihood
        beta_ij = np.sqrt(1/N_ij * chi_squared_ij_scaled)
        return beta_ij
    
    @staticmethod
    def solve_linear(data, M, Cov):
        '''Solution to the linear system of equations M^T * cov^-1 * M * f = M^T * cov^-1 * d
        using scipy.nnls. This is a non-negative least-squares solver.
        '''
        if Cov.is_matrix:
            lhs = np.dot(M, Cov.solve(M.T))
            # Right-hand side
            rhs = np.dot(M, Cov.solve(data))
        else:
            inv_cov = np.diag(1/Cov.cov)
            lhs = np.dot(M, np.dot(inv_cov, M.T))
            # Right-hand side
            rhs = np.dot(M, np.dot(inv_cov, data))
        # Solve
        f, _ = nnls(lhs, rhs)
        return f
