import numpy as np
from scipy.optimize import nnls


class LogLikelihood:

    def __init__(self, 
                 d_spec, 
                 n_params, 
                 scale_flux=False, 
                 scale_err=False, 
                 scale_flux_eps=0.05,
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
        
    def __call__(self, m_spec, Cov, 
                 is_first_w_set=False, 
                 ln_L_penalty=0, 
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
        self.ln_L = 0
        self.chi_squared = 0

        N_knots = m_spec.flux.shape[0]
        # Array to store the linear flux-scaling terms
        self.f    = np.ones((N_knots, self.d_spec.n_orders, self.d_spec.n_dets))
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

                m_flux_ij = m_spec.flux[:,i,j,mask_ij] # shape must be (n_knots, n_orders, n_dets, n_pixels)
                d_flux_ij = self.d_spec.flux[i,j,mask_ij]
                d_err_ij  = Cov[i,j].err

                res_ij = (d_flux_ij - m_flux_ij)
                
                if Cov[i,j].is_matrix:
                    # Retrieve a Cholesky decomposition
                    Cov[i,j].get_cholesky()
                    # print(f' Cholesky shape {Cov[i,j].cov_cholesky.shape}')

                # Get the log of the determinant (log prevents over/under-flow)
                Cov[i,j].get_logdet()
                
                # Set up the log-likelihood for this order/detector
                # Chi-squared and optimal uncertainty scaling terms still need to be added
                ln_L_ij = -(N_ij/2*np.log(2*np.pi) + 1/2*Cov[i,j].logdet)
                
                
                 # Without linear scaling of detectors
                f_ij = 1
                if N_knots > 1:
                    f_ij = self.solve_linear(d_flux_ij, m_flux_ij, Cov[i,j])
                    if ((i+j) == 0 and m_spec.fit_radius) or (not self.scale_flux):
                        # NEW 2024-05-26: recover the absolute scaling by dividing by the central value
                        f_ij_ref = f_ij[len(f_ij)//2]
                        if f_ij_ref == 0:
                            print(f'Zero scaling factor in order {i}, detector {j}')
                            self.ln_L = -np.inf
                            return self.ln_L
                        
                        f_ij /= f_ij_ref
                        if (i+j) > 0:
                            # allow a 5% maximum deviation from the reference scaling
                            eps = min(self.scale_flux_eps, f_ij_ref-1.0) if f_ij_ref > 1.0 else max(-self.scale_flux_eps, f_ij_ref-1.0)
                            f_ij *= (1.0 + eps)
                        
                    m_flux_ij_scaled = f_ij @ m_flux_ij
                    
                else:
                    # Without linear scaling of detectors
                    apply_flux_scaling = self.scale_flux and (not (i==0 and j==0) or not is_first_w_set)
                    if self.scale_flux_all: # override previous setting
                        apply_flux_scaling = True
                    # if self.scale_flux and (not (i==0 and j==0) or not is_first_w_set):
                    if apply_flux_scaling:
                        # Only scale the flux relative to the first order/detector

                        # Scale the model flux to minimize the chi-squared error
                        m_flux_ij_scaled, f_ij = self.get_flux_scaling(d_flux_ij, m_flux_ij[0], Cov[i,j])
                        if (i+j) == 0 and self.fit_radius:
                            # NEW 2024-05-26: recover the absolute scaling by dividing by the central value
                            f_ij_ref = f_ij[len(f_ij)//2]
                            f_ij /= f_ij_ref
                            m_flux_ij_scaled /= f_ij_ref
                        # print(f' FLux scaling {f_ij}')
                    else:
                        # No additional flux scaling
                        m_flux_ij_scaled = m_flux_ij[0]
                        
                
                # print(f'FLux scaling {f_ij}')
                res_ij = (d_flux_ij - m_flux_ij_scaled)
                if np.sum(np.isnan(res_ij)) > 0: 
                    print(f'NaNs in residuals: {np.sum(np.isnan(res_ij))}')
                    self.ln_L = -np.inf
                    return self.ln_L
                
                # Chi-squared for the optimal linear scaling
                inv_cov_ij_res_ij = Cov[i,j].solve(res_ij)
                chi_squared_ij_scaled = np.dot(res_ij, inv_cov_ij_res_ij)
                
                if self.scale_err:
                    # Scale the flux uncertainty that maximizes the log-likelihood
                    beta_ij = self.get_err_scaling(chi_squared_ij_scaled, N_ij)
                else:
                    # No additional uncertainty scaling
                    beta_ij = 1

                # Chi-squared for optimal linear scaling and uncertainty scaling
                chi_squared_ij = 1/beta_ij**2 * chi_squared_ij_scaled

                # Add chi-squared and optimal uncertainty scaling terms to log-likelihood
                ln_L_ij += -0.5 * N_ij*np.log(beta_ij**2) 
                ln_L_ij += -0.5 * chi_squared_ij

                # Add to the total log-likelihood and chi-squared
                self.ln_L += ln_L_ij
                #self.chi_squared += chi_squared_ij
                self.chi_squared += np.nansum((res_ij/d_err_ij)**2)

                # Store in the arrays
                self.f[:,i,j] = f_ij
            
                self.beta[i,j] = beta_ij
                self.m_flux[i,j,mask_ij] = m_flux_ij_scaled

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
