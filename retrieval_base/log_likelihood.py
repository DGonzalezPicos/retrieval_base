import numpy as np

from scipy.optimize import nnls
from retrieval_base.spline_model import SplineModel

class LogLikelihood:

    # reference_order = 0 # bluest order
    reference_order = 2
    reference_det   = 0
    def __init__(self, 
                 d_spec, 
                 n_params, 
                 scale_flux=False, 
                 scale_err=False, 
                 N_spline_knots=1,
                 ):

        # Observed spectrum is constant
        self.d_spec   = d_spec
        self.n_params = n_params

        # Number of degrees of freedom
        self.n_dof = self.d_spec.mask_isfinite.sum() - self.n_params

        self.scale_flux   = scale_flux
        self.scale_err    = scale_err
        
        self.N_knots = N_spline_knots
        if self.N_knots > 1:
            self.spline = SplineModel(N_knots=self.N_knots, spline_degree=3)
        
    def __call__(self, m_spec, Cov, is_first_w_set=False, ln_L_penalty=0, evaluation=False):
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
        
        #self.m_spec = m_spec

        # Set up the total log-likelihood for this model 
        # (= 0 if there is no penalty)
        #self.ln_L = ln_L_penalty
        self.ln_L = 0
        self.chi_squared = 0

        # Array to store the linear flux-scaling terms
        # if self.N_knots > 1:
        self.phi = np.ones((self.d_spec.n_orders, self.d_spec.n_dets, self.N_knots+m_spec.N_veiling))
        # Array to store the uncertainty-scaling terms
        self.s = np.ones((self.d_spec.n_orders, self.d_spec.n_dets))
        self.m = np.nan * np.ones_like(self.d_spec.flux) # save full model spectrum
        
        if evaluation:
            # Arrays to store log-likelihood and chi-squared per pixel in
            self.ln_L_per_pixel        = np.nan * np.ones_like(self.d_spec.flux)
            self.chi_squared_per_pixel = np.nan * np.ones_like(self.d_spec.flux)
            
            N_tot = self.d_spec.mask_isfinite.sum()
            self.ln_L_per_pixel[self.d_spec.mask_isfinite] = ln_L_penalty/N_tot

        # Loop over all orders and detectors
        for i in range(self.d_spec.n_orders):
            for j in range(self.d_spec.n_dets):

                # Apply mask to model and data, calculate residuals
                # print(f' shape mask_isfinite {self.d_spec.mask_isfinite.shape}')
                # print(f' shape flux {self.d_spec.flux.shape}')
                mask_ij = self.d_spec.mask_isfinite[i,j,:]
                # print(f' order {i}, detector {j} --> {mask_ij.sum()} finite pixels')
                # Number of data points
                N_ij = mask_ij.sum()
                # if N_ij == 0:
                if N_ij < 200:
                    continue
                # print(f' m_spec.flux.shape {m_spec.flux.shape}')
                # m_flux_ij = m_spec.flux[i,j,mask_ij]
                d_flux_ij = self.d_spec.flux[i,j,mask_ij]
                d_err_ij  = Cov[i,j].err
                
                # model matrix, at least shape (1, N_ij)
                M_ij = self.spline(m_spec.flux[0,i,j,mask_ij]) if self.N_knots > 1 else m_spec.flux[:,i,j,mask_ij]

                # res_ij = (d_flux_ij - m_flux_ij)
                
                if Cov[i,j].is_matrix:
                    # Retrieve a Cholesky decomposition
                    Cov[i,j].get_cholesky()
                    Cov[i,j].get_logdet()
                    # print(f' logdet {Cov[i,j].logdet}')
                    # print(f' Cov[i,j].cov_cholesky.shape {Cov[i,j].cov_cholesky.shape}')

                    LHS = np.dot(M_ij, Cov[i,j].solve(M_ij.T))
                    RHS = np.dot(M_ij, Cov[i,j].solve(d_flux_ij))
                else:
                    inv_cov = np.diag(1/Cov[i,j].cov)
                    LHS = np.dot(M_ij, np.dot(inv_cov, M_ij.T))
                    RHS = np.dot(M_ij, np.dot(inv_cov, d_flux_ij))

                # check for NaNs
                # assert not np.isnan(M_ij).any(), f'NaNs in M_ij for order {i} and detector {j}'
                # assert not np.isnan(d_flux_ij).any(), f'NaNs in d_flux_ij for order {i} and detector {j}'
                # assert not np.isnan(d_err_ij).any(), f'NaNs in d_err_ij for order {i} and detector {j}'
                # print(f' M_ij shape {M_ij.shape}')
                try:
                    
                    phi_ij = nnls(LHS, RHS)[0] 
                except RuntimeError:
                    phi_ij = np.zeros(M_ij.shape[0])
                    # phi_ij[1:] =  np.dot(M_ij, Cov[i,j].solve(M_ij.T))
                    # print(f' Error in nnls for order {i} and detector {j}... setting phi_ij to 0.')
                    # return -np.inf
                    

                # Chi-squared for the optimal linear scaling
                m_ij = phi_ij @ M_ij
                res_ij = (d_flux_ij - m_ij)
                inv_cov_ij_res_ij = Cov[i,j].solve(res_ij)
                chi_squared_ij_scaled = np.dot(res_ij, inv_cov_ij_res_ij)
                
                s2_ij = chi_squared_ij_scaled / N_ij
                if (i==self.reference_order) and (j==self.reference_det):
                    s2_ij = 1.0 # set reference order/detector to 1.0
                

                # Chi-squared for optimal linear scaling and uncertainty scaling
                chi_squared_ij = 1/s2_ij * chi_squared_ij_scaled
                
                # if not Cov[i,j].is_matrix:
                #     Cov[i,j].add_data_err_scaling(np.sqrt(s2_ij))
                # Get the log of the determinant (log prevents over/under-flow)
                
                # Set up the log-likelihood for this order/detector
                # Chi-squared and optimal uncertainty scaling terms still need to be added
                ln_L_ij = -(N_ij/2*np.log(2*np.pi) + 1/2*Cov[i,j].logdet)

                # Add chi-squared and optimal uncertainty scaling terms to log-likelihood
                ln_L_ij += -(N_ij/2*np.log(s2_ij) + 1/2*chi_squared_ij)

                # Add to the total log-likelihood and chi-squared
                self.ln_L += ln_L_ij
                #self.chi_squared += chi_squared_ij
                self.chi_squared += np.nansum((res_ij/d_err_ij)**2)

                # Store in the arrays
                self.phi[i,j]       = phi_ij # linear amplitudes
                self.s[i,j]         = np.sqrt(s2_ij) # uncertainty scaling
                self.m[i,j,mask_ij] = m_ij # full model spectrum


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
        m_flux_ij*phi_ij : np.ndarray
            Scaled model flux.
        phi_ij : 
            Optimal linear scaling factor.
        '''
        
        # Left-hand side
        lhs = np.dot(m_flux_ij, cov_ij.solve(m_flux_ij))
        # Right-hand side
        rhs = np.dot(m_flux_ij, cov_ij.solve(d_flux_ij))
        
        # Return the scaled model flux
        phi_ij = rhs / lhs
        return m_flux_ij * phi_ij, phi_ij

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
        s2_ij : float
            Optimal uncertainty scaling factor.
        '''

        # Find uncertainty scaling that maximizes log-likelihood
        s2_ij = np.sqrt(1/N_ij * chi_squared_ij_scaled)
        return s2_ij
    
    # create alias for phi=f and beta=s
    @property
    def f(self):
        return self.phi
    @property
    def beta(self):
        return self.s