#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 17:47:42 2023

@author: pp423
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import linprog
import cvxpy as cp


def quantize_new(beta_hat, thresh):
    beta_hat[beta_hat < thresh] = 0
    beta_hat[beta_hat >= thresh] = 1
    return beta_hat

''' Optimization methods '''


def run_LP(n, p, L, Y, X, B_prop):

  # Configuring our inputs to suit LP
  Y_LP = Y.flatten('F')
  X_LP = np.zeros((n*L,p*L))
  for l in range(L):
    X_LP[n*l:n*(l+1),p*l:p*(l+1)] = X
  C_LP = np.eye(p)
  for l in range(L-1):
    C_LP = np.concatenate((C_LP, np.eye(p)), axis=1)
  one_p = np.ones(p)

  # Setting the objective vector
  c = np.zeros(p*L)
  for l in range(L):
    I_pL = np.eye(p*L)
    I_pL_trun = I_pL[p*l:p*(l+1),:]
    c -= np.log(B_prop[l]) * np.dot(one_p.T,I_pL_trun)

  # Setting the equality constraints matrix
  A = np.concatenate((X_LP, C_LP), axis=0)
  # Setting the equality constraints vector
  b = np.concatenate((Y_LP, one_p), axis=0)

  # Setting the inequality constraints matrix
  G = np.concatenate((np.eye(p*L), -np.eye(p*L)), axis=0)
  # Setting the inequality constraints vector
  h = np.concatenate((np.ones(p*L), np.zeros(p*L)), axis=0)

  # Solving linear programming problem
  res = linprog(c, A_eq=A, b_eq=b, A_ub=G, b_ub=h)
  print("Linear program:", res.message)

  # Reconfiguring our outputs to suit pooled data
  B_LP_est = res.x
  B_est = np.zeros((p,L))
  for l in range(L):
    B_col = B_LP_est[p*l:p*(l+1)]
    B_est[:,l] = B_col

  return B_est

def run_NP(n, p, L, Y, X, sigma, B_prop):

  # Configuring our inputs to suit LP
  Y_opt = Y.flatten('F')
  X_opt = np.zeros((n*L,p*L))
  for l in range(L):
    X_opt[n*l:n*(l+1),p*l:p*(l+1)] = X
  C_opt = np.eye(p)
  for l in range(L-1):
    C_opt = np.concatenate((C_opt, np.eye(p)), axis=1)
  one_p = np.ones(p)

  # Setting the objective matrix and vector
  q = np.zeros(p*L)
  for l in range(L):
    I_pL = np.eye(p*L)
    I_pL_trun = I_pL[p*l:p*(l+1),:]
    q -= np.log(B_prop[l]) * np.dot(one_p.T,I_pL_trun)

  # Setting the equality constraints matrix
  A = C_opt
  # Setting the equality constraints vector
  b = one_p

  # Setting the inequality constraints matrix
  G = np.concatenate((np.eye(p*L), -np.eye(p*L)), axis=0)
  # Setting the inequality constraints vector
  h = np.concatenate((np.ones(p*L), np.zeros(p*L)), axis=0)

  # Define and solve the CVXPY problem
  constant = 1/(2*p*(sigma**2))
  B_opt = cp.Variable(p*L)
  objective = cp.Minimize(constant*cp.sum_squares(Y_opt - X_opt @ B_opt) + (q.T @ B_opt))
  constraints = []
  constraints.append(G @ B_opt <= h)
  constraints.append(A @ B_opt == b)
  problem = cp.Problem(objective, constraints)
  problem.solve(solver=cp.OSQP)
  print('optimal obj value:', problem.value)

  # Reconfiguring our outputs to suit pooled data
  B_QP_est = B_opt.value
  B_est = np.zeros((p,L))
  for l in range(L):
    B_col = B_QP_est[p*l:p*(l+1)]
    B_est[:,l] = B_col

  return B_est


'''=== iid design matrix ==='''
def create_X_iid(alpha, m, n):
    X = np.random.binomial(1, alpha, (m, n))
    return X

'''=== spatially coupled design matrix ==='''
def create_base_matrix(lam, omega, alpha):
  R = lam + omega - 1
  C = lam
  W = np.zeros((R, C))
  for c in range(C):
      for r in range(c, c + omega):
        if alpha >= 0.5:
          W[r, c] = (1 + np.sqrt(1-4*alpha*(1-alpha)/omega)) / (2*alpha)
        elif alpha < 0.5:
          W[r, c] = (1 - np.sqrt(1-4*alpha*(1-alpha)/omega)) / (2*alpha)
  return W

def create_SC_matrix(W, N, alpha, delin):
  R, C = np.shape(W)
  M = int(delin*N)
  m, n = M*R, N*C
  print(m,n)
  X_sc = np.zeros((m,n))
  for c in range(C):
      for r in range(R):
          X_sc[M*r:(M*r+M), N*c:(N*c + N )] = np.random.binomial(1, alpha * W[r,c], (M, N))
  return X_sc

#Can this be done without for loop?
def create_B(pi, n):
    assert np.sum(pi)==1
    L = len(pi)
    B_int = np.random.choice(L,n,p=pi)
    B = np.zeros((n, L))
    for i in range(len(B)):
        B[i, B_int[i]] = 1
    return B

def Xiid_to_Xtilde(X_iid, alpha):
    m, n = np.shape(X_iid)
    X_tilde = (1/np.sqrt(m*alpha*(1-alpha)))*(X_iid - alpha)
    return X_tilde

#print(Xiid_to_Xtilde(X, alpha))

def Xsc_to_Xsctilde(X_sc, W, alpha):
    m, n = np.shape(X_sc)
    R, C = np.shape(W)
    M, N = int(m/R), int(n/C)
    lam = C
    omega = R + 1 - lam
    X_sc_tilde = np.zeros((m,n))
    for c in range(C):
        for r in range(c, c + omega):
            X_sc_tilde[M*r:(M*r+M), N*c:(N*c + N )] = (1/np.sqrt(m*alpha*(1-alpha)/R))*(X_sc[M*r:(M*r+M), N*c:(N*c + N )] - alpha*W[r,c])
    return X_sc_tilde        
            
def Y_iid_to_Y_iid_tilde(Y, alpha, m, n, pi_true='None'):
    if pi_true == 'None':
        #Estimate no. defective items - doesn't work well
        print("True pi not known")
    else:
        Y_tilde = (1/np.sqrt(m*alpha*(1-alpha)))*(Y - alpha*n*pi_true)
    return Y_tilde

def Psi_to_Psi_tilde(Psi, alpha, m):
    Psi_tilde = (1/np.sqrt(m*alpha*(1-alpha)))*Psi
    return Psi_tilde

def Y_sc_to_Y_sc_tilde(Y, alpha, W, m, n, M, N, pi_true=None):
    """defect_no here is a length-C array where the cth element corresponds
    to the empirical distribution in the cth column-block of B_0"""
    R, C = np.shape(W)
    if pi_true is None:
        #Estimate no. defective items - doesn't work well
        print("True pi not known")
    else:
        L = len(pi_true[0])
        Y_tilde = np.zeros((m, L))
        for r in range(R):
            Y_tilde[r*M:(r+1)*M] = (1/np.sqrt(M*alpha*(1-alpha)))*(Y[r*M:(r+1)*M] - alpha*N*np.einsum('i, ij -> j', W[r,:], pi_true))
    return Y_tilde

def W_to_Wtilde(W, alpha):
    W_tilde = (W*(1-alpha*W))/(1-alpha)
    return W_tilde


def eta_pool(S, T_B, pi, calcJac=False):
    """
    Calculates the MMSE denoising function eta(S) = E[B | B + G = S]
    for general r>=1.

    B: n-by-L signal matrix; row-wise iid.
       Each row has a single 1, 1 is in ith column with probability pi[i]. 
    G: n-by-L Gaussian (noise) matrix; row-wise iid.
       Each row is drawn from L-dim Gaussian with mean 0 and covariance
       matrix T_B.
       Note that T_B can be obtained either empirically on the fly via
       the residual matrix Z in the AMP algo (see the amp_mmse function
       below) or from SE.
    When calcJac=True, the function also calculates the Jacobian of eta wrt
    its argument.
    The function applies row-wise to matrix inputs.

    Outputs:
        eta:    the n-by-L matrix eta(S)
        etaJac: n-by-L-by-L tensor; each r-by-r matrix is the Jacobian
                matrix corresponding to each row of S
    """
    assert np.sum(pi) == 1
    check_cov_mat(T_B)
    #Sigma = scalar_to_arr2d(Sigma)
    assert S.ndim == 2 and S.shape[1] == T_B.shape[0]
    p, L = S.shape
    etaJac = []  # initialise as empty
    if np.allclose(T_B, 0):  # np.array_equal(Sigma, np.zeros((r, r))):
        eta = S
        if calcJac:
            etaJac = np.tile(np.eye(L), (p, 1, 1))
    else:
        cb = np.eye(L)  # L-by-L, list of all posible codewords - implicit in code
        # Numerator:
        T_B_inv_cb = np.linalg.solve(T_B, cb)
        assert T_B_inv_cb.shape == (L, L)
        cb_T_B_inv_cb = np.sum(T_B_inv_cb.T*cb, axis=1, keepdims=True)
        #cb_T_B_inv_cb = np.diag(T_B_inv)
        #print(cb_T_B_inv_cb.shape)
        assert cb_T_B_inv_cb.shape == (L, 1)
        S_T_B_inv_cb = (S @ T_B_inv_cb).T #np.einsum('ij, hj -> hi', T_B_inv, S)
        assert S_T_B_inv_cb.shape == (L, p)
        pwr_arr = -0.5* (cb_T_B_inv_cb - 2 * S_T_B_inv_cb)
        max_pwr = np.max(pwr_arr, axis=0)
        #print(pwr_arr, max_pwr)
        #max_pwr = max_pwr[:, np.newaxis]
        pwr_minus_max_arr = pwr_arr - max_pwr
        expo = np.exp(pwr_minus_max_arr)
        pi_expo = pi[:,np.newaxis]*expo
        num = cb @ pi_expo
        assert num.shape == (L,p)
        denom = np.sum(pi_expo, axis = 0)
        #denom = denom[:, np.newaxis]
        assert denom.shape == (p,)
        eta = (num/denom).T
        assert eta.shape == (p, L)
        
        if calcJac:
            etaJac = etaJac_pool(pi_expo, cb, T_B_inv_cb, num, denom, L, p)

    return eta, etaJac

def etaJac_pool(pi_expo, cb, T_B_inv_cb, num, denom, L, p):
    """
    Same setup as eta_etaJac_mmse, and only used internally by the latter.
    Calculates the r-by-r Jacobian matrix for S given in etaJac.
    etaJac: L-by-r-by-r
    """
    piexpo_TBinv_c = np.einsum('ij,ik->ijk', pi_expo, T_B_inv_cb.T) # dim L-p-L
    sum_piexpo_TBinv_c = np.sum(piexpo_TBinv_c, axis=0)
    sum_c_piexpo_TBinv_c = matmul_2dmat_w_3dmat(cb.T, piexpo_TBinv_c)  # dim L-p-L
    term_1 = sum_c_piexpo_TBinv_c/denom[np.newaxis, :, np.newaxis]
    assert term_1.shape == (L,p,L)
    
    term_2_num = num[:, :, np.newaxis]*sum_piexpo_TBinv_c[np.newaxis, :, :]
    assert term_2_num.shape == (L,p,L)
    term_2_den = (denom**2)[np.newaxis, :, np.newaxis]
    term_2 = term_2_num/term_2_den
    #print(term_2.shape)
    etaJac = np.transpose(term_1 - term_2, axes=(1, 0, 2))
    assert etaJac.shape == (p, L, L)

    assert np.logical_not(
        np.any(np.isnan(etaJac))
    ), "eta output from etaJac_mmse_alpha0 contains nans"  # no nans
    return etaJac

#STATE EVOLUTION
def se_pool(delta, alpha, pi, sigma, iter_max, num_mc_samples):
    L = len(pi)
    """Run SE for the specified setting"""
    SigmaW = ((sigma**2)/(delta*alpha*(1-alpha)))*np.eye(L)
    MSE_mat_arr = np.zeros((iter_max, L, L))
    mse_est_array = np.zeros(iter_max)
    eff_noise_cov_arr = np.zeros((iter_max, L, L))
    correlation_arr = np.zeros(iter_max)
    signal_pwr = np.diag(pi) #analytical solution for E[B0_bar B0_bar^T]
    MSE_mat_arr[0] =  signal_pwr # r-by-r
    eff_noise_cov_arr[0] = SigmaW + (1/delta)*MSE_mat_arr[0]
    mse_est_array[0] = np.average(np.diag(MSE_mat_arr[0,:,:]))
    for t in range(1, iter_max):
        print("se iter: ", t)
        
        Sigma = eff_noise_cov_arr[t-1] + np.eye(L)*1e-2 #Add noise to avoid singularity
        """print(Sigma, np.linalg.eigvals(Sigma))
        #check_cov_mat(Sigma)
        
        Sigma_chol = np.linalg.cholesky(Sigma)
        G_arr = (Sigma_chol @ np.random.randn(L, num_mc_samples)).T
        X_arr = create_B(pi, num_mc_samples)
        S = X_arr + G_arr
        assert S.shape == (num_mc_samples, L)
        eta_S, _ = eta_pool(S, Sigma, pi)
        print(eta_S)
        eta_sqr_arr = np.einsum('ij,ik->ijk', eta_S, eta_S)
        E_S_eta_sqr = np.mean(eta_sqr_arr, axis=0)"""
        
        MSE_mat_arr[t], correlation_arr[t] = se_mse_mc(pi, Sigma, num_mc_samples, signal_pwr)#signal_pwr - E_S_eta_sqr
        
        eff_noise_cov_arr[t] = SigmaW + (1/delta)*MSE_mat_arr[t]
        
        mse_est_array[t] = np.average(np.diag(MSE_mat_arr[t,:,:]))
        
        #correlation_arr[t] = np.mean(np.einsum('ij,ij->i', eta_S, eta_S))
        
        #Stopping criterion - Relative norm tolerance
        if (mse_est_array[t] - mse_est_array[t-1])**2/mse_est_array[t]**2 < 1e-6 or mse_est_array[t]<1e-50:
            mse_est_array = mse_est_array[:t+1]
            MSE_mat_arr = MSE_mat_arr[:t+1]
            eff_noise_cov_arr = eff_noise_cov_arr[:t+1]
            correlation_arr = correlation_arr[:t+1]
            break
        
        print((mse_est_array[t] - mse_est_array[t-1])**2/mse_est_array[t]**2)
        
    return MSE_mat_arr, eff_noise_cov_arr, mse_est_array, correlation_arr


def se_mse_mc(pi, Sigma, num_mc_samples, signal_power):
    L = len(pi)
    X_arr = np.eye(L)
    S_arr = np.zeros((L, num_mc_samples, L))
    EX_eta_sqr_nz_X = np.zeros((L, num_mc_samples, L, L))
    
    Sigma_chol = np.linalg.cholesky(Sigma)
    G_arr = (Sigma_chol @ np.random.randn(L, num_mc_samples)).T
    eta_S_arr = [np.zeros((int(L*num_mc_samples),L))]
    
    for j in range(L):
        S_arr[j] = X_arr[j] + G_arr
        #print(np.shape(S_arr[j]))
        eta_nz_X, _ = eta_pool(S_arr[j], Sigma, pi) #Apply denoiser here
        #print(eta_nz_X)
        EX_eta_sqr_nz_X[j] = pi[j]*np.einsum("ij, ik->ijk", eta_nz_X, eta_nz_X)
        eta_S_arr[j*num_mc_samples:(j+1)*num_mc_samples] = eta_nz_X
        
    EX_eta_sqr = np.sum(EX_eta_sqr_nz_X, axis=0)
    EGEX_eta_sqr = np.mean(EX_eta_sqr, axis=0)
    
    correlation_arr = np.mean(np.einsum('ij,ij->i', eta_S_arr, eta_S_arr))
    
    mse = signal_power - EGEX_eta_sqr
    return mse, correlation_arr

def se_mse_mc_final_thresh(pi, Sigma, num_mc_samples):
    L = len(pi)
    X_arr = np.eye(L)
    S_arr = np.zeros((L, num_mc_samples, L))
    
    Sigma_chol = np.linalg.cholesky(Sigma)
    G_arr = (Sigma_chol @ np.random.randn(L, num_mc_samples)).T
    se_corr_est = 0


    for j in range(L):
        print(np.shape(X_arr[j]))
        S_arr[j] = G_arr + X_arr[j]
        #print(np.shape(S_arr[j]))
        eta_nz_X, _ = eta_pool(S_arr[j], Sigma, pi) #Apply denoiser here
        quant_eta = np.round(eta_nz_X)
        quant_eta_av = np.average(quant_eta, axis=0)
        print(np.shape(quant_eta_av))
        print(np.dot(X_arr[j], quant_eta_av))
        se_corr_est += pi[j]*np.dot(X_arr[j], quant_eta_av)
    
    return se_corr_est

def sc_se_pool(W, delin, pi, alpha, sigma2, iter_max, num_mc_samples):
    """Run Spatially Coupled State Evolution for the specified setting"""
    R, C = len(W), len(W[0])
    
    """SE does not depend on actual value of n,L - only on ratio"""
    #assert n % R == 0
    #assert L % C == 0
    #n_R = n // R
    #L_C = L // C
    #delin = n_R/L_C
    L = len(pi)
    delta = delin*(R/C)
    SigmaW = ((sigma2)/(delta*alpha*(1-alpha)))*np.eye(L)
    
    Psi = np.zeros((C,L,L))
    P_t = np.zeros((C,L,L))
    Phi = np.zeros((R,L,L))
    Phi_inv = np.zeros((R,L,L))
    Psi_diag = np.zeros(C)
    Psi_diag_prev = np.zeros(C)
    
    Phi_array = np.zeros((iter_max,R,L,L))
    Psi_array = np.zeros((iter_max,C,L,L))
    mse_est_array = np.zeros(iter_max)
    correlation_arr = np.zeros(iter_max)
    correlation_arr_c = np.zeros(C)
    
    
    signal_pwr = np.diag(pi)
    
    #X_arr = np.zeros
    #X_arr = create_B(pi, num_mc_samples)
    #G_base_arr = np.random.randn(L, num_mc_samples)
    
    """Initialize each of the Psi matrices as E[X.X^T]"""
    for i in range(C):
        Psi[i] =  signal_pwr
        Psi_diag[i] = np.average(np.diag(Psi[i]))
    for a in range(R):
        Phi[a] = SigmaW + (1/delin)*np.einsum('i,ijk->jk', W[a,:], Psi)
        Phi_inv[a] = np.linalg.inv(Phi[a])
    
    Phi_array[0] = Phi
    Psi_array[0] = Psi
    mse_est_array[0] = np.average(Psi_diag)
    
    for t in range(1, iter_max):
        print("se iter: ", t)
        
        """Update Psi and covariance matrices P_t"""
        for i in range(C):
            print("c=", i)
            
            #print(Phi_inv)
            
            P_t[i] = np.zeros((L,L)) + np.linalg.inv(np.einsum('i,ijk->jk', W[:,i], Phi_inv)) + np.eye(L)*1e-6
            #print(np.linalg.eigvals(P_t[i]), np.min(np.linalg.eigvals(P_t[i])))
            """if np.min(np.linalg.eigvals(P_t[i])) < 1e-10:
                print("Flag - negative eigenvalues")
                #print(np.linalg.eigvals(P_t[i]))
                P_t[i] = P_t[i] + np.eye(L)*1e-2
            
            
            eigvals, eigvecs = np.linalg.eigh(P_t[i])
            if np.min(eigvals) < 0:
                print(P_t[i])
                print(eigvals)
                idx = eigvals.argsort()[::-1]
                print(idx)
                idx = idx[:-1]
                print(idx)
                eigvals = eigvals[idx]
                print(eigvecs)
                F = eigvecs[:,idx]
                D = np.diag(eigvals)
                print(F, D)
                FDF = np.linalg.multi_dot([F, D, F.T])
                print(FDF)
                rec = np.linalg.norm(L - FDF)
                print(rec)
                P_t[i] = np.copy(FDF)
                print(np.linalg.eigvals(P_t[i]))
            
            #print(P_t[i])
            #print(np.linalg.eigvals(P_t[i]), np.linalg.svd(P_t[i])[1])
            #P_t[i] = np.linalg.pinv(np.einsum('i,ijk->jk', W[:,i], Phi_inv), 1e-7) + np.eye(L)*1e-2
            #print(np.linalg.eigvals(P_t[i]), np.linalg.svd(P_t[i])[1])
            
            Sigma_chol = np.linalg.cholesky(P_t[i])
            G_arr = (Sigma_chol @ G_base_arr).T
            
            I_L = np.eye(L)
            
            for j in range(L):
                X_arr = I_L[j]
            
            
             #rng.multivariate_normal(np.zeros(L), P_t[i], num_mc_samples)#, check_valid='ignore')   #
            #X_arr = create_B(pi, num_mc_samples)
            S = X_arr + G_arr
            assert S.shape == (num_mc_samples, L)
            eta_S, _ = eta_pool(S, P_t[i], pi)
            #print(eta_S)
            eta_sqr_arr = np.einsum('ij,ik->ijk', eta_S, eta_S)
            E_S_eta_sqr = np.mean(eta_sqr_arr, axis=0)"""
            
            Psi[i], correl = se_mse_mc(pi, P_t[i], num_mc_samples, signal_pwr)
            Psi_diag[i] = np.average(np.diag(Psi[i]))
            
            correlation_arr_c[i] = correl
            
        """Update Phi and compute the inverse of each of the R matrices"""
        for a in range(R):
            #print("L=", a)
            Phi[a]= SigmaW + (1/delin)*np.einsum('i,ijk->jk', W[a,:], Psi)
            Phi_inv[a] = np.linalg.inv(Phi[a])
        
        Phi_array[t] = Phi
        Psi_array[t] = Psi
        print("Psi_norm", Psi_diag)
        mse_est = np.average(Psi_diag)
        mse_est_array[t] = mse_est
        correlation_arr[t] = np.mean(correlation_arr_c)
        
        print(np.linalg.norm(Psi_diag - Psi_diag_prev, 2)/np.linalg.norm(Psi_diag, 2))
        #Stopping criterion - Relative norm tolerance
        if (np.linalg.norm(Psi_diag - Psi_diag_prev, 2)/np.linalg.norm(Psi_diag, 2)) < 1e-3 or np.linalg.norm(Psi_diag, 2)<1e-10:
            mse_est_array = mse_est_array[:t+1]
            correlation_arr = correlation_arr[:t+1]
            break
        
        Psi_diag_prev = np.zeros(C) + Psi_diag
        
    return Phi_array, Psi_array, P_t, mse_est_array, correlation_arr

def amp_pool(pi, A, Y, max_iter, X0):
    """
    AMP recursion to update the estimate X and the residual Z.
    The model is Y = AX + W.
    Inputs:
        alpha: prob that a row of X is all-zero
        X_amp: L-by-r; current AMP estimate of the signal matrix X
        Z_amp: n-by-r; current AMP residual matrix
        eta_etaJac_fn: function of the form:
            eta, etaJac = eta_etaJac_fn(alpha, S, Sigma, calcJac)
    This function empirically estimates Sigma, i.e. the covariance of
    the effective noise Z. An alternative way is to estimate Sigma via SE.

    The functions invoked in this function are tested externally.
    The remaining lines are simple so arent tested.
    """
    assert np.sum(pi) == 1
    n, p = A.shape
    L = Y.shape[1]
    assert n == Y.shape[0]
    
    scalar_mse_arr = np.zeros(max_iter)
    eff_noise_cov_arr = np.zeros((max_iter, L, L))
    
    #Initialization
    X_amp = np.zeros((p,L))#pi*np.ones((p,L))
    Z_amp = Y
    
    for t in range(max_iter):
    
        print(t)
        eff_noise_cov_arr[t] = np.cov(Z_amp, rowvar=False)
        Sigma = eff_noise_cov_arr[t] + np.eye(L)*1e-5
        #print(Sigma)
        # Effective noisy observation of signal matrix X:
        S = X_amp + A.T @ Z_amp
        #print("Denoise S: ", S[:10], "Covariance: ", Sigma)
        # Update estimate of X and calculate Jacobian:
        X_amp, etaJac = eta_pool(S, Sigma, pi, calcJac=True)
        # Update residual matrix Z using the current Z and the updated X:
        Z_amp = Y - A @ X_amp + (1/n)*Z_amp @ (np.sum(etaJac, axis=0).T)
        
        scalar_mse_arr[t] = np.mean((X_amp - X0) ** 2)
        
        
        print(scalar_mse_arr[t], eff_noise_cov_arr[t])
        
        #Stopping criterion - Relative norm tolerance
        if (np.linalg.norm(eff_noise_cov_arr[t] - eff_noise_cov_arr[t-1], 2)/np.linalg.norm(eff_noise_cov_arr[t], 2)) < 1e-9:
            break
    
    mse_final = scalar_mse_arr[t]
        
    return X_amp, Z_amp, scalar_mse_arr, eff_noise_cov_arr, mse_final






def sc_amp_pool_opt(pi, A, M, N, W, Y, X_0, max_iter):
    """
    Spatially Coupled AMP recursion to update the estimate X and the residual Z.
    The model is Y = AX + Noise.
    Inputs:
        pi: determines law of row RV
        X_amp: p-by-L; current AMP estimate of the signal matrix X
        Z_amp: n-by-L; current AMP residual matrix
        eta_etaJac_fn: function of the form:
            eta, etaJac = eta_etaJac_fn(alpha, S, Sigma, calcJac)
    This function empirically estimates Sigma, i.e. the covariance of
    the effective noise Z. An alternative way is to estimate Sigma via SE.

    The functions invoked in this function are tested externally.
    The remaining lines are simple so arent tested.
    """
    
    scalar_mse_arr = np.zeros(max_iter)


    n, p = A.shape
    delin = M/N
    L = Y.shape[1]
    
    R, C = W.shape
    
    P = np.zeros((C,L,L))
    B = np.zeros((R,L,L))
    B_full = np.zeros((n, L, L))
    Q_tilde = np.zeros((R,C,L,L))
    Phi = np.zeros((R, L, L))
    eta_prime_av_T = np.zeros((C,L,L))
    
    Phi_inv = np.zeros((R, L, L))
    Phi_diag = np.zeros(R)
    Phi_diag_prev = np.zeros(R)
    

    etaJac = np.zeros((p,L,L))
   
    
    X_amp = np.zeros((p,L))

    
    scalar_mse_arr[0] = np.mean((X_amp - X_0)**2)
    
    #Phi_array, _, _, _ = se.sc_se_fn(W,delin,r,alpha,sigma2,100, 100, eta_etaJac_mmse)
    
    for iter_no in range(max_iter):
    
        print(iter_no)    
    
        if iter_no==0:
            Z_amp = Y
            Phi_diag_prev = np.zeros(R)
            
        else:
            for r in range(R):
                #print(r, np.nan_to_num(np.average(etaJac[M*r: M*(r+1)], axis=0).T))
                #WARNING - unsure about transpose in Jacobian
                #Avoid error for n>=L - add nan_to_num in Jacobian
                B[r]= (1/delin)*np.einsum('i,ijk->jk', W[r,:], np.einsum('ijk,ikl->ijl', 
                     Q_tilde[r,:], eta_prime_av_T))
            #print("B: ", B)    
            B_full = np.repeat(B, repeats=M, axis=0)
            Z_tilde = np.einsum('ik,ikl->il', Z_amp, B_full) #Checked - correct multiplication
            #print("Z_tilde: ", Z_tilde)
            Z_amp = Y - (A @ X_amp) + Z_tilde
            #print("Z_amp: ", Z_amp)
            Phi_diag_prev = np.zeros(R) + np.copy(Phi_diag)
        
        #print(Phi_diag_prev)
    
        #Calculate phi from z - approximation of SE
        for r in range(R):
            Phi[r] = np.cov(Z_amp[M*r:(M*r+M)], rowvar=False)
            #print("Phi[r]: ", Phi[r])
            Phi_inv[r] = np.linalg.inv(Phi[r])
            Phi_diag[r] = np.average(np.diag(Phi[r]))
        """
        #Calculate phi exact from SE
        for r in range(R):
            Phi_r = Phi_array[iter_no][r]
            Phi[r] = Phi_r
            #print("Phi[r]: ", Phi[r])
            Phi_inv[r] = np.linalg.inv(Phi_r)
            Phi_diag[r] = np.average(np.diag(Phi_r))
        """
        print(Phi_diag)
        
        #Calculate Q from phi
        for c in range(C):
            P[c] = np.linalg.inv(np.einsum('i,ijk->jk', W[:,c], Phi_inv))
            for r in range(R):
                Q_tilde[r,c]= Phi_inv[r] @ P[c]        

        #print("Denoise arg: ", denoise_arg[:10], "Covariance: ", P)
        for c in range(C):
            #Compute Denoiser Argument
            Q_c_use = np.repeat(Q_tilde[:,c], repeats=M, axis=0)
            Z_Q = np.einsum('ij,ijk->ik', Z_amp, Q_c_use)
            D_c_amp = (A[:, N*c:(N*c + N )]).T@ Z_Q
            
            
            denoise_argument = X_amp[N*c:(N*c + N )] + D_c_amp
            # Update estimate of X and calculate Jacobian:
            X_amp[N*c:(N*c + N)], etaJac[N*c:(N*c + N)] = eta_pool(denoise_argument, P[c], pi, calcJac=True) 

            
            eta_prime_av_T[c] = np.transpose(np.average(etaJac[N*c:(N*c + N)], axis=0))
        #print(etaJac)
        
        
        scalar_mse_arr[iter_no] = np.mean((X_amp - X_0)**2)
        print("MSE: ", scalar_mse_arr[iter_no])
        
        print(np.linalg.norm(Phi_diag - Phi_diag_prev, 2)/np.linalg.norm(Phi_diag, 2))
        
        #Stopping criterion - Relative norm tolerance
        if (np.linalg.norm(Phi_diag - Phi_diag_prev, 2)/np.linalg.norm(Phi_diag, 2)) < 1e-6:
            break
        
        if (np.linalg.norm(Phi_diag)) < 1e-6:
            break
        
    mse_final = scalar_mse_arr[iter_no]
    
    return X_amp, Z_amp, scalar_mse_arr, mse_final



def matmul_2dmat_w_3dmat(mat_2d, mat_3d):
    """
    mat_2d has dimension d0-by-d1
    mat_3d has dimension d0-by-d2-by-d3
    Align mat_2d and mat_3d along their zeroth axis;
    calculate the outer product of each row of mat_2d with
    each layer of mat_3d (spanned between 1st and 2nd axes)
    to yield a 4D tensor;
    sum along the zeroth axis to give res_3d, of dimension d1-by-d2-by-d3.
    """
    assert mat_2d.ndim == 2 and mat_3d.ndim == 3
    assert mat_2d.shape[0] == mat_3d.shape[0]
    d1 = mat_2d.shape[1]
    d2 = mat_3d.shape[1]
    d3 = mat_3d.shape[2]
    res_3d = np.einsum("ij,ikl->jkl", mat_2d, mat_3d)
    assert res_3d.shape == (d1, d2, d3)
    return res_3d

def check_cov_mat(Sigma):
    """Check if Sigma is a valid covariance matrix."""
    if np.ndim(Sigma) == 0:  # scalar
        assert Sigma >= 0
    else:
        assert Sigma.ndim == 2
        assert Sigma.shape[0] == Sigma.shape[1]
        assert np.all(np.linalg.eigvals(Sigma) >= 0)  # +ve semidefinite
        # assert np.linalg.matrix_rank(Sigma) == Sigma.shape[0]
        assert np.allclose(Sigma, Sigma.T)

