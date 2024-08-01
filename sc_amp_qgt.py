#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 12:29:26 2023

@author: pp423
"""
import numpy as np
from scipy.optimize import linprog
from numpy.random import binomial
import cvxpy as cp




def fpr_fnr(beta_est, beta_0):    
    fp = np.sum((beta_est == 1) & (beta_0 == 0))
    tp = np.sum((beta_est == 1) & (beta_0 == 1))
    fn = np.sum((beta_est == 0) & (beta_0 == 1))
    tn = np.sum((beta_est == 0) & (beta_0 == 0))
    print(fp, tp, fn, tn)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    err_r = (fp + fn)/len(beta_0)
    return fpr, fnr, err_r
    

def quantize_new(beta_hat, thresh):
    beta_hat[beta_hat <= thresh] = 0
    beta_hat[beta_hat > thresh] = 1
    return beta_hat


def run_LP(n, p, X, y):  
  # the X and y inputs are uncentered and unscaled.
  obj = np.ones(p)
  LHS_ineq = np.concatenate((np.eye(p),-np.eye(p)))
  RHS_ineq = np.concatenate((np.ones(p),np.zeros(p)))
  LHS_eq = X
  RHS_eq = y
  opt = linprog(c=obj, A_ub=LHS_ineq, b_ub=RHS_ineq, A_eq=LHS_eq, b_eq=RHS_eq)
  print("Linear program:", opt.message)
  sol = opt.x
  #sol = quantize_new(sol, threshold)
  return sol

def run_NP_sc(n, p, y, X, N_sc, sigma, pi):
    
    one_p = np.ones(p)
    beta_opt = cp.Variable(p)
    constraints = []
    constraints.append(beta_opt <= np.ones(p))
    constraints.append(beta_opt >= np.zeros(p))
    constant = 1/(N_sc*(sigma**2))
    const_2 = np.log(1- pi) - np.log(pi)
    objective = cp.Minimize(constant*cp.sum_squares(y - X @ beta_opt) + const_2*cp.scalar_product(beta_opt,one_p))
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP)
    print('optimal obj value:', problem.value)
    beta_est = beta_opt.value
    
    return beta_est


'''=== iid design matrix ==='''
def create_X_iid(alpha, m, n):
    X = binomial(1, alpha, (m, n))
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

def create_beta(nu, n):
    beta_0 = np.zeros(n)
    for i in range(len(beta_0)):
        rand = np.random.random_sample()
        if rand < nu:
            beta_0[i] = 1
    return beta_0

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
            
def y_iid_to_y_iid_tilde(y, alpha, nu, m, n, defect_no='None'):
    if defect_no == 'None':
        #Estimate no. defective items - doesn't work well
        y_tilde = (1/np.sqrt(m*alpha*(1-alpha)))*(y - alpha*n*nu)
    else:
        y_tilde = (1/np.sqrt(m*alpha*(1-alpha)))*(y - alpha*defect_no)
    return y_tilde

def y_sc_to_y_sc_tilde(y, alpha, W, nu, omega, m, n, defect_no='None'):
    """defect_no here is a length-C array where the cth element corresponds
    to the number of defective items in the cth column-block of beta_0"""
    R, C = np.shape(W)
    N = int(n/C)
    M = int(m/R)
    #print("M: ", M)
    #print("N: ", N)
    if defect_no == 'None':
        #Estimate no. defective items - doesn't work well
        y_tilde = (1/np.sqrt(m*alpha*(1-alpha)/R))*(y - alpha*W[0,0]*omega*N*nu)
    else:
        y_tilde = np.zeros(m)
        for r in range(R):
            y_tilde[r*M:(r+1)*M] = (1/np.sqrt(m*alpha*(1-alpha)/R))*(y[r*M:(r+1)*M] - alpha*np.sum(W[r,:]*defect_no))
    return y_tilde

def W_to_Wtilde(W, alpha):
    W_tilde = (W*(1-alpha*W))/(1-alpha)
    return W_tilde


#Denoising function g_in
def g_in_bayes(s, tau_2, nu):
    if tau_2 < 1e-3:
        output = g_in_bayes(s, 1e-3, nu)
    else:
        gamma = s/tau_2
        beta = 1/(2*tau_2)
        m_1 = np.maximum(gamma, 0)
        m_2 = np.maximum(gamma, beta)
        num = nu*np.exp(gamma-m_1)
        denomin = (1-nu)*np.exp(beta-m_2) + nu*np.exp(gamma-m_2)
        output = np.exp(m_1 - m_2)*(num/denomin)
    return output

def deriv_g_in_bayes(s, tau_2, nu):
    if tau_2 < 1e-3:
        output = deriv_g_in_bayes(s, 1e-3, nu)
    else:
        gamma = s/tau_2
        beta = 1/(2*tau_2)
        m_3 = np.maximum(beta+gamma, 0)
        m_2 = np.maximum(gamma, beta)
        num = np.exp(beta+gamma-m_3)
        denomin = ((1-nu)*np.exp(beta-m_2) + nu*np.exp(gamma-m_2))**2
        output = np.exp(m_3 - 2*m_2)*((nu*(1-nu))/tau_2)*(num/denomin)
    return output





def create_A_iid(m,n):
    A_iid = np.random.normal(0,1/np.sqrt(m),size=(m,n))
    return A_iid




def amp_bayes(X, X_T, y, t, nu, beta_0):
    m, n = len(X), len(X[0])
    delta_true = m/n
    
    tau_array = []
    error_norm_array = []
    nc_array = []
    
    tau_2 = 100
    #Initialise x, z and tau
    beta = np.ones(n)*nu   #Initialise x
    for iter_no in range(t):
        print(iter_no)
        if iter_no==0:
            z = y - np.dot(X, beta)
            tau_prev = 0
        else:
            Onsager = (1/m) * z * np.sum(deriv_g_in_bayes(betaXz, tau_2, nu))
            z             = y - np.dot(X,beta) + Onsager
            tau_prev = np.copy(tau)

        tau_2  = (np.linalg.norm(z, ord=2)**2)/ m
        #tau_2 = ((1/delta_true)*se.mmse_new(np.sqrt(1/tau_2), nu))
        print(tau_2)
        betaXz = np.dot(X_T, z) + beta
        beta     = g_in_bayes(betaXz, tau_2, nu)
        #print(x)

        beta = np.nan_to_num(beta)
        #print(x)
        tau = np.sqrt(tau_2)
        tau_array.append(tau)
        #print(tau)
        mse = (1/n)*(np.linalg.norm(beta - beta_0)**2)
        norm_correl = (np.dot(beta, beta_0)/(np.linalg.norm(beta)*np.linalg.norm(beta_0)))**2

        error_norm_array.append(mse)
        nc_array.append(norm_correl)
        
        #Stopping criterion - Relative norm tolerance
        if (tau - tau_prev)**2/tau_2 < 1e-9:
            break
        
        tau_prev = tau
        
    mse_pred = delta_true*(tau_2)

    #print((1/N)*(np.linalg.norm(x_final - x_0)**2))
    return beta, mse_pred, tau_array, error_norm_array, nc_array





def sc_amp_bayes(W, X, y, beta_0, nu, delta_hat, t):
    m, n = len(X), len(X[0])
    R, C = len(W), len(W[0])
    M = int(m/R)#
    N = int(n/C)
    Q = np.zeros((m,n))
    Q_tilde = np.zeros((R,C))
    deriv = np.zeros(C)
    phi = np.ones(R)*1e6
    psi = np.ones(C)*1e3
    phi_prev = np.zeros(R)
    b = np.zeros(m)
    
    mse_list = []
    
    #Initialise x, z and tau
    beta = np.zeros(n)   #Initialise x
    for iter_no in range(t):
        #print(iter_no)
        if iter_no==0:
            z = y
            phi_prev = np.zeros(R)
        else:
            for r in range(R):
                b[M*r:(M*r+M)]= (1/delta_hat)*np.sum(W[r,:]*Q_tilde[r,:]*deriv)
            Onsager = b * z
            z = y - np.dot(X,beta) + Onsager
            phi_prev = np.zeros(R) + phi

        #Calculate phi from z
        for r in range(R):
            phi[r] = (1/M)*(np.linalg.norm(z[M*r:(M*r+M)],ord=2)**2) #+ 1e-50
        #print("Phi: ", phi)
        
        
        
        #Calculate Q from phi
        for c in range(C):
            denomin = np.dot((1/phi),W[:,c])
            for r in range(R):
                Q[M*r:(M*r+M), N*c:(N*c + N )]=((1/phi[r])/denomin)*np.ones((M,N))
                Q_tilde[r,c]=(1/phi[r])/denomin
        
        denoise_arg = beta + np.dot(np.transpose(Q*X),z)
        for c in range(C):
            denoise_argument = denoise_arg[N*c:(N*c + N )]
            s = np.dot((1/phi),W[:,c])
            beta[N*c:(N*c + N )] = g_in_bayes(denoise_argument, s**(-1), nu)
            deriv[c]= (1/N)*np.sum(deriv_g_in_bayes(denoise_argument, s**(-1), nu))

        """#Exact Phi
        for r in range(R):
            phi[r]= (1/delta_hat)*np.dot(psi, W[r,:]) + 1e-50
        print("Phi: ", phi)
        for c in range(C):
            psi[c]=se.mmse_new(np.sqrt(np.dot(W[:,c],(1/phi))), nu)"""
            
        #print(x)
        print("Phi: ", phi)
        mse = (1/n)*(np.linalg.norm(beta - beta_0)**2)

        mse_list.append(mse)
        
        #Stopping criterion - Relative norm tolerance
        if (np.linalg.norm(phi - phi_prev, 2)/np.linalg.norm(phi, 2)) < 1e-6:
            break
        #print(mse, (np.linalg.norm(phi - phi_prev, 2)))
        

    return phi, beta, mse_list



def create_A_sc(W,N, delta_hat):
    R, C = np.shape(W)
    M = int(delta_hat*N)
    m, n = M*R, N*C
    A_sc = np.zeros((m,n))
    for c in range(C):
        for r in range(R):
            A_sc[M*r:(M*r+M), N*c:(N*c + N )] = np.random.normal(0, np.sqrt((1/M)*W[r,c]), (M, N))
    return A_sc

