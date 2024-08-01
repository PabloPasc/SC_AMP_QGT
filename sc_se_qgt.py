#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 15:56:34 2023

@author: pp423
"""

from numba import jit
import numpy as np
from scipy.integrate import quad


def create_beta(nu, n):
    beta_0 = np.zeros(n)
    for i in range(len(beta_0)):
        rand = np.random.random_sample()
        if rand < nu:
            beta_0[i] = 1
    return beta_0

@jit(nopython=True)
def norm_pdf(x, loc=0, scale=1):
    return np.exp(-((x-loc)/scale)**2/2)/(np.sqrt(2*np.pi)*scale)

@jit(nopython=True)
def cond_exp_s(arg, s_sqrt, nu):
    #arg = np.float128(arg)
    denomin = nu*norm_pdf(arg - s_sqrt) + (1 - nu)*norm_pdf(arg)
    frac = (1/denomin)*(nu*norm_pdf(arg - s_sqrt))
    #frac = np.float64(frac)
    return frac

@jit(nopython=True)
def mmse_integrand(y, s_sqrt, nu):
    f_y = nu*norm_pdf(y, s_sqrt, 1) + (1-nu)*norm_pdf(y)
    integ = f_y*(cond_exp_s(y, s_sqrt,nu)**2)
    return integ


def mmse_new(s_sqrt, nu):
    if s_sqrt < 25:
        integral = quad(mmse_integrand, -30, 30, args=(s_sqrt,nu))[0]
        mmse = nu - integral
    elif nu < 1:
        integral = quad(mmse_integrand, -10, 10, args =(s_sqrt, nu))[0] + \
        quad(mmse_integrand, s_sqrt - 5, s_sqrt + 5, args =(s_sqrt, nu))[0]
        mmse = nu - integral
    else:
        integral = quad(mmse_integrand, s_sqrt - 5, s_sqrt + 5, args =(s_sqrt, nu))[0]
        mmse = nu - integral
    return mmse


def state_ev_iid_disc(delta, t, nu, alt_init=False):
    if alt_init:
        tau = 0.01
    else:
        tau = np.sqrt(nu/delta) #The 1 here corresponds to the power of x
    tau_array = []
    for _ in range(t):
        tau_array.append(tau)
        tau_prev = tau

        #1e-10 added here to avoid sqrt of neg. value    
        tau = np.sqrt((1/delta)*mmse_new(1/tau, nu)+1e-10)
        #print(tau)
        
        if tau < 1e-50:
            break
        
        if (tau - tau_prev)**2/tau**2 < 1e-12:
            break   
    mse_pred = delta*(tau**2)
    nc_pred = 1 - (mse_pred/nu)
    return tau, mse_pred, nc_pred, tau_array

def noisy_state_ev_iid_disc(delta, t, nu, alpha, sigma):
    # tau initialized as E[X^2]/delta
    sigma2 = ((sigma**2)/(delta*alpha*(1-alpha)))
    tau = np.sqrt(sigma2 + nu/delta) 
    tau_array = []
    for _ in range(t):
        tau_array.append(tau)
        tau_prev = tau

        #1e-10 added here to avoid sqrt of neg. value    
        tau = np.sqrt(sigma2 + (1/delta)*mmse_new(1/tau, nu)+1e-10)
        print(tau)
        
        #Stopping criteria
        if tau < 1e-50:
            break
        
        if (tau - tau_prev)**2/tau**2 < 1e-12:
            break 
        
    mse_pred = delta*(tau**2 - sigma2)
    nc_pred = 1 - (mse_pred/nu)
    return tau, mse_pred, nc_pred, tau_array

def quantize(beta_hat, threshold):
  q_sign = beta_hat - threshold
  beta_q = 0.5*np.sign(q_sign + 1e-10) + 0.5 #Avoid np.sign returning 0
  return beta_q

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

def phi(x):
    #'Cumulative distribution function for the standard normal distribution'
    return (1.0 + math.erf(x / np.sqrt(2.0))) / 2.0
    
def est_error_se_iid(tau, n_samples, nu, thresh):
    beta_0 = create_beta(nu, n_samples)
    tau_G = tau*np.random.randn(n_samples)
    beta_est = beta_0 + tau_G
    beta_q = quantize(beta_est, thresh)
    false_pos, false_neg, err_r = fpr_fnr(beta_q, beta_0)
    return false_pos, false_neg, err_r


#print(state_ev_iid_disc(0.01, 100, 0.3, alt_init=True))



def state_ev_sc_disc(W, delta_hat, it, nu):
    
    R, C = len(W), len(W[0])
    
    #EXACT STATE EVOLUTION
    phi = np.zeros(R)
    psi = np.ones(C)
    p = np.zeros(C)
    phi_prev = np.zeros(R)
    
    phi_array = []
    psi_array = []
    mse_array = []
    
    #start_time = time.time()
    
    for t in range(it):
        #print(t)
        
        #Exact Phi
        for r in range(R):
            #1e-10 added here to avoid sqrt of neg. value
            phi[r]= (1/delta_hat)*np.dot(psi, W[r,:]) + 1e-10
        
        #Exact Psi
        for c in range(C):
            p[c] = np.sqrt(np.dot(W[:,c],(1/phi)))
            psi[c]=mmse_new(p[c], nu)
        
        #print(psi, phi)
        #print(np.linalg.norm(phi - phi_prev, 2)/np.linalg.norm(phi, 2))
        
        phi_array.append(np.zeros(R) + phi)
        psi_array.append(np.zeros(C) + psi)
        mse_array.append((1/C)*np.sum(psi))
        
        if (np.linalg.norm(phi - phi_prev, 2)/np.linalg.norm(phi, 2)) < 1e-6:
            break
        """if np.min(psi) < 1e-5:
            break"""
        
        phi_prev = np.zeros(R) + phi
        
        #print(phi_array)
    #sc_se_time = time.time() - start_time
    p_final = p
    mse_state_ev = (1/C)*np.sum(psi)
    nc_state_ev = 1 - (mse_state_ev/nu)
    
    return phi_array, mse_array, psi_array, mse_state_ev, nc_state_ev, p_final

def noisy_state_ev_sc_disc(W, delta_hat, it, nu, alpha, sigma):
    
    R, C = len(W), len(W[0])
    sigma2 = ((sigma**2)/(delta_hat*alpha*(1-alpha)))
    
    #EXACT STATE EVOLUTION
    phi = np.zeros(R)
    psi = np.ones(C)
    p = np.zeros(C)
    phi_prev = np.zeros(R)
    
    phi_array = []
    psi_array = []
    mse_array = []
    
    #start_time = time.time()
    
    for t in range(it):
        #print(t)
        
        #Exact Phi
        for r in range(R):
            #1e-10 added here to avoid sqrt of neg. value
            phi[r]= sigma2 + (1/delta_hat)*np.dot(psi, W[r,:]) + 1e-10
        
        #Exact Psi
        for c in range(C):
            p[c] = np.sqrt(np.dot(W[:,c],(1/phi)))
            psi[c]=mmse_new(p[c], nu)
        
        #print(psi, phi)
        #print(np.linalg.norm(phi - phi_prev, 2)/np.linalg.norm(phi, 2))
        
        phi_array.append(np.zeros(R) + phi)
        psi_array.append(np.zeros(C) + psi)
        mse_array.append((1/C)*np.sum(psi))
        
        if (np.linalg.norm(phi - phi_prev, 2)/np.linalg.norm(phi, 2)) < 1e-6:
            break
        """if np.min(psi) < 1e-5:
            break"""
        
        phi_prev = np.zeros(R) + phi
        
        #print(phi_array)
    #sc_se_time = time.time() - start_time
    p_final = p
    mse_state_ev = (1/C)*np.sum(psi)
    nc_state_ev = 1 - (mse_state_ev/nu)
    
    return phi_array, mse_array, psi_array, mse_state_ev, nc_state_ev, p_final

def est_error_se_sc(p, N_samples, C, nu, thresh):
    beta_0 = create_beta(nu, N_samples*C)
    beta_est = np.zeros(N_samples*C)
    for c in range(C):
        tau_c_G = (1/p[c])*np.random.randn(N_samples)
        beta_est[N_samples*c: N_samples*(c+1)] = beta_0[N_samples*c: N_samples*(c+1)] + tau_c_G
    beta_q = quantize(beta_est, thresh)
    false_pos, false_neg, err_r = fpr_fnr(beta_q, beta_0)
    return false_pos, false_neg, err_r
    
    
"""
lam = 40
omega = 6
alpha = 0.5
W = amp.create_base_matrix(lam, omega, alpha)
W_tilde = amp.W_to_Wtilde(W, alpha)
phi_array, mse_array, psi_array, mse_state_ev, _, p_final = state_ev_sc_disc(W_tilde, 0.2, 500, 0.2)
print(est_error_se_sc(p_final, 1000, lam, 0.2, 0.5))"""