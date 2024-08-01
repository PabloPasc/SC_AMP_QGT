#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 15:56:34 2023

@author: pp423
"""
import math
from numba import jit
import numpy as np
from scipy.integrate import quad

'''=== Create signal vector beta ==='''
def create_beta(nu, n):
    beta_0 = np.zeros(n)
    for i in range(len(beta_0)):
        rand = np.random.random_sample()
        if rand < nu:
            beta_0[i] = 1
    return beta_0

'''=== Compute pdf of a Gaussian ==='''
@jit(nopython=True)
def norm_pdf(x, loc=0, scale=1):
    return np.exp(-((x-loc)/scale)**2/2)/(np.sqrt(2*np.pi)*scale)

'''=== Compute cdf of a Gaussian ==='''
def phi(x):
    #'Cumulative distribution function for the standard normal distribution'
    return (1.0 + math.erf(x / np.sqrt(2.0))) / 2.0

'''=== Compute Bayes-optimal denoiser in terms of snr s ==='''
@jit(nopython=True)
def cond_exp_s(arg, s_sqrt, nu):
    denomin = nu*norm_pdf(arg - s_sqrt) + (1 - nu)*norm_pdf(arg)
    frac = (1/denomin)*(nu*norm_pdf(arg - s_sqrt))
    return frac

@jit(nopython=True)
def mmse_integrand(y, s_sqrt, nu):
    f_y = nu*norm_pdf(y, s_sqrt, 1) + (1-nu)*norm_pdf(y)
    integ = f_y*(cond_exp_s(y, s_sqrt,nu)**2)
    return integ

'''=== Compute MSE of Bayes-optimal denoiser by integrating ==='''
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

'''=== State Evolution for AMP QGT with discrete {0,1} signal prior 
       and Gaussian noise (or noiseless) ==='''
def state_ev_iid_disc(delta, t, nu):
    # tau initialized as E[X^2]/delta
    tau = np.sqrt(nu/delta) 
    tau_array = []
    for _ in range(t):
        tau_array.append(tau)
        tau_prev = tau

        #1e-10 added here to avoid sqrt of neg. value    
        tau = np.sqrt((1/delta)*mmse_new(1/tau, nu)+1e-10)
        print(tau)
        
        #Stopping criteria
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
        
    mse_pred = delta*(tau**2)
    nc_pred = 1 - (mse_pred/nu)
    return tau, mse_pred, nc_pred, tau_array

'''=== Quantize signal entries to 0,1 ==='''
def quantize(beta_hat, threshold):
  q_sign = beta_hat - threshold
  beta_q = 0.5*np.sign(q_sign + 1e-10) + 0.5 #Avoid np.sign returning 0
  return beta_q

'''=== Compute FPR/FNR for a given signal estimate ==='''
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

'''=== Estimate FPR/FNR from SE ==='''
def est_error_se_iid(tau, n_samples, nu, thresh):
    beta_0 = create_beta(nu, n_samples)
    tau_G = tau*np.random.randn(n_samples)
    beta_est = beta_0 + tau_G
    beta_q = quantize(beta_est, thresh)
    false_pos, false_neg, err_r = fpr_fnr(beta_q, beta_0)
    return false_pos, false_neg, err_r
