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

'''=== Find FPR and FNR for signal estimate ==='''
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
    
'''=== Quantize signal entries to 0, 1 according to threshold ==='''
def quantize_new(beta_hat, thresh):
    beta_hat[beta_hat <= thresh] = 0
    beta_hat[beta_hat > thresh] = 1
    return beta_hat

'''=== Linear Programming algorithm for QGT ==='''
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
  return sol


def run_NP(n, p, y, X, sigma, pi):
    
    one_p = np.ones(p)
    beta_opt = cp.Variable(p)
    constraints = []
    constraints.append(beta_opt <= np.ones(p))
    constraints.append(beta_opt >= np.zeros(p))
    constant = 1/(2*p*(sigma**2))
    const_2 = np.log(1- pi) - np.log(pi)
    objective = cp.Minimize(constant*cp.sum_squares(y - X @ beta_opt) + const_2*cp.scalar_product(beta_opt,one_p))
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP)
    print('optimal obj value:', problem.value)
    beta_est = beta_opt.value
    
    return beta_est


'''=== Create iid design matrix ==='''
def create_X_iid(alpha, m, n):
    X = binomial(1, alpha, (m, n))
    return X

'''=== Create signal vector beta ==='''
def create_beta(nu, n):
    beta_0 = np.zeros(n)
    for i in range(len(beta_0)):
        rand = np.random.random_sample()
        if rand < nu:
            beta_0[i] = 1
    return beta_0

'''=== Transform iid design matrix for AMP ==='''
def Xiid_to_Xtilde(X_iid, alpha):
    m, n = np.shape(X_iid)
    X_tilde = (1/np.sqrt(m*alpha*(1-alpha)))*(X_iid - alpha)
    return X_tilde

'''=== Transform measurements vector y for AMP ==='''
def y_iid_to_y_iid_tilde(y, alpha, nu, m, n, defect_no='None'):
    if defect_no == 'None':
        #Estimate no. defective items - doesn't work well
        y_tilde = (1/np.sqrt(m*alpha*(1-alpha)))*(y - alpha*n*nu)
    else:
        y_tilde = (1/np.sqrt(m*alpha*(1-alpha)))*(y - alpha*defect_no)
    return y_tilde

'''=== Denoising function g_in (aka f_k) ==='''
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

'''=== Derivative of denoising function g_in (aka f_k) ==='''
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

'''=== Bayes-optimal AMP algorithm for QGT ==='''
def amp_bayes(X, X_T, y, t, nu, beta_0):
    m, n = len(X), len(X[0])
    delta_true = m/n
    
    tau_array = []
    error_norm_array = []
    nc_array = []
    
    tau_2 = 100
    #Initialise x
    beta = np.ones(n)*nu   
    for iter_no in range(t):
        print(iter_no)
        if iter_no==0:
            #Initialise x, z and tau
            z = y - np.dot(X, beta)
            tau_prev = 0
        else:
            Onsager = (1/m) * z * np.sum(deriv_g_in_bayes(betaXz, tau_2, nu))
            z             = y - np.dot(X,beta) + Onsager
            tau_prev = np.copy(tau)

        #Estimate noise variance tau from residual z
        tau_2  = (np.linalg.norm(z, ord=2)**2)/ m
        #tau_2 = ((1/delta_true)*se.mmse_new(np.sqrt(1/tau_2), nu))
        print(tau_2)
        
        betaXz = np.dot(X_T, z) + beta
        beta     = g_in_bayes(betaXz, tau_2, nu)

        beta = np.nan_to_num(beta) #Avoid nan
    
        tau = np.sqrt(tau_2)
        tau_array.append(tau)
        
        #Compute performance metrics
        mse = (1/n)*(np.linalg.norm(beta - beta_0)**2)
        norm_correl = (np.dot(beta, beta_0)/(np.linalg.norm(beta)*np.linalg.norm(beta_0)))**2

        error_norm_array.append(mse)
        nc_array.append(norm_correl)
        
        #Stopping criterion - Relative norm tolerance
        if (tau - tau_prev)**2/tau_2 < 1e-9:
            break
        
        tau_prev = tau
        
    mse_pred = delta_true*(tau_2)

    return beta, mse_pred, tau_array, error_norm_array, nc_array
