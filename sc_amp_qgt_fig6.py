#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 14:46:18 2021

@author: pp423
"""

import sc_amp_pool as amp
import sc_amp_qgt as amp_qgt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import tikzplotlib


#Generate omega-lambda base matrix
omega = 6
lam = 40
C = lam
R = lam + omega - 1



alpha = 0.5
W = amp.create_base_matrix(lam, omega, alpha)
W_tilde = amp.W_to_Wtilde(W, alpha)
it = 500
num_mc_samples = 10000

pi = np.array([1/3, 1/3, 1/3])
signal_pwr = np.diag(pi)
L = len(pi)
sigma = 1e-2
sigma2 = sigma**2

se_delta_array = np.arange(0.1, 0.27, 0.01)#[0.1, 0.15]#[0.19, 0.21, 0.22, 0.23, 0.24]#np.arange(0.24,0.7,0.01)

sc_se_pred = []
iid_se_pred = []
sc_nc_pred_arr = []
iid_nc_pred_arr = []
sc_amp_nc_pred_arr = []
iid_amp_nc_pred_arr = []


#SE - iid + SC
for index_delta in range(len(se_delta_array)):
    delta = se_delta_array[index_delta]
    print("pi = {}, delta = {}".format(pi, delta))
    delta_hat = delta*lam/(lam + omega - 1)
    
    #iid SE - quantized
    _, eff_noise_cov_arr, iid_mse_pred, iid_nc_pred = amp.se_pool(delta, alpha, pi, sigma, it, num_mc_samples)
    Sigma = eff_noise_cov_arr[-1] + np.eye(L)*1e-4
    corr = amp.se_mse_mc_final_thresh(pi, Sigma, 1000000)
    
    
    
    #SC SE
    correlation_arr_c = np.zeros(C)
    _, _, P_t, sc_mse_pred, sc_nc_pred = amp.sc_se_pool(W_tilde, delta_hat, pi, alpha, sigma2, it, num_mc_samples)
    #sc_mse_pred, sc_nc_pred = sc_se_no_err(W_tilde, delta_hat, pi, alpha, sigma2, it, num_mc_samples)
    for c in range(C):
        correl = amp.se_mse_mc_final_thresh(pi, P_t[c], num_mc_samples)
        correlation_arr_c[c] = correl

    
    #print("MSE Prediction, iid SE: ", iid_mse_pred)
    print("MSE Prediction, SC SE: ", sc_mse_pred)
    
    sc_nc_quant = np.average(correlation_arr_c)
    
    
    #iid_se_pred.append(iid_mse_pred[-1])
    sc_se_pred.append(sc_mse_pred[-1])
    #iid_nc_pred_arr.append(corr)
    sc_nc_pred_arr.append(sc_nc_quant)

#AMP - iid + SC
N = 500
p = int(N*lam)
sigma = 1e-5
run_no = 1
amp_delta_array_zoom = [0.7]#[0.22, 0.24, 0.26, 0.28, 0.32, 0.34]#np.arange(0.1,0.5,0.1)
amp_delta_array = [0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.425, 0.45]#[0.2, 0.25...]
nc_array_av = []
nc_array_std = []
sc_nc_array_av = []
sc_nc_array_std = []
sc_qgt_nc_array_av = []
sc_qgt_nc_array_std = []
lp_nc_array_av = []
lp_nc_array_std = []

for delta in amp_delta_array:
    print("nu = {}, delta = {}".format(pi, delta))
    delta_hat = delta*lam/(lam + omega - 1)
    M = int(delta_hat*N)
    n = int(M*(lam+omega-1))
    
    
    print("True delta: ", n/p)
    
    mse_runs = []
    nc_runs = []
    mse_runs_sc = []
    nc_runs_sc = []
    nc_runs_sc_qgt = []
    nc_runs_lp = []
    
    #IID
    for run in range(run_no):
        
    
        t = 300
    
        alpha = 0.5
    
        
        
        B_0 = amp.create_B(pi, p)
        
        print("Run: ", run)
        
        X = np.random.binomial(1, alpha, (n,p))
        Psi = np.random.normal(0, np.sqrt(p)*sigma, (n,L))
        Y = np.dot(X, B_0) + Psi
        
        sparsity_lvls = np.einsum('i,ij->j', np.ones(p), B_0)
        pi_true = (1/p)*sparsity_lvls
        
        X_tilde = amp.Xiid_to_Xtilde(X, alpha)
        Y_tilde = amp.Y_iid_to_Y_iid_tilde(Y, alpha, n, p, pi_true)
        B, _, mse_arr, noise_arr, mse_final_b = amp.amp_pool(pi, X_tilde, Y_tilde, t, B_0)
        #Round each entry in B to 0 or 1 - if necessary
        B_q = B#np.round(B)
        corr_av_b = np.mean(np.einsum('ij, ij->i', B_q, B_0))
        print(mse_final_b)
        nc_runs.append(corr_av_b)
        
        
        
        #SC AMP QGT - on Bernoulli SC matrix
        X_sc = amp.create_SC_matrix(W, N, alpha, delta_hat)
        Psi_sc = np.random.normal(0, np.sqrt(p)*sigma, (n,L))
        Y_sc = np.dot(X_sc, B_0) + Psi_sc
        
        #Additional C tests to find number of defective items in each column block
        C = lam
        pi_true = np.zeros((C,L))
        for c in range(C):
            pi_true[c] = (1/N)*np.einsum('i,ij->j', np.ones(N), B_0[N*c:N*c+N])

        
        
        
        X_tilde_sc = amp.Xsc_to_Xsctilde(X_sc, W, alpha)
        Y_tilde_sc = amp.Y_sc_to_Y_sc_tilde(Y_sc, alpha, W, n, p, M, N, pi_true)
        
        #SC AMP -pooled
        B_sc, _, scalar_mse_arr_sc, mse_final_sc = amp.sc_amp_pool(pi, X_tilde_sc, M, N, W_tilde, Y_tilde_sc, B_0, t)
        corr_av_b_sc = np.mean(np.einsum('ij, ij->i', B_sc, B_0))
        #print(mse_final_sc)
        nc_runs_sc.append(corr_av_b_sc)
        
        #Decompose into cols, do SC AMP QGT
        L = len(pi)
        B_sc_qgt = np.zeros((p, L))
        for l in range(L):
            nu = pi[l]
            #Perform QGT on lth column of Y_tilde_sc
            phi, beta_sc_qgt, error_norm_array_sc = amp_qgt.sc_amp_bayes(W_tilde, X_tilde_sc, Y_tilde_sc[:,l], B_0[:, l], nu, delta_hat, t)
            B_sc_qgt[:, l] = beta_sc_qgt
        #Round each entry in B to 0 or 1
        B_correct = np.round(B_sc_qgt)
        
        
        #Alternative method - set largest entry in each row to 1
        #I_L = np.eye(L)
        #max_arr = np.argmax(B_sc_qgt, axis=1)
        #B_correct = np.zeros((p,L))
        #for j in range(p):
        #    B_correct[j] = I_L[max_arr[j]]
        
        corr_av_b_sc_qgt = np.mean(np.einsum('ij, ij->i', B_correct, B_0))
        nc_runs_sc_qgt.append(corr_av_b_sc_qgt)

    nc_array_av.append(np.average(nc_runs))
    nc_array_std.append(np.std(nc_runs))
    sc_nc_array_av.append(np.average(nc_runs_sc))
    sc_nc_array_std.append(np.std(nc_runs_sc))
    sc_qgt_nc_array_av.append(np.average(nc_runs_sc_qgt))
    sc_qgt_nc_array_std.append(np.std(nc_runs_sc_qgt))
    
    


        
plt.figure()
plt.plot(se_delta_array, iid_nc_pred_arr, label=r'iid SE', color = 'blue', linestyle = 'dashed')
plt.plot(se_delta_array, sc_nc_pred_arr, label=r'SC SE', color = 'red', linestyle = 'dashed')
plt.errorbar(amp_delta_array, nc_array_av, yerr=nc_array_std, label =r"mat iid-AMP", fmt='*', color='blue',ecolor='lightblue', elinewidth=3, capsize=0)
plt.errorbar(amp_delta_array, sc_nc_array_av, yerr=sc_nc_array_std, label =r"mat SC-AMP", fmt='*', color='red',ecolor='coral', elinewidth=3, capsize=0)
plt.errorbar(amp_delta_array, sc_qgt_nc_array_av, yerr=sc_qgt_nc_array_std, label =r"col SC-AMP", fmt='*', color='orange',ecolor='coral', elinewidth=3, capsize=0)
plt.grid(alpha=0.4)
plt.xlabel(r'$\delta=n/p$')
plt.ylabel('Correlation')
plt.legend()
#tikzplotlib.save("noiseless_qgt_lp_pi{}_p{}1.tex".format(nu, p))
