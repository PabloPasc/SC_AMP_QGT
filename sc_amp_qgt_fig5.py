#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 14:46:18 2021

@author: pp423
"""

import sc_amp_pool as amp
from pool_amp import run_LP
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib


def create_A_sc(W,N, delta_hat):
    R, C = np.shape(W)
    M = int(delta_hat*N)
    m, n = M*R, N*C
    A_sc = np.zeros((m,n))
    for c in range(C):
        for r in range(R):
            A_sc[M*r:(M*r+M), N*c:(N*c + N )] = np.random.normal(0, np.sqrt((1/M)*W[r,c]), (M, N))
    return A_sc


def sc_se_no_err(W_tilde, delta_hat, pi, alpha, sigma2, it, num_mc_samples):
    try:
        _, _, _, sc_mse_pred, sc_nc_pred = amp.sc_se_pool(W_tilde, delta_hat, pi, alpha, sigma2, it, num_mc_samples)
    except np.linalg.LinAlgError:
        print("LinALgError, delta = ", delta)
        sc_se_no_err(W_tilde, delta_hat, pi, alpha, sigma2, it, num_mc_samples)
    return sc_mse_pred, sc_nc_pred

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
L = len(pi)
sigma = 1e-2
sigma2 = sigma**2

se_delta_array = np.arange(0.24,0.7,0.01)

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
    
    #iid SE
    _, _, iid_mse_pred, iid_nc_pred = amp.se_pool(delta, alpha, pi, sigma, it, num_mc_samples)
    
    
    #SC SE
    #_, _, _, sc_mse_pred, sc_nc_pred = amp.sc_se_pool(W_tilde, delta_hat, pi, alpha, sigma2, it, num_mc_samples)
    sc_mse_pred, sc_nc_pred = sc_se_no_err(W_tilde, delta_hat, pi, alpha, sigma2, it, num_mc_samples)
        

    
    #print("MSE Prediction, iid SE: ", iid_mse_pred)
    print("MSE Prediction, SC SE: ", sc_mse_pred)
    
    
    iid_se_pred.append(iid_mse_pred[-1])
    sc_se_pred.append(sc_mse_pred[-1])
    iid_nc_pred_arr.append(iid_nc_pred[-1])
    sc_nc_pred_arr.append(sc_nc_pred[-1])

#AMP - iid + SC
N = 50
p = int(N*lam)
sigma = 0
run_no = 10
amp_delta_array_zoom = [0.7]#[0.22, 0.24, 0.26, 0.28, 0.32, 0.34]#np.arange(0.1,0.5,0.1)
amp_delta_array = [0.2, 0.25, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.425, 0.45]#np.sort(np.concatenate([np.arange(0.1,0.5,0.1), amp_delta_array_zoom]))
nc_array_av = []
nc_array_std = []
sc_nc_array_av = []
sc_nc_array_std = []
lp_nc_array_av = []
lp_nc_array_std = []
sc_lp_nc_array_av = []
sc_lp_nc_array_std = []


true_delta_array = []
for delta in amp_delta_array:
    print("nu = {}, delta = {}".format(pi, delta))
    delta_hat = delta*lam/(lam + omega - 1)
    M = int(delta_hat*N)
    n = int(M*(lam+omega-1))
    
    true_delta_array.append(n/p)
    print("True delta: ", n/p)

for delta in true_delta_array:
    print("nu = {}, delta = {}".format(pi, delta))
    delta_hat = delta*lam/(lam + omega - 1)
    M = int(delta_hat*N)
    n = int(M*(lam+omega-1))
    
    
    print("True delta: ", n/p)
    
    mse_runs = []
    nc_runs = []
    mse_runs_sc = []
    nc_runs_sc = []
    nc_runs_sc_lp = []
    nc_runs_lp = []
    
    #IID
    for run in range(run_no):
        
    
        t = 300
    
        alpha = 0.5
    
        B_0 = amp.create_B(pi, p)
        
        print("Run: ", run)
        #AMP - on Bernoulli matrix
        X = np.random.binomial(1, alpha, (n,p))
        Psi = np.random.normal(0, np.sqrt(p)*sigma, (n,L))
        Y = np.dot(X, B_0) + Psi
        
        #iid LP
        B_LP_est = run_LP(n, p, L, Y, X, pi)
        corr_av_lp = np.mean(np.einsum('ij, ij->i', B_LP_est, B_0))
        nc_runs_lp.append(corr_av_lp)
        
        #AMP - on Bernoulli matrix
        sparsity_lvls = np.einsum('i,ij->j', np.ones(p), B_0)
        pi_true = (1/p)*sparsity_lvls
        
        X_tilde = amp.Xiid_to_Xtilde(X, alpha)
        Y_tilde = amp.Y_iid_to_Y_iid_tilde(Y, alpha, n, p, pi_true)
        B, _, mse_arr, noise_arr, mse_final_b = amp.amp_pool(pi, X_tilde, Y_tilde, t, B_0)
        #Round each entry in B to 0 or 1 - if necessary
        #B_q = np.round(B)
        corr_av_b = np.mean(np.einsum('ij, ij->i', B, B_0))
        print(mse_final_b)
        nc_runs.append(corr_av_b)
        
        """
        #SC AMP - on iid Gaussian matrix
        X_sc_G = create_A_sc(W_tilde, N, delta_hat)
        Y_G = np.dot(X_sc_G, B_0) + np.random.normal(0, sigma, (n,L))
        B_sc, _, scalar_mse_arr_sc, mse_final_sc = amp.sc_amp_pool(pi, X_sc_G, M, N, W_tilde, Y_G, B_0, t)
        corr_av_b_sc = np.mean(np.einsum('ij, ij->i', B_sc, B_0))
        #print(mse_final_sc)
        nc_runs_sc.append(corr_av_b_sc)
        """
        
        #SC AMP - on Bernoulli SC matrix
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
        
        
        #start = timeit.default_timer()
        B_sc, _, scalar_mse_arr_sc, mse_final_sc = amp.sc_amp_pool_opt(pi, X_tilde_sc, M, N, W_tilde, Y_tilde_sc, B_0, t)
        #stop = timeit.default_timer()
        #print('Time, new version: ', stop - start) 
        
        #Round each entry in B to 0 or 1
        #B_sc_q = np.round(B_sc)
        corr_av_b_sc = np.mean(np.einsum('ij, ij->i', B_sc, B_0))
        #print(mse_final_sc)
        nc_runs_sc.append(corr_av_b_sc)
        
        
        
        #SC LP - with pi
        B_LP_sc_est = amp.run_LP(n, p, L, Y_sc, X_sc, pi)
        corr_av_lp_sc = np.mean(np.einsum('ij, ij->i', B_LP_sc_est, B_0))
        nc_runs_sc_lp.append(corr_av_lp_sc)

    #nc_array_av.append(np.average(nc_runs))
    #nc_array_std.append(np.std(nc_runs))
    #sc_nc_array_av.append(np.average(nc_runs_sc))
    #sc_nc_array_std.append(np.std(nc_runs_sc))
    lp_nc_array_av.append(np.average(nc_runs_lp))
    lp_nc_array_std.append(np.std(nc_runs_lp))
    sc_lp_nc_array_av.append(np.average(nc_runs_sc_lp))
    sc_lp_nc_array_std.append(np.std(nc_runs_sc_lp))

""""""
#plt.figure()
#plt.plot(sc_nc_pred)

#se_delta_array = [0.1, 0.15, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24]
#sc_nc_pred_arr = np.sort(sc_nc_pred_arr)
        
plt.figure()
#plt.plot(se_delta_array, iid_nc_pred_arr, label=r'iid SE', color = 'blue', linestyle = 'dashed')
#plt.plot(se_delta_array, sc_nc_pred_arr, label=r'SC SE', color = 'red', linestyle = 'dashed')
#plt.errorbar(amp_delta_array, nc_array_av, yerr=nc_array_std, label =r"iid AMP", fmt='*', color='blue',ecolor='lightblue', elinewidth=3, capsize=0)
#plt.errorbar(amp_delta_array, sc_nc_array_av, yerr=sc_nc_array_std, label =r"SC AMP", fmt='*', color='red',ecolor='coral', elinewidth=3, capsize=0)
plt.errorbar(true_delta_array, lp_nc_array_av, yerr=lp_nc_array_std, label =r"iid LP", fmt='*', color='green',ecolor='lightgreen', elinewidth=3, capsize=0)
plt.errorbar(true_delta_array, sc_lp_nc_array_av, yerr=sc_lp_nc_array_std, label =r"SC LP", fmt='*', color='pink',ecolor='mistyrose', elinewidth=3, capsize=0)
plt.grid(alpha=0.4)
plt.xlabel(r'$\delta=n/p$')
plt.ylabel('Correlation')
plt.legend()
#tikzplotlib.save("noiseless_qgt_lp_pi{}_p{}1.tex".format(nu, p))
tikzplotlib.save("lp_pool_lam{}_om{}_pi{}_sigma{}_quantized.tex".format(lam, omega, pi, sigma))