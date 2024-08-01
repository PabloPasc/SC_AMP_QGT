#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 14:46:18 2021

@author: pp423
"""
import sc_amp_qgt as amp
import sc_se_qgt as se
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


nu = 0.3
sigma = 0

se_delta_array = np.arange(0.05,0.8,0.01)

sc_se_pred = []
iid_se_pred = []
sc_nc_pred_arr = []
iid_nc_pred_arr = []
sc_amp_nc_pred_arr = []
iid_amp_nc_pred_arr = []
#SE - iid + SC
for index_delta in range(len(se_delta_array)):
    delta = se_delta_array[index_delta]
    print("nu = {}, delta = {}".format(nu, delta))
    delta_hat = delta*lam/(lam + omega - 1)
    
    #iid SE
    _, iid_mse_pred, iid_nc_pred, _ = se.noisy_state_ev_iid_disc(delta, it, nu, alpha, sigma)
    #SC SE
    _, _, _, sc_mse_pred, sc_nc_pred, _ = se.noisy_state_ev_sc_disc(W_tilde, delta_hat, it, nu, alpha, sigma)
    

    
    print("MSE Prediction, iid SE: ", iid_mse_pred)
    print("MSE Prediction, SC SE: ", sc_mse_pred)
    
    
    iid_se_pred.append(iid_mse_pred)
    sc_se_pred.append(sc_mse_pred)
    iid_nc_pred_arr.append(iid_nc_pred)
    sc_nc_pred_arr.append(sc_nc_pred)

#AMP - iid + SC
N = 500
p = int(N*lam)
run_no = 1
amp_delta_array_zoom = [0.4, 0.42, 0.44, 0.46, 0.48, 0.5]#np.arange(0.1,0.5,0.1)
amp_delta_array = np.sort(np.concatenate([np.arange(0.1,0.5,0.1), amp_delta_array_zoom]))
nc_array_av = []
nc_array_std = []
sc_nc_array_av = []
sc_nc_array_std = []
mse_array_av = []
mse_array_std = []
sc_mse_array_av = []
sc_mse_array_std = []
lp_nc_array_av = []
lp_nc_array_std = []
lp_mse_array_av = []
lp_mse_array_std = []
sc_lp_nc_array_av = []
sc_lp_nc_array_std = []

true_delta_array = []
for delta in amp_delta_array:
    print("nu = {}, delta = {}".format(nu, delta))
    delta_hat = delta*lam/(lam + omega - 1)
    M = int(delta_hat*N)
    n = int(M*(lam+omega-1))
    
    true_delta_array.append(n/p)
    print("True delta: ", n/p)

for delta in true_delta_array:
    print("nu = {}, delta = {}".format(nu, delta))
    delta_hat = delta*lam/(lam + omega - 1)
    M = int(delta_hat*N)
    #n = int(M*(lam+omega-1))
    n = int(delta*p)
    
    print("True delta: ", n/p)
    
    mse_runs = []
    nc_runs = []
    mse_runs_sc = []
    nc_runs_sc = []
    nc_runs_lp = []
    mse_runs_lp = []
    nc_runs_sc_lp = []
    
    #IID
    for run in range(run_no):
        beta_0 = amp.create_beta(nu, p)
    
        t = 100
    
        alpha = 0.5
    
        print("Run: ", run)
        X = amp.create_X_iid(alpha, n, p)
        X_sc = amp.create_SC_matrix(W, N, alpha, delta_hat)
        #X = amp.binomial(1, alpha, (n,p))
        psi_sc = np.random.normal(0, np.sqrt(N)*sigma, n)
        y = np.dot(X, beta_0)
        y_sc = np.dot(X_sc, beta_0) + psi_sc
        
        #SC LP
        beta_sc_lp = amp.run_LP(n, p, X_sc, y_sc)
        nc_sc_lp = (np.dot(beta_sc_lp, beta_0)/(np.linalg.norm(beta_sc_lp)*np.linalg.norm(beta_0)))**2
        print(nc_sc_lp)
        nc_runs_sc_lp.append(nc_sc_lp)
        
        #LP
        beta_lp = amp.run_LP(n, p, X, y)
        nc_lp = (np.dot(beta_lp, beta_0)/(np.linalg.norm(beta_lp)*np.linalg.norm(beta_0)))**2
        print(nc_lp)
        nc_runs_lp.append(nc_lp)
        mse_lp = (1/p)*(np.linalg.norm(beta_lp - beta_0)**2)
        mse_runs_lp.append(mse_lp)
        
        
        #SC AMP
        X_tilde_sc = amp.Xsc_to_Xsctilde(X_sc, W, alpha)
        
        #Additional C tests to find number of defective items in each column block
        C = lam
        defect_no = np.zeros(C)
        for c in range(C):
            defect_no[c] = np.sum(beta_0[N*c:N*c+N])

        
        y_tilde_sc = amp.y_sc_to_y_sc_tilde(y_sc, alpha, W, nu, omega, n, p, defect_no)
        phi, beta_sc, error_norm_array_sc = amp.sc_amp_bayes(W_tilde, X_tilde_sc, y_tilde_sc, beta_0, nu, delta_hat, t)
        norm_correl_sc = (np.dot(beta_sc, beta_0)/(np.linalg.norm(beta_sc)*np.linalg.norm(beta_0)))**2
        print("SC MSE: ", (1/p)*(np.linalg.norm(beta_sc - beta_0)**2))
        
        
        nc_runs_sc.append(norm_correl_sc)
        mse_runs_sc.append(error_norm_array_sc[-1])
        
        
        
        #iid AMP
        X_iid = amp.create_X_iid(alpha, n, p)
        X_tilde_iid = amp.Xiid_to_Xtilde(X_iid, alpha)
        
        y_iid = np.dot(X_iid, beta_0)
        
        defect_no_iid = np.sum(beta_0)
        y_tilde = amp.y_iid_to_y_iid_tilde(y_iid, alpha, nu, n, p, defect_no_iid)
        X_tilde_iid_T = np.transpose(X_tilde_iid)
        beta, mse_pred, tau_array, error_norm_array, nc_array = amp.amp_bayes(X_tilde_iid, X_tilde_iid_T, y_tilde, t, nu, beta_0)
        norm_correl = (np.dot(beta, beta_0)/(np.linalg.norm(beta)*np.linalg.norm(beta_0)))**2
        
        nc_runs.append(norm_correl)
        mse_runs.append(error_norm_array[-1])
        

    nc_array_av.append(np.average(nc_runs))
    nc_array_std.append(np.std(nc_runs))
    sc_nc_array_av.append(np.average(nc_runs_sc))
    sc_nc_array_std.append(np.std(nc_runs_sc))
    sc_mse_array_av.append(np.average(mse_runs_sc))
    sc_mse_array_std.append(np.std(mse_runs_sc))
    mse_array_av.append(np.average(mse_runs))
    mse_array_std.append(np.std(mse_runs))
    sc_lp_nc_array_av.append(np.average(nc_runs_sc_lp))
    sc_lp_nc_array_std.append(np.std(nc_runs_sc_lp))
    lp_mse_array_av.append(np.average(mse_runs_lp))
    lp_mse_array_std.append(np.std(mse_runs_lp))
    

""""""

        
plt.figure()
plt.plot(se_delta_array, iid_se_pred, label=r'iid SE', color = 'blue', linestyle = 'dashed')
plt.plot(se_delta_array, sc_se_pred, label=r'SC SE', color = 'red', linestyle = 'dashed')
plt.errorbar(amp_delta_array, nc_array_av, yerr=nc_array_std, label =r"iid AMP", fmt='*', color='blue',ecolor='lightblue', elinewidth=3, capsize=0)
plt.errorbar(amp_delta_array, sc_nc_array_av, yerr=sc_nc_array_std, label =r"SC AMP", fmt='*', color='red',ecolor='coral', elinewidth=3, capsize=0)
plt.errorbar(amp_delta_array, mse_array_av, yerr=mse_array_std, label =r"iid AMP", fmt='*', color='blue',ecolor='lightblue', elinewidth=3, capsize=0)
plt.errorbar(amp_delta_array, sc_mse_array_av, yerr=sc_mse_array_std, label =r"SC AMP", fmt='*', color='red',ecolor='coral', elinewidth=3, capsize=0)
plt.errorbar(amp_delta_array, lp_mse_array_av, yerr=lp_mse_array_std, label =r"iid LP", fmt='*', color='green',ecolor='lightgreen', elinewidth=3, capsize=0)
plt.errorbar(amp_delta_array, sc_lp_nc_array_av, yerr=sc_lp_nc_array_std, label =r"SC LP", fmt='*', color='black',ecolor='mistyrose', elinewidth=3, capsize=0)
plt.grid(alpha=0.4)
plt.xlabel(r'$\delta=n/p$')
plt.ylabel('Mean Squared Error')
plt.legend()
#tikzplotlib.save("noiseless_qgt_lp_pi{}_p{}1.tex".format(nu, p))
tikzplotlib.save("noiseless_qgt_mse_lp_lam{}_om{}_nu{}_N{}.tex".format(lam, omega, nu, N))