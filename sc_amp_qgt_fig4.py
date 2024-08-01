#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 14:56:53 2023

@author: pp423
"""

import numpy as np
import tikzplotlib
import matplotlib.pyplot as plt
from amp_qgt import amp_bayes, create_X_iid, create_beta, Xiid_to_Xtilde, y_iid_to_y_iid_tilde
from amp_qgt import fpr_fnr, g_in_bayes, run_NP
from se_qgt import state_ev_iid_disc, quantize, noisy_state_ev_iid_disc
import sc_amp_qgt as amp
import sc_se_qgt as se


#----------------Figure 4a------------------------------------------------------
thresh_array = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
alpha = 0.5
nu = 0.3
run_no = 10

sigma = 0
delta = 0.38


#Generate omega-lambda base matrix
omega = 6
lam_40 = 40
C_40 = lam_40
R_40 = lam_40 + omega - 1


W_40 = amp.create_base_matrix(lam_40, omega, alpha)
W_tilde_40 = amp.W_to_Wtilde(W_40, alpha)


lam_20 = 20
C_20 = lam_20
R_20 = lam_20 + omega - 1


W_20 = amp.create_base_matrix(lam_20, omega, alpha)
W_tilde_20 = amp.W_to_Wtilde(W_20, alpha)



#State Evolution
fpr_se_iid = np.ones(len(thresh_array))
fnr_se_iid = np.ones(len(thresh_array))
fpr_se_sc_20 = np.ones(len(thresh_array))
fnr_se_sc_20 = np.ones(len(thresh_array))
fpr_se_sc_40 = np.ones(len(thresh_array))
fnr_se_sc_40 = np.ones(len(thresh_array))

n_samples = 1000000
beta_0 = create_beta(nu, n_samples)

#iid SE
tau, mse_pred, _, tau_array_se =  noisy_state_ev_iid_disc(delta, 200, nu, alpha, sigma)

print("MSE SE: ", mse_pred)
tau_G = tau*np.random.randn(n_samples)
beta_est = g_in_bayes(beta_0 + tau_G, tau**2, nu)
for i in range(len(thresh_array)):
    beta_q = quantize(beta_est, thresh_array[i])
    fpr_se_iid[i], fnr_se_iid[i], _ = fpr_fnr(beta_q, beta_0)
    
"""# (6, 20) SC SE
N_samples = 1000000
delta_hat_20 = delta*lam_20/(lam_20 + omega - 1)
_, _, _, sc_mse_pred, sc_nc_pred, p_final = se.state_ev_sc_disc(W_tilde_20, delta_hat_20, 500, nu)
beta_0 = create_beta(nu, N_samples*C_20)
beta_est = np.zeros(N_samples*C_20)
for c in range(C_20):
    tau_c_G = (1/p_final[c])*np.random.randn(N_samples)
    beta_est[N_samples*c: N_samples*(c+1)] = beta_0[N_samples*c: N_samples*(c+1)] + tau_c_G
for i in range(len(thresh_array)):
    beta_q = quantize(beta_est, thresh_array[i])
    fpr_se_sc_20[i], fnr_se_sc_20[i], _ = fpr_fnr(beta_q, beta_0)"""
    
# (6, 40) SC SE
N_samples = 1000000
delta_hat_40 = delta*lam_40/(lam_40 + omega - 1)
_, _, _, sc_mse_pred, sc_nc_pred, p_final = se.noisy_state_ev_sc_disc(W_tilde_40, delta_hat_40, 500, nu, alpha, sigma)
beta_0 = create_beta(nu, N_samples*C_40)
beta_est = np.zeros(N_samples*C_40)
for c in range(C_40):
    tau_c_G = (1/p_final[c])*np.random.randn(N_samples)
    beta_est[N_samples*c: N_samples*(c+1)] = g_in_bayes(beta_0[N_samples*c: N_samples*(c+1)] + tau_c_G, (1/p_final[c])**2, nu)
for i in range(len(thresh_array)):
    beta_q = quantize(beta_est, thresh_array[i])
    fpr_se_sc_40[i], fnr_se_sc_40[i], _ = fpr_fnr(beta_q, beta_0)
    
#AMP
fpr_runs_iid = np.ones((run_no, len(thresh_array)))*5
fnr_runs_iid = np.ones((run_no, len(thresh_array)))*5
fpr_runs_sc_20 = np.ones((run_no, len(thresh_array)))*5
fnr_runs_sc_20 = np.ones((run_no, len(thresh_array)))*5
fpr_runs_sc_40 = np.ones((run_no, len(thresh_array)))*5
fnr_runs_sc_40 = np.ones((run_no, len(thresh_array)))*5
fpr_runs_lp = np.ones((run_no, len(thresh_array)))*5
fnr_runs_lp = np.ones((run_no, len(thresh_array)))*5
fpr_runs_sc_lp = np.ones((run_no, len(thresh_array)))*5
fnr_runs_sc_lp = np.ones((run_no, len(thresh_array)))*5
fpr_runs_sc_lp2 = np.ones((run_no, len(thresh_array)))*5
fnr_runs_sc_lp2 = np.ones((run_no, len(thresh_array)))*5

t = 100

p = 20000
n = int(delta*p)

p_lp = 2000
n_lp = int(delta*p_lp)


N_sc_lp = 50
p_sc_lp = int(N_sc_lp*lam_40)
delta_hat_40 = delta*lam_40/(lam_40 + omega - 1)
M_sc_lp = int(delta_hat_40*N_sc_lp)
n_sc_lp = int(M_sc_lp*(lam_40+omega-1))
print(n_sc_lp/p_sc_lp)



N_40 = 500
p_40 = int(N_40*lam_40)
delta_hat_40 = delta*lam_40/(lam_40 + omega - 1)
M_40 = int(delta_hat_40*N_40)
n_40 = int(M_40*(lam_40+omega-1))
print(n_40/p_40)

N_20 = 1000
p_20 = int(N_20*lam_20)
delta_hat_20 = delta*lam_20/(lam_20 + omega - 1)
M_20 = int(delta_hat_20*N_20)
n_20 = int(M_20*(lam_20+omega-1))
print(n_20/p_20)

for run in range(run_no):
    print("Run: ", run)
    #LP
    X =  create_X_iid(alpha, n_lp, p_lp)
    beta_0 = create_beta(nu, p_lp)
    psi = np.random.normal(0, np.sqrt(p_lp)*sigma, n_lp)
    y = np.dot(X, beta_0) + psi
    
    beta_lp = run_NP(n_lp, p_lp, y, X, sigma, nu)
    
    #Without noise
    #beta_lp = amp.run_LP(n_lp, p_lp, X, y)
    
    for i in range(len(thresh_array)):
        beta_lp_q = quantize(beta_lp, thresh_array[i])
        fpr_runs_lp[run, i], fnr_runs_lp[run, i], _ = fpr_fnr(beta_lp_q, beta_0)
    
    #iid AMP
    X =  create_X_iid(alpha, n, p)
    beta_0 = create_beta(nu, p)
    psi = np.random.normal(0, np.sqrt(p)*sigma, n)
    y = np.dot(X, beta_0) + psi
    
    defect_no = np.sum(beta_0)
    
    X_tilde = Xiid_to_Xtilde(X, alpha)
    X_tilde_T = np.transpose(X_tilde)
    y_tilde = y_iid_to_y_iid_tilde(y, alpha, nu, n, p, defect_no)
    beta, _, tau_array, _, _ = amp_bayes(X_tilde, X_tilde_T, y_tilde, 200, nu, beta_0)
    mse = (1/p)*(np.linalg.norm(beta - beta_0)**2)
    print("MSE AMP: ", mse)
    
    for i in range(len(thresh_array)):
        beta_amp_q = quantize(beta, thresh_array[i])
        fpr_runs_iid[run, i], fnr_runs_iid[run, i], _ = fpr_fnr(beta_amp_q, beta_0)
        
        
    #SC LP, lam=40
    X = amp.create_SC_matrix(W_40, N_sc_lp, alpha, delta_hat_40)
    beta_0 = create_beta(nu, p_sc_lp)
    psi = np.random.normal(0, np.sqrt(0.5*N_sc_lp)*sigma, n_sc_lp)
    y = np.dot(X, beta_0) + psi
    
    beta_sc_lp = amp.run_LP(n_sc_lp, p_sc_lp, X, y)
    for i in range(len(thresh_array)):
        beta_sc_lp_q = quantize(beta_sc_lp, thresh_array[i])
        fpr_runs_sc_lp[run, i], fnr_runs_sc_lp[run, i], _ = fpr_fnr(beta_sc_lp_q, beta_0)
        
    
    """#SC LP/NP, lam=40, regular LP alg
    X = amp.create_SC_matrix(W_40, N_sc_lp, alpha, delta_hat_40)
    beta_0 = create_beta(nu, p_sc_lp)
    psi = np.random.normal(0, np.sqrt(N_sc_lp)*sigma, n_sc_lp)
    y = np.dot(X, beta_0) + psi
    
    beta_sc_lp = run_NP(n_sc_lp, p_sc_lp, y, X, sigma, nu)#amp.run_LP(n_sc_lp, p_sc_lp, X, y)
    for i in range(len(thresh_array)):
        beta_sc_lp_q = quantize(beta_sc_lp, thresh_array[i])
        fpr_runs_sc_lp2[run, i], fnr_runs_sc_lp2[run, i], _ = fpr_fnr(beta_sc_lp_q, beta_0)"""
    
    
    
    
    #SC AMP, lam=40
    X = amp.create_SC_matrix(W_40, N_40, alpha, delta_hat_40)
    beta_0 = create_beta(nu, p_40)
    psi = np.random.normal(0, np.sqrt(0.5*N_40)*sigma, n_40)
    #X = amp.binomial(1, alpha, (n,p))
    y = np.dot(X, beta_0) + psi
    
    X_tilde_sc = amp.Xsc_to_Xsctilde(X, W_40, alpha)
    
    #Additional C tests to find number of defective items in each column block
    defect_no = np.zeros(C_40)
    for c in range(C_40):
        defect_no[c] = np.sum(beta_0[N_40*c:N_40*c+N_40])

    
    y_tilde_sc = amp.y_sc_to_y_sc_tilde(y, alpha, W_40, nu, omega, n_40, p_40, defect_no)
    phi, beta_sc, error_norm_array_sc = amp.sc_amp_bayes(W_tilde_40, X_tilde_sc, y_tilde_sc, beta_0, nu, delta_hat_40, t)
    for i in range(len(thresh_array)):
        beta_amp_q_40 = quantize(beta_sc, thresh_array[i])
        fpr_runs_sc_40[run, i], fnr_runs_sc_40[run, i], _ = fpr_fnr(beta_amp_q_40, beta_0)
        
    """#SC AMP, lam=20
    X = amp.create_SC_matrix(W_20, N_20, alpha, delta_hat_20)
    beta_0 = create_beta(nu, p_20)
    #X = amp.binomial(1, alpha, (n,p))
    y = np.dot(X, beta_0)
    
    X_tilde_sc = amp.Xsc_to_Xsctilde(X, W_20, alpha)
    
    #Additional C tests to find number of defective items in each column block
    defect_no = np.zeros(C_20)
    for c in range(C_20):
        defect_no[c] = np.sum(beta_0[N_20*c:N_20*c+N_20])

    
    y_tilde_sc = amp.y_sc_to_y_sc_tilde(y, alpha, W_20, nu, omega, n_20, p_20, defect_no)
    phi, beta_sc, error_norm_array_sc = amp.sc_amp_bayes(W_tilde_20, X_tilde_sc, y_tilde_sc, beta_0, nu, delta_hat_20, t)
    for i in range(len(thresh_array)):
        beta_amp_q_20 = quantize(beta_sc, thresh_array[i])
        fpr_runs_sc_20[run, i], fnr_runs_sc_20[run, i], _ = fpr_fnr(beta_amp_q_20, beta_0)"""
    

fpr_amp_iid = np.average(fpr_runs_iid, axis = 0)
fnr_amp_iid = np.average(fnr_runs_iid, axis = 0)
fpr_amp_sc_20 = np.average(fpr_runs_sc_20, axis = 0)
fnr_amp_sc_20 = np.average(fnr_runs_sc_20, axis = 0)
fpr_amp_sc_40 = np.average(fpr_runs_sc_40, axis = 0)
fnr_amp_sc_40 = np.average(fnr_runs_sc_40, axis = 0)
fpr_lp = np.average(fpr_runs_lp, axis = 0)
fnr_lp = np.average(fnr_runs_lp, axis = 0)
fpr_sc_lp = np.average(fpr_runs_sc_lp, axis = 0)
fnr_sc_lp = np.average(fnr_runs_sc_lp, axis = 0)
fpr_sc_lp2 = np.average(fpr_runs_sc_lp2, axis = 0)
fnr_sc_lp2 = np.average(fnr_runs_sc_lp2, axis = 0)
 
    
plt.figure()
plt.plot(fnr_se_iid, fpr_se_iid, marker='x', label=r'iid SE', linestyle='dashed', color='blue')
#plt.plot(fnr_se_sc_20, fpr_se_sc_20, marker='x', label=r'SC SE, (6, 20)', linestyle='dashed', color='red')
plt.plot(fnr_se_sc_40, fpr_se_sc_40, marker='x', label=r'SC SE, (6, 40)', linestyle='dashed', color='purple')
plt.plot(fnr_amp_iid, fpr_amp_iid, marker='o', label=r'iid AMP', linestyle='none', color='blue')
#plt.plot(fnr_amp_sc_20, fpr_amp_sc_20, marker='o', label=r'SC AMP, (6, 20)', linestyle='none', color='red')
plt.plot(fnr_amp_sc_40, fpr_amp_sc_40, marker='o', label=r'SC AMP, (6, 40)', linestyle='none', color='purple')
plt.plot(fnr_lp, fpr_lp, marker='o', label=r'iid LP', linestyle='none', color='green')
plt.plot(fnr_sc_lp, fpr_sc_lp, marker='o', label=r'SC LP', linestyle='none', color='black')
#plt.plot(fnr_sc_lp2, fpr_sc_lp2, marker='o', label=r'SC LP', linestyle='none', color='red')
plt.grid(alpha=0.4)
plt.legend()
plt.ylabel('FPR')
plt.xlabel('FNR')
tikzplotlib.save("qgt_fpr_fnr_delta{}_pi{}_sigma{}.tex".format(delta, nu, sigma))



#----------------Figure 4b - with noise------------------------------------------------------
thresh_array = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
alpha = 0.5
nu = 0.3
run_no = 10

sigma = 0.04
delta = 0.46


#Generate omega-lambda base matrix
omega = 6
lam_40 = 40
C_40 = lam_40
R_40 = lam_40 + omega - 1


W_40 = amp.create_base_matrix(lam_40, omega, alpha)
W_tilde_40 = amp.W_to_Wtilde(W_40, alpha)


lam_20 = 20
C_20 = lam_20
R_20 = lam_20 + omega - 1


W_20 = amp.create_base_matrix(lam_20, omega, alpha)
W_tilde_20 = amp.W_to_Wtilde(W_20, alpha)



#State Evolution
fpr_se_iid = np.ones(len(thresh_array))
fnr_se_iid = np.ones(len(thresh_array))
fpr_se_sc_20 = np.ones(len(thresh_array))
fnr_se_sc_20 = np.ones(len(thresh_array))
fpr_se_sc_40 = np.ones(len(thresh_array))
fnr_se_sc_40 = np.ones(len(thresh_array))

n_samples = 1000000
beta_0 = create_beta(nu, n_samples)

#iid SE
tau, mse_pred, _, tau_array_se =  noisy_state_ev_iid_disc(delta, 200, nu, alpha, sigma)

print("MSE SE: ", mse_pred)
tau_G = tau*np.random.randn(n_samples)
beta_est = g_in_bayes(beta_0 + tau_G, tau**2, nu)
for i in range(len(thresh_array)):
    beta_q = quantize(beta_est, thresh_array[i])
    fpr_se_iid[i], fnr_se_iid[i], _ = fpr_fnr(beta_q, beta_0)
    
"""# (6, 20) SC SE
N_samples = 1000000
delta_hat_20 = delta*lam_20/(lam_20 + omega - 1)
_, _, _, sc_mse_pred, sc_nc_pred, p_final = se.state_ev_sc_disc(W_tilde_20, delta_hat_20, 500, nu)
beta_0 = create_beta(nu, N_samples*C_20)
beta_est = np.zeros(N_samples*C_20)
for c in range(C_20):
    tau_c_G = (1/p_final[c])*np.random.randn(N_samples)
    beta_est[N_samples*c: N_samples*(c+1)] = beta_0[N_samples*c: N_samples*(c+1)] + tau_c_G
for i in range(len(thresh_array)):
    beta_q = quantize(beta_est, thresh_array[i])
    fpr_se_sc_20[i], fnr_se_sc_20[i], _ = fpr_fnr(beta_q, beta_0)"""
    
# (6, 40) SC SE
N_samples = 1000000
delta_hat_40 = delta*lam_40/(lam_40 + omega - 1)
_, _, _, sc_mse_pred, sc_nc_pred, p_final = se.noisy_state_ev_sc_disc(W_tilde_40, delta_hat_40, 500, nu, alpha, sigma)
beta_0 = create_beta(nu, N_samples*C_40)
beta_est = np.zeros(N_samples*C_40)
for c in range(C_40):
    tau_c_G = (1/p_final[c])*np.random.randn(N_samples)
    beta_est[N_samples*c: N_samples*(c+1)] = g_in_bayes(beta_0[N_samples*c: N_samples*(c+1)] + tau_c_G, (1/p_final[c])**2, nu)
for i in range(len(thresh_array)):
    beta_q = quantize(beta_est, thresh_array[i])
    fpr_se_sc_40[i], fnr_se_sc_40[i], _ = fpr_fnr(beta_q, beta_0)
    
#AMP
fpr_runs_iid = np.ones((run_no, len(thresh_array)))*5
fnr_runs_iid = np.ones((run_no, len(thresh_array)))*5
fpr_runs_sc_20 = np.ones((run_no, len(thresh_array)))*5
fnr_runs_sc_20 = np.ones((run_no, len(thresh_array)))*5
fpr_runs_sc_40 = np.ones((run_no, len(thresh_array)))*5
fnr_runs_sc_40 = np.ones((run_no, len(thresh_array)))*5
fpr_runs_lp = np.ones((run_no, len(thresh_array)))*5
fnr_runs_lp = np.ones((run_no, len(thresh_array)))*5
fpr_runs_sc_lp = np.ones((run_no, len(thresh_array)))*5
fnr_runs_sc_lp = np.ones((run_no, len(thresh_array)))*5
fpr_runs_sc_lp2 = np.ones((run_no, len(thresh_array)))*5
fnr_runs_sc_lp2 = np.ones((run_no, len(thresh_array)))*5

t = 100

p = 20000
n = int(delta*p)

p_lp = 2000
n_lp = int(delta*p_lp)


N_sc_lp = 50
p_sc_lp = int(N_sc_lp*lam_40)
delta_hat_40 = delta*lam_40/(lam_40 + omega - 1)
M_sc_lp = int(delta_hat_40*N_sc_lp)
n_sc_lp = int(M_sc_lp*(lam_40+omega-1))
print(n_sc_lp/p_sc_lp)



N_40 = 500
p_40 = int(N_40*lam_40)
delta_hat_40 = delta*lam_40/(lam_40 + omega - 1)
M_40 = int(delta_hat_40*N_40)
n_40 = int(M_40*(lam_40+omega-1))
print(n_40/p_40)

N_20 = 1000
p_20 = int(N_20*lam_20)
delta_hat_20 = delta*lam_20/(lam_20 + omega - 1)
M_20 = int(delta_hat_20*N_20)
n_20 = int(M_20*(lam_20+omega-1))
print(n_20/p_20)

for run in range(run_no):
    print("Run: ", run)
    #CVX
    X =  create_X_iid(alpha, n_lp, p_lp)
    beta_0 = create_beta(nu, p_lp)
    psi = np.random.normal(0, np.sqrt(p_lp)*sigma, n_lp)
    y = np.dot(X, beta_0) + psi
    
    beta_lp = run_NP(n_lp, p_lp, y, X, sigma, nu)
    
    
    for i in range(len(thresh_array)):
        beta_lp_q = quantize(beta_lp, thresh_array[i])
        fpr_runs_lp[run, i], fnr_runs_lp[run, i], _ = fpr_fnr(beta_lp_q, beta_0)
    
    #iid AMP
    X =  create_X_iid(alpha, n, p)
    beta_0 = create_beta(nu, p)
    psi = np.random.normal(0, np.sqrt(p)*sigma, n)
    y = np.dot(X, beta_0) + psi
    
    defect_no = np.sum(beta_0)
    
    X_tilde = Xiid_to_Xtilde(X, alpha)
    X_tilde_T = np.transpose(X_tilde)
    y_tilde = y_iid_to_y_iid_tilde(y, alpha, nu, n, p, defect_no)
    beta, _, tau_array, _, _ = amp_bayes(X_tilde, X_tilde_T, y_tilde, 200, nu, beta_0)
    mse = (1/p)*(np.linalg.norm(beta - beta_0)**2)
    print("MSE AMP: ", mse)
    
    for i in range(len(thresh_array)):
        beta_amp_q = quantize(beta, thresh_array[i])
        fpr_runs_iid[run, i], fnr_runs_iid[run, i], _ = fpr_fnr(beta_amp_q, beta_0)
        
        
    #SC CVX, lam=40
    X = amp.create_SC_matrix(W_40, N_sc_lp, alpha, delta_hat_40)
    beta_0 = create_beta(nu, p_sc_lp)
    psi = np.random.normal(0, np.sqrt(0.5*N_sc_lp)*sigma, n_sc_lp)
    y = np.dot(X, beta_0) + psi
    
    beta_sc_lp = amp.run_NP_sc(n_sc_lp, p_sc_lp, y, X, N_sc_lp, sigma, nu)
    for i in range(len(thresh_array)):
        beta_sc_lp_q = quantize(beta_sc_lp, thresh_array[i])
        fpr_runs_sc_lp[run, i], fnr_runs_sc_lp[run, i], _ = fpr_fnr(beta_sc_lp_q, beta_0)
        
    
    """#SC CVX, lam=40, regular LP alg
    X = amp.create_SC_matrix(W_40, N_sc_lp, alpha, delta_hat_40)
    beta_0 = create_beta(nu, p_sc_lp)
    psi = np.random.normal(0, np.sqrt(N_sc_lp)*sigma, n_sc_lp)
    y = np.dot(X, beta_0) + psi
    
    beta_sc_lp = run_NP(n_sc_lp, p_sc_lp, y, X, sigma, nu)#amp.run_LP(n_sc_lp, p_sc_lp, X, y)
    for i in range(len(thresh_array)):
        beta_sc_lp_q = quantize(beta_sc_lp, thresh_array[i])
        fpr_runs_sc_lp2[run, i], fnr_runs_sc_lp2[run, i], _ = fpr_fnr(beta_sc_lp_q, beta_0)"""
    
    
    
    
    #SC AMP, lam=40
    X = amp.create_SC_matrix(W_40, N_40, alpha, delta_hat_40)
    beta_0 = create_beta(nu, p_40)
    psi = np.random.normal(0, np.sqrt(0.5*N_40)*sigma, n_40)
    #X = amp.binomial(1, alpha, (n,p))
    y = np.dot(X, beta_0) + psi
    
    X_tilde_sc = amp.Xsc_to_Xsctilde(X, W_40, alpha)
    
    #Additional C tests to find number of defective items in each column block
    defect_no = np.zeros(C_40)
    for c in range(C_40):
        defect_no[c] = np.sum(beta_0[N_40*c:N_40*c+N_40])

    
    y_tilde_sc = amp.y_sc_to_y_sc_tilde(y, alpha, W_40, nu, omega, n_40, p_40, defect_no)
    phi, beta_sc, error_norm_array_sc = amp.sc_amp_bayes(W_tilde_40, X_tilde_sc, y_tilde_sc, beta_0, nu, delta_hat_40, t)
    for i in range(len(thresh_array)):
        beta_amp_q_40 = quantize(beta_sc, thresh_array[i])
        fpr_runs_sc_40[run, i], fnr_runs_sc_40[run, i], _ = fpr_fnr(beta_amp_q_40, beta_0)
        
    """#SC AMP, lam=20
    X = amp.create_SC_matrix(W_20, N_20, alpha, delta_hat_20)
    beta_0 = create_beta(nu, p_20)
    #X = amp.binomial(1, alpha, (n,p))
    y = np.dot(X, beta_0)
    
    X_tilde_sc = amp.Xsc_to_Xsctilde(X, W_20, alpha)
    
    #Additional C tests to find number of defective items in each column block
    defect_no = np.zeros(C_20)
    for c in range(C_20):
        defect_no[c] = np.sum(beta_0[N_20*c:N_20*c+N_20])

    
    y_tilde_sc = amp.y_sc_to_y_sc_tilde(y, alpha, W_20, nu, omega, n_20, p_20, defect_no)
    phi, beta_sc, error_norm_array_sc = amp.sc_amp_bayes(W_tilde_20, X_tilde_sc, y_tilde_sc, beta_0, nu, delta_hat_20, t)
    for i in range(len(thresh_array)):
        beta_amp_q_20 = quantize(beta_sc, thresh_array[i])
        fpr_runs_sc_20[run, i], fnr_runs_sc_20[run, i], _ = fpr_fnr(beta_amp_q_20, beta_0)"""
    

fpr_amp_iid = np.average(fpr_runs_iid, axis = 0)
fnr_amp_iid = np.average(fnr_runs_iid, axis = 0)
fpr_amp_sc_20 = np.average(fpr_runs_sc_20, axis = 0)
fnr_amp_sc_20 = np.average(fnr_runs_sc_20, axis = 0)
fpr_amp_sc_40 = np.average(fpr_runs_sc_40, axis = 0)
fnr_amp_sc_40 = np.average(fnr_runs_sc_40, axis = 0)
fpr_lp = np.average(fpr_runs_lp, axis = 0)
fnr_lp = np.average(fnr_runs_lp, axis = 0)
fpr_sc_lp = np.average(fpr_runs_sc_lp, axis = 0)
fnr_sc_lp = np.average(fnr_runs_sc_lp, axis = 0)
fpr_sc_lp2 = np.average(fpr_runs_sc_lp2, axis = 0)
fnr_sc_lp2 = np.average(fnr_runs_sc_lp2, axis = 0)

    
    
plt.figure()
plt.plot(fnr_se_iid, fpr_se_iid, marker='x', label=r'iid SE', linestyle='dashed', color='blue')
#plt.plot(fnr_se_sc_20, fpr_se_sc_20, marker='x', label=r'SC SE, (6, 20)', linestyle='dashed', color='red')
plt.plot(fnr_se_sc_40, fpr_se_sc_40, marker='x', label=r'SC SE, (6, 40)', linestyle='dashed', color='purple')
plt.plot(fnr_amp_iid, fpr_amp_iid, marker='o', label=r'iid AMP', linestyle='none', color='blue')
#plt.plot(fnr_amp_sc_20, fpr_amp_sc_20, marker='o', label=r'SC AMP, (6, 20)', linestyle='none', color='red')
plt.plot(fnr_amp_sc_40, fpr_amp_sc_40, marker='o', label=r'SC AMP, (6, 40)', linestyle='none', color='purple')
plt.plot(fnr_lp, fpr_lp, marker='o', label=r'iid CVX', linestyle='none', color='green')
plt.plot(fnr_sc_lp, fpr_sc_lp, marker='o', label=r'SC CVX', linestyle='none', color='black')

plt.grid(alpha=0.4)
plt.legend()
plt.ylabel('FPR')
plt.xlabel('FNR')
tikzplotlib.save("qgt_fpr_fnr_delta{}_pi{}_sigma{}.tex".format(delta, nu, sigma))

