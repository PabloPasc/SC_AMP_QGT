# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 20:52:54 2021

@author: Pablo
"""

import numpy as np
import scipy.stats, scipy.optimize, scipy.signal
import matplotlib.pyplot as plt
from scipy.integrate import quad
import tikzplotlib


def diff_ent_int(y, s_sqrt, nu):
    
    f_y = (1-nu)*scipy.stats.norm.pdf(y) + nu*scipy.stats.norm.pdf(y, s_sqrt, 1)
    ent = -f_y * np.log(f_y)
    return ent

def mutual_inf_exact(s_sqrt, nu):
    if s_sqrt < 25:
        integ = quad(diff_ent_int, -30, 30, args=(s_sqrt,nu))[0]
    elif nu < 1:
        print("Flag - truncation of bounds")
        integ = quad(diff_ent_int, -10, 10, args =(s_sqrt, nu))[0] + quad(diff_ent_int, s_sqrt - 5, s_sqrt + 5, args =(s_sqrt, nu))[0]
    elif nu == 1:
        integ = quad(diff_ent_int, s_sqrt - 5, s_sqrt + 5, args =(s_sqrt, nu))[0]
    mutu = integ - 0.5*np.log(2*np.pi*np.exp(1))
    #print(integ, 0.5*np.log(2*np.pi*np.exp(1)))
    return mutu

#print(mutual_inf_exact(5, 0.3, 1/np.sqrt(0.3)))


def scalar_pot(x, sigma_2, delta_arg, nu):
    #outp= -(x/(sigma_2 + (x/delta_arg)))+ delta_arg*np.log(1 + (x/(delta_arg*sigma_2))) + \
    #2*mutual_inf_exact(np.sqrt(1/(sigma_2 + (x/delta_arg))), eps, a) #- 2*mutual_inf_exact(np.sqrt(1/sigma_2), eps, a)
    
    outp= -delta_arg*(1 - (sigma_2/(sigma_2 + (x/delta_arg)))) + delta_arg*np.log(1 + (x/(delta_arg*sigma_2))) + \
    2*mutual_inf_exact(np.sqrt(1/(sigma_2 + (x/delta_arg))), nu) - 2*mutual_inf_exact(np.sqrt(1/sigma_2), nu)
        
    return outp



def mse_bound_disc(sigma, delta, x_domain, nu):
    
    #start_time = time.time()
    
    scalar_pot_arr = np.zeros(len(x_domain))
    for i in range(len(x_domain)):
        print(i)
        scalar_pot_arr[i] = scalar_pot(x_domain[i], sigma, delta, nu)
    #plt.plot(x_domain, scalar_pot_arr)
    sp = scipy.signal.argrelmin(scalar_pot_arr)[0]
    sp_pot=[]
    final_value = scalar_pot_arr[-1]
    start_value = scalar_pot_arr[0]
    
    if sp.size != 0:
        largest = x_domain[sp[-1]]
        for it in sp:
            sp_pot.append(scalar_pot_arr[it])
        if np.min(sp_pot) < start_value:
            if np.min(sp_pot)<final_value:
                minimizer = x_domain[sp[np.argmin(sp_pot)]]
            else:
                minimizer = x_domain[-1]
        else:
            if final_value < start_value:
                minimizer = x_domain[-1]
            else:
                minimizer = 0
    else:
        if final_value < start_value:
            minimizer = x_domain[-1]
        else:
            minimizer = 0
        largest = minimizer
    
    
    mse_bound = minimizer#*((lam + omega)/lam)
    iid_bound = largest#*((lam + omega)/lam)
    #print("Minimiser of Potential Method: --- %s seconds ---" % (time.time() - start_time))
    return mse_bound, iid_bound, scalar_pot_arr



sigma_2 = 1e-30
nu = 0.1
delta_array = [0.3, 0.2, 0.1, 0.05, 0.02]#np.arange(0.01,0.1,0.02)

var_x = nu - nu**2

x_domain = np.linspace(0,var_x, 500)

plt.figure()

scalar_pot_arr = np.zeros(len(x_domain))
deriv_arr = np.zeros(len(x_domain))

for delta in delta_array:
    for i in range(len(x_domain)):
        print(i)
        scalar_pot_arr[i] = scalar_pot(x_domain[i], sigma_2, delta, nu)
        #deriv_arr[i] = ((1/delta)/((sigma_2 + x_domain[i]/delta)**2))*(x_domain[i] - mmse_new(np.sqrt(1/(sigma_2 + (x_domain[i]/delta))), eps, a))
    plt.plot(x_domain, scalar_pot_arr, label=r'$\delta=$ {}'.format(delta))
#plt.plot(x_domain, deriv_arr, label=r'$\frac{dU_s}{dx}$')
#plt.title(r'''Comparison of Scalar Potential $U_s$ and is derivative (in terms of mmse), \\
#          for normalized discrete prior, changing $\epsilon$, $\delta=${}, $\sigma^2=${}'''.format(delta, sigma_2))
plt.xlabel(r'$b$')
plt.ylabel(r'Potential $U(b;\delta)$')
plt.legend()
tikzplotlib.save('potential_qgt_nu{}_sigma2{}.tex'.format(nu, sigma_2))

