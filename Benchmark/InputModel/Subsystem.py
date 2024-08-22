import scipy.integrate
import numpy as np
import matplotlib.pyplot as plt
from visulize import *

D_0, G_0, I_0, gamma_0 = 0.5, 3.2, 2.0, 11


'''
### Submodel a: body model

The body model describes the change of cumulative glucose concentration from food digestion ($D_t^a$) 
as the change of insulin secretion amount of an islet ($I_t^a$) as a function of time. 
'''

def ODEModel_a(x, dt):
    
    ''' defint dx/dt '''

    D_a, I_a = x

    para = [1.47702822040744, -0.6461834269007786, 0.07434489524163206, 
            -2.937179498275664, 22.219341907981303]

    dD_a = para[0]*np.log(I_a) + para[1]
    dI_a = para[2]*D_a**2 + para[3]*D_a + para[4]

    dx = [dD_a, dI_a]
    
    return np.array(dx)

sim_time_a = 8 # min
dt_a = 0.05 # min
time_a = np.arange(0, sim_time_a, dt_a)
n_step_a = len(time_a)
model_var_a = [r'$D^a [mM]$',r'$I_{islet}^a\ [pg/islet]$'] # [pg/islet]
input_a = scipy.integrate.odeint(ODEModel_a, [D_0, I_0], time_a)
input_a_std = input_a*0.15
# plot_inputmodel(input_a, dt=dt_a, sim_time=sim_time_a, name=model_var_a)





'''
### Submodel b: $\beta$-cell model

The $\beta$-cell model describes the change of insulin secretion amount ($I_t^b$) as a function of time, 
$\beta$-cell activity ($\gamma_t$) and the plasma glucose concentration ($G_t^b$). 

'''

def ODEModel_b(x, dt):
    
    gamma_b, I_b, G_b = x

    para = [-0.02764259292818268, -3.2929482000130115, 21.06470961431193, 
            -0.38572678573975416, -4.37583184690527, 53.88775990530528, 
            0.001192339928239537, -0.14355227056573305]

    dG_b = para[0]*I_b + para[1]*gamma_b**(2/3) + para[2]
    dI_b = para[3]*gamma_b + para[4]*G_b + para[5]
    dgamma_b = para[6]*I_b**2 + para[7]

    dx = [dgamma_b, dI_b, dG_b]
    
    return np.array(dx)

sim_time_b = 8 # min
dt_b = 0.01 # min
time_b = np.arange(0, sim_time_b, dt_b)
n_step_b = len(time_b)
model_var_b = [r'$\gamma^b$', r'$I_{cell}^b [pg/islet]$', r'$G^b [mM]$']
input_b = scipy.integrate.odeint(ODEModel_b, [gamma_0, I_0, G_0], time_b)
input_b_std = input_b*0.15
# plot_inputmodel(input_b, dt=dt_b, sim_time=sim_time_b, name=model_var_b)