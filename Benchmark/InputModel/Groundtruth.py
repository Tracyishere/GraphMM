import scipy.integrate
import numpy as np
import matplotlib.pyplot as plt
from visulize import *


'''
################## Synthetic groundtruth ##################
The glucose stimulated insulin secretion pathway describes the process of insulin secretion under the trigger of glucose. 
We include 4 variables to describe a simplified version of system:

- cumulative glucose concentration from food digestion ($D_t$)
- beta-cell activity ($\gamma_t$)
- insulin secretion amount of an islet ($I_t$)
- plasma glucose concentration ($G_t$)

'''

sim_time_gt = 8 # min
dt_gt = 0.01 # min
time_gt = np.arange(0, sim_time_gt, dt_gt)
n_step_gt = len(time_gt)

D_0, G_0, I_0, gamma_0 = 0.5, 3.2, 2.0, 11

def ODEModel_gt(x, t):

    D, gamma, I, G  = x

    para = [-4.692838342888769, 14.594753296706418, -0.9636023755917396, 4.194832165616945, 
            2.1981069938082958, -4.676025862707219, 24.35454215164307, -1.165628133709453, 
            -0.08397403669598998, 8.6464663285339]

    dD = para[0]*np.log(G) + para[1]
    dgamma = para[2]*D**(1/2) + para[3]
    dI = para[4]*gamma + para[5]*G + para[6]
    dG = para[7]*D**(2/3) + para[8]*I**(2/3) + para[9]
    
    dx = [dD, dgamma, dI, dG]
    
    return np.array(dx)

groundtruth = scipy.integrate.odeint(ODEModel_gt, [D_0, gamma_0, I_0, G_0], time_gt)
groundtruth_std = groundtruth*0.1
model_var_gt = ['D [mM]', 'gamma', '$I_{islet}$ [pg/islet]', 'G [mM]']
# plot_groundtruth(groundtruth, dt=dt_gt, sim_time=sim_time_gt, error=groundtruth_std, name=model_var_gt)