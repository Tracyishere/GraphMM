import math
import scipy.io
import numpy as np

Fig6Para = scipy.io.loadmat('./InputModel/Input_IHC/Fig6Parameters.mat')
hub = 0;

'''STEP 1: Parameter set up '''
SimTime = 300  # simulation time in seconds

gkatp = Fig6Para['gkatp'].reshape(-1, )
gca = Fig6Para['gca'].reshape(-1, )
gk = Fig6Para['gk'].reshape(-1, )
gs = Fig6Para['gs'].reshape(-1, )
kca = Fig6Para['kca'].reshape(-1, )
x0 = Fig6Para['x0'].reshape(-1, )
M = Fig6Para['M']
N = M.shape[0]

''' # parameters-2 '''
Cm = 5.3;
Vm = -20;
theta_m = 12;
Vn = -16;
theta_n = 5.6;
Vs = -52;
theta_s = 10;
Vca = 25;
Vk = -75;
tau_n = 20e-3;
tau_s = 20000e-3;
f = 0.01;
alph = 4.5e-3;
gcl=100;
Vcl=-70;
sigma=np.zeros((N,));

# The number of custom hub cell
if hub != 0: 
    sigma[hub]=1;

''' Initial values '''

def fig6Init():

    return np.array(x0)