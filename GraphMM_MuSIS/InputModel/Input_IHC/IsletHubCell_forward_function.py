'''
    Author: Jingjing Zheng(zhengjj1@shanghaitech.edu.cn)
    Date: 2022-07-26
'''

from InputModel.Input_IHC.IsletHubCell_parameter import *
from math import *
import numpy as np
import time
from scipy.integrate import odeint


def _f_input_ode_IHC(x, dt):

    # Declaration of variable
    V = x[:N];  # mV  N=57, initial values of a list of 57 cells
    n = x[N:2*N];  # unitless
    s = x[2*N:3*N];  # unitless
    Ca = x[3*N:4*N];  # uM

    # Calculate expressions
    minf = (1 + np.exp((Vm - V) / theta_m)) ** (-1)
    ninf = (1 + np.exp((Vn - V) / theta_n)) ** (-1)
    sinf = (1 + np.exp((Vs - V) / theta_s)) ** (-1)
    Ica = gca * minf * (V - Vca)
    Ikatp = gkatp * (V - Vk)
    Ik = gk * n * (V - Vk)
    Is = gs * s * (V - Vk)
    Icl = gcl * sigma * (V - Vcl)
    Icoup = np.dot(M, V)

    # RHS
    dVdt = (Ica + Ikatp + Ik + Is + Icl + Icoup)/(-Cm); # mV/s
    dndt = (ninf - n)/tau_n; # 1/s
    dsdt = (sinf - s)/tau_s; # 1/s
    dCadt = f*(-alph*Ica - kca*Ca); # uM/s

    dxdt = list(dVdt) + list(dndt) + list(dsdt) + list(dCadt)

    return np.array(dxdt)


def _fx_IHC(x, dt, time):

    SimTime = np.linspace(time*dt, (time+1)*dt, 2)
    xout = odeint(_f_input_ode_IHC, x, SimTime)
    # xout = odeint(odefunc, x, SimTime)

    return xout[-1]

