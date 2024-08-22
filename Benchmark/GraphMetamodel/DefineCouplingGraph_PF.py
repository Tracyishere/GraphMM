'''
    Author: Chenxi Wang (chenxi.wang@salilab.org)
    Date: 2022-03-04

    This script defines the coupling scheme for 2 models.
'''

import numpy as np
from re import L
from scipy.stats import norm
import numpy as np


def compute_overlap_of_normal_dist(m1,m2,std1,std2):

    '''
    A more developed package is provided under python 3.8, fail to include this in python 3.6, yet to test.

    url='https://raw.githubusercontent.com/python/cpython/3.8/Lib/statistics.py' 
    import urllib.request
    import os
    urllib.request.urlretrieve(url, os.path.basename(url)) 
    # urllib.request.urlretrieve(url, 'statistics_basic.py') 
    import statistics_basic as stat
    '''
    
    a = 1/(2*std1**2) - 1/(2*std2**2)
    b = m2/(std2**2) - m1/(std1**2)
    c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
    
    result = np.roots([a,b,c])
    # TBD: check the intersection of the distribution
    
    x1 = np.linspace(m1-3*std1, m1+3*std1, 10000)
    x2 = np.linspace(m2-3*std2, m2+3*std2, 10000)
    lower = min(np.min(x1), np.min(x2))
    upper = max(np.max(x1), np.max(x2))
    
    # 'lower' and 'upper' represent the lower and upper bounds of the space within which we are computing the overlap
    if(len(result)==0): # Completely non-overlapping 
        overlap = 0.0

    elif(len(result)==1): # One point of contact
        r = result[0]
        if(m1>m2):
            tm,ts=m2,std2
            m2,std2=m1,std1
            m1,std1=tm,ts
        if(r<lower): # point of contact is less than the lower boundary. order: r-l-u
            overlap = (norm.cdf(upper,m1,std1)-norm.cdf(lower,m1,std1))
        elif(r<upper): # point of contact is more than the upper boundary. order: l-u-r
            overlap = (norm.cdf(r,m2,std2)-norm.cdf(lower,m2,std2))+(norm.cdf(upper,m1,std1)-norm.cdf(r,m1,std1))
        else: # point of contact is within the upper and lower boundaries. order: l-r-u
            overlap = (norm.cdf(upper,m2,std2)-norm.cdf(lower,m2,std2))

    elif(len(result)==2): # Two points of contact
        r1 = result[0]
        r2 = result[1]
        if(r1>r2):
            temp=r2
            r2=r1
            r1=temp
        if(std1>std2):
            tm,ts=m2,std2
            m2,std2=m1,std1
            m1,std1=tm,ts
        if(r1<lower):
            if(r2<lower):           # order: r1-r2-l-u
                overlap = (norm.cdf(upper,m1,std1)-norm.cdf(lower,m1,std1))
            elif(r2<upper):         # order: r1-l-r2-u
                overlap = (norm.cdf(r2,m2,std2)-norm.cdf(lower,m2,std2))+(norm.cdf(upper,m1,std1)-norm.cdf(r2,m1,std1))
            else:                   # order: r1-l-u-r2
                overlap = (norm.cdf(upper,m2,std2)-norm.cdf(lower,m2,std2))
        elif(r1<upper): 
            if(r2<upper):         # order: l-r1-r2-u
                overlap = (norm.cdf(r1,m1,std1)-norm.cdf(lower,m1,std1))+(norm.cdf(r2,m2,std2)-norm.cdf(r1,m2,std2))+(norm.cdf(upper,m1,std1)-norm.cdf(r2,m1,std1))
            else:                   # order: l-r1-u-r2
                overlap = (norm.cdf(r1,m1,std1)-norm.cdf(lower,m1,std1))+(norm.cdf(upper,m2,std2)-norm.cdf(r1,m2,std2))
        else:                       # l-u-r1-r2
            overlap = (norm.cdf(upper,m1,std1)-norm.cdf(lower,m1,std1))

    return overlap


def compute_overlap_steps(m1_dt,m2_dt,m1_total_time,m2_total_time):

    m1_time_seq = np.array([round(i,5) for i in np.arange(0,m1_total_time,m1_dt,dtype=float)])
    m2_time_seq = np.array([round(i,5) for i in np.arange(0,m2_total_time,m2_dt,dtype=float)])

    # tol = 1e-10 # tolerance
    # overlap_steps = m1_time_seq[(np.abs(m2_time_seq[:,None] - m1_time_seq) < tol).any(0)]
    overlap_steps = set(m1_time_seq).intersection(set(m2_time_seq))

    return m1_time_seq, m2_time_seq, overlap_steps



class coupling_graph:

    def __init__(self, m1, m2, connect_var_m1, connect_var_m2, phi_1=0.5, phi_2=0.5, omega_1=1, omega_2=1, epsilon=0.1):

        self.m1 = m1; self.m2 = m2
        connect_idx_m1 = m1.state.index(connect_var_m1)
        connect_idx_m2 = m2.state.index(connect_var_m2)
        self.connect_var = [connect_var_m1, connect_var_m2]
        self.connect_idx = [connect_idx_m1, connect_idx_m2]
        self.phi_1 = phi_1; self.phi_2 = phi_2
        self.omega_1 = omega_1; self.omega_2 = omega_2
        self.epsilon = epsilon
            

    def get_coupling_graph_multi_scale(self, verbose=1):

        '''
        for models with different time scales, the shape of the coupling graph depends on the step of the models and the ts_scale
        '''

        ################ TBD: deliver the value of n_step using the json? ################
        # count number of overlap steps for the two models
        n_step = len(np.arange(0, max(self.m1.total_time, self.m2.total_time), min(self.m1.dt, self.m2.dt)))
        if self.m1.dt > self.m2.dt:
            ts_scale = round(self.m1.dt/self.m2.dt)
        else:
            ts_scale = round(self.m2.dt/self.m1.dt)

        m1_time_seq, m2_time_seq, overlap_steps = compute_overlap_steps(self.m1.dt, self.m2.dt, self.m1.total_time, self.m2.total_time)
        
        self.overlap_area = [compute_overlap_of_normal_dist(
            self.m1.mean[list(m1_time_seq).index(ts),self.connect_idx[0]], 
            self.m2.mean[list(m2_time_seq).index(ts),self.connect_idx[1]], 
            self.m1.std[list(m1_time_seq).index(ts),self.connect_idx[0]], 
            self.m2.std[list(m2_time_seq).index(ts),self.connect_idx[1]]) for ts in overlap_steps]

        self.overlap_area = np.array(self.overlap_area)

        coupler_step = len(self.overlap_area)

        ################# TBD: check the output of overlap area valid ################

        # this is a herustic function
        self.phi_1 *= np.array([self.m1.Q[i,self.connect_idx[0],self.connect_idx[0]] for i in range(self.m1.n_step)])
        self.phi_2 *= np.array([self.m2.Q[i,self.connect_idx[1],self.connect_idx[1]] for i in range(self.m2.n_step)])
        self.omega_1 *= 1-np.array(self.overlap_area)
        self.omega_2 *= 1-np.array(self.overlap_area)
        self.epsilon = np.array([self.epsilon]*coupler_step).reshape(-1,1)

        coupling_variable = []

        for ts in range(coupler_step): # 0.5 for 2 models by default
            coupling_variable_state_ts = np.array([0.5*self.m1.mean[ts,self.connect_idx[0]]\
                + 0.5*self.m2.mean[ts_scale*ts,self.connect_idx[1]], self.epsilon[ts]], dtype='object')
            coupling_variable += [coupling_variable_state_ts]

        self.coupling_variable = np.array(coupling_variable)

        if verbose==1:

            print('******** Coupling Graph info ********')
            print('model_one_name: {}'.format(self.m1.modelname))
            print('connecting_variable: {}'.format(self.connect_var[0]))
            print('model_two_name: {}'.format(self.m2.modelname))
            print('connecting_variable: {}'.format(self.connect_var[1]))
            print('overlap steps: {}'.format(coupler_step))
            print('******** Coupling Graph info ********')