'''
    Author: Chenxi Wang (chenxi.wang@salilab.org)
    Date: 2022-03-04

    This script defines the coupling scheme for 3 models.
'''

import numpy as np
from re import L
from scipy.stats import norm
import numpy as np
from scipy.interpolate import interp1d


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

    m1_time_seq = np.around(np.arange(0,m1_total_time,m1_dt,dtype=float), 5)
    m2_time_seq = np.around(np.arange(0,m2_total_time,m2_dt,dtype=float), 5)
    overlap_steps = set(m1_time_seq).intersection(set(m2_time_seq))

    return m1_time_seq, m2_time_seq, overlap_steps



class coupling_graph:

    def __init__(self, models, connect_var, unit_weights, w_phi=0.5, w_omega=1, w_epsilon=0.1):

        ''' 
        connect_var_m1: list for connecting variables 
        phi: list for parameter phi of each connecting variable


        Example:
            models: {'GI_VE':(surrogate_GI, surrogate_VE), 'VE_Pa':(surrogate_VE, surrogate_Pa)}
            connect_var: {'GI_VE': ('Glu_ec.GI', 'G_ex.VE'), 'VE_Pa': ('ISR.VE', 'S_pa.Pa')}
            connect_idx: {'GI_VE': (0, 9), 'VE_Pa': (8, 2)}
        '''

        self.models = models
        self.connect_var = connect_var
        self.unit_weight = unit_weights
        self.connect_idx = {}
        for var in self.models:
            ma = self.models[var][0]
            mb = self.models[var][1]
            var1 = self.connect_var[var][0]
            var2 = self.connect_var[var][1]
            ma.con_var_idx, ma.con_omega, ma.con_phi, ma.con_unit_weight = [], [], [], []
            mb.con_var_idx, mb.con_omega, mb.con_phi, mb.con_unit_weight = [], [], [], []
            self.connect_idx[var] = (ma.state.index(var1), mb.state.index(var2))
        self.n_coupling_var = len(self.connect_var)

        # self.w_phi = np.array([w_phi]*np.sum([len(item) for item in list(self.connect_var.values())]))
        self.w_phi = [[w_phi, w_phi], [w_phi, w_phi]]
        self.w_omega = np.array([w_omega]*self.n_coupling_var)
        self.w_epsilon = np.array([w_epsilon]*self.n_coupling_var)

        self.model_idx = {}
        for key in self.models.keys():
            ma = key.split('_')[0]
            mb = key.split('_')[1]
            self.model_idx[ma] = self.models[key][0]
            self.model_idx[mb] = self.models[key][1]


    def _pair_connecting_variable(self, ma, mb, unit_weight, epsilon, connect_idx, verbose=1):

        '''
        for models with different time scales, the shape of the coupling graph depends on the step of the models and the ts_scale
        by default, adjust unit of model a, model b with more steps
        return overlap_area, coupling_variable, epsilon for two surrogate model
        '''

        if verbose==1:
            print('===== Sub-coupling graph =====')
            print('model_one_name: {}'.format(ma.modelname))
            print('connecting_variable: {}'.format(ma.state[connect_idx[0]]))
            print('model_two_name: {}'.format(mb.modelname))
            print('connecting_variable: {}'.format(mb.state[connect_idx[1]]))

        # update variable units for computing the overlap area 
        upd_ma_mean = unit_weight[0]*ma.mean[:,connect_idx[0]]
        upd_ma_std = unit_weight[0]*ma.std[:,connect_idx[0]]
        upd_mb_mean = unit_weight[1]*mb.mean[:,connect_idx[1]]
        upd_mb_std = unit_weight[1]*mb.std[:,connect_idx[1]]

        # count number of overlap steps for the two models
        # ts_scale = max(round(ma.dt/mb.dt), round(mb.dt/ma.dt))
        if ma.dt > mb.dt:
            ts_scale = round(ma.dt/mb.dt)
            ma_time_seq, mb_time_seq, overlap_steps = compute_overlap_steps(ma.dt, mb.dt, ma.total_time, mb.total_time)
            
            # not sure about model states or model observations here
            overlap_area = np.array([compute_overlap_of_normal_dist(
                upd_ma_mean[list(ma_time_seq).index(ts)], 
                upd_mb_mean[list(mb_time_seq).index(ts)], 
                upd_ma_std[list(ma_time_seq).index(ts)], 
                upd_mb_std[list(mb_time_seq).index(ts)]) for ts in overlap_steps])

            coupler_step = len(overlap_area)

            # this is a herustic function
            epsilon = np.array([epsilon]*coupler_step).reshape(-1,)

            coupling_variable = []
            for ts in range(coupler_step): # 0.5 for 2 models by default, model b with smaller steps by default
                coupling_variable_state_ts = np.array([0.5*upd_ma_mean[ts] + 0.5*upd_mb_mean[ts_scale*ts], epsilon[ts]])
                coupling_variable += [coupling_variable_state_ts]
            coupling_variable = np.array(coupling_variable)
        
        else:
            ts_scale = round(mb.dt/ma.dt)
            ma_time_seq, mb_time_seq, overlap_steps = compute_overlap_steps(ma.dt, mb.dt, ma.total_time, mb.total_time)

            # not sure about model states or model observations here
            overlap_area = np.array([compute_overlap_of_normal_dist(
                upd_ma_mean[list(ma_time_seq).index(ts)], 
                upd_mb_mean[list(mb_time_seq).index(ts)], 
                upd_ma_std[list(ma_time_seq).index(ts)], 
                upd_mb_std[list(mb_time_seq).index(ts)]) for ts in overlap_steps])

            coupler_step = len(overlap_area)

            # this is a herustic function
            epsilon = np.array([epsilon]*coupler_step).reshape(-1,)

            coupling_variable = []
            for ts in range(coupler_step): # 0.5 for 2 models by default, model b with smaller steps by default
                coupling_variable_state_ts = np.array([0.5*upd_ma_mean[ts_scale*ts] + 0.5*upd_mb_mean[ts], epsilon[ts]])
                coupling_variable += [coupling_variable_state_ts]
            coupling_variable = np.array(coupling_variable)

        if verbose==1:
            print('overlap steps: {}'.format(coupler_step))

        return overlap_area, coupling_variable, epsilon



    def get_coupling_graph_multi_scale(self, verbose=1):

        self._coupling_variable, self._epsilon, self._omega = [], [], []
        self._phi = {}
        
        if verbose==1:
            print('******** Coupling Graph info ********')

        for num in range(self.n_coupling_var):
            overlap_area, coupling_variable, epsilon = self._pair_connecting_variable(list(self.models.values())[num][0], 
                                                                                      list(self.models.values())[num][1],
                                                                                      self.unit_weight[num], 
                                                                                      self.w_epsilon[num],
                                                                                      list(self.connect_idx.values())[num]) 
            self._coupling_variable += [coupling_variable]   
            self._epsilon += [epsilon]  
            # this is a herustic function 
            self._omega += [self.w_omega[num]*(1-overlap_area)]                                                                           
        

        for num, key in enumerate(self.models):
            ma = self.models[key][0]
            mb = self.models[key][1]
            ma_var_idx = self.connect_idx[key][0]
            mb_var_idx = self.connect_idx[key][1]
            phi_v1 = self.w_phi[num][0]*np.array([ma.Q[i, ma_var_idx, ma_var_idx] for i in range(ma.n_step)])
            phi_v2 = self.w_phi[num][1]*np.array([mb.Q[i, mb_var_idx, mb_var_idx] for i in range(mb.n_step)])
            self._phi[key] = (phi_v1, phi_v2)
        
        if verbose==1:
            print('******** Coupling Graph info ********') 


    def _interpolate(self, x, y, xnew):
        
        f = interp1d(x, y, kind='cubic', fill_value="extrapolate")
        ynew = f(xnew)

        return ynew


    def _update_pair_coupling_graph(self, ma, mb, unit_weight, epsilon_ma_mb, connect_idx, p):

        ''' update the coupling graph for multi-scale model inference '''

        # ############# TBD ############# 
        # the updated attribute should not be visited
        # m1 total time larger than m2 total time
        # decide the time interval for model coupling
        if ma.dt > mb.dt:
            ts_scale = round(ma.dt/mb.dt)

            # print(ts_scale)
            upd_epsilon = np.repeat(epsilon_ma_mb, ts_scale, axis=0)
            coupling_graph_ma_obs = np.repeat(ma.obs, ts_scale, axis=0)
            ma_R = np.repeat(ma.R, ts_scale, axis=0)
            mb_R = mb.R

            ma_Q = np.repeat(ma.Q, ts_scale, axis=0)
            mb_Q = mb.Q
            
            # !!!not sure - interpolate using model variable states or model variable observations?
            for i in range(ma.n_state):
                coupling_graph_ma_obs[:,i,0] = self._interpolate(x=np.linspace(0,ma.total_time,num=ma.n_step,endpoint=False), y=ma.obs[:,i,0],
                                                                xnew=np.linspace(0,ma.total_time,num=len(coupling_graph_ma_obs),endpoint=False))
            

            # print(unit_weight[0])
            # print(coupling_graph_ma_obs[:,connect_idx[0],0])

            upd_ma_connect_obs = unit_weight[0]*coupling_graph_ma_obs[:,connect_idx[0],0]
            upd_ma_connect_state_std = unit_weight[0]*np.repeat(ma.std[:,connect_idx[0]], ts_scale, axis=0) 
            # not sure, should be obs std, but too small
            upd_mb_connect_obs = unit_weight[1]*mb.obs[:,connect_idx[1],0]
            upd_mb_connect_state_std = unit_weight[1]*mb.std[:,connect_idx[1]]

            upd_overlap_steps = mb.n_step # not sure
            upd_overlap_area = np.array([compute_overlap_of_normal_dist(
                                upd_ma_connect_obs[ts], 
                                upd_mb_connect_obs[ts], 
                                upd_ma_connect_state_std[ts], 
                                upd_mb_connect_state_std[ts]) for ts in range(upd_overlap_steps)])
            
            # print(upd_ma_connect_obs[:upd_overlap_steps].shape)
            # print(upd_mb_connect_obs.shape)
            # print(upd_epsilon.shape)

            # how to deal with different model length? e.g., 60min VS 420min 
            upd_coupling_variable = np.concatenate(((0.5*upd_ma_connect_obs[:upd_overlap_steps]\
                +0.5*upd_mb_connect_obs).reshape(-1,1), upd_epsilon.reshape(-1,1)), axis=1)

            coupling_graph_mb_obs = mb.obs
        
        else:
            ts_scale = round(mb.dt/ma.dt)

            upd_epsilon = np.repeat(epsilon_ma_mb, ts_scale, axis=0)
            coupling_graph_mb_obs = np.repeat(mb.obs, ts_scale, axis=0)
            ma_R = ma.R
            mb_R = np.repeat(mb.R, ts_scale, axis=0)
            ma_Q = ma.Q
            mb_Q = np.repeat(mb.Q, ts_scale, axis=0)
            
            # !!!not sure - interpolate using model variable states or model variable observations?
            for i in range(mb.n_state):
                coupling_graph_mb_obs[:,i,0] = self._interpolate(x=np.linspace(0,mb.total_time,num=mb.n_step,endpoint=False), y=mb.obs[:,i,0],
                                                                xnew=np.linspace(0,mb.total_time,num=len(coupling_graph_mb_obs),endpoint=False))
            
            upd_ma_connect_obs = unit_weight[0]*ma.obs[:,connect_idx[0],0]
            upd_ma_connect_state_std = unit_weight[0]*ma.std[:,connect_idx[0]]
            upd_mb_connect_obs = unit_weight[1]*coupling_graph_mb_obs[:,connect_idx[1],0]
            upd_mb_connect_state_std = unit_weight[1]*np.repeat(mb.std[:,connect_idx[1]], ts_scale, axis=0) 
            # not sure, should be obs std, but its too small
            
            upd_overlap_steps = ma.n_step # not sure
            upd_overlap_area = np.array([compute_overlap_of_normal_dist(
                                upd_ma_connect_obs[ts], 
                                upd_mb_connect_obs[ts], 
                                upd_ma_connect_state_std[ts], 
                                upd_mb_connect_state_std[ts]) for ts in range(upd_overlap_steps)])
            
            # how to deal with different model length? e.g., 60min VS 420min 
            upd_coupling_variable = np.concatenate(((p*upd_ma_connect_obs\
                +(1-p)*upd_mb_connect_obs[:upd_overlap_steps]).reshape(-1,1), upd_epsilon.reshape(-1,1)), axis=1)

            coupling_graph_ma_obs = ma.obs

        return upd_overlap_area, upd_coupling_variable, coupling_graph_ma_obs, coupling_graph_mb_obs, ma_R, mb_R, ma_Q, mb_Q
        
       

    def get_upd_coupling_graph_multi_scale(self, p):

        self.coupling_variable, self.omega = [], []
        self._upd_obs, self._upd_R, self._upd_Q = {}, {}, {}

        for num,key in enumerate(self.connect_var):
            overlap_area, coupling_variable, ma_obs, mb_obs, ma_R, mb_R, ma_Q, mb_Q = self._update_pair_coupling_graph(list(self.models.values())[num][0], 
                                                                                    list(self.models.values())[num][1],
                                                                                    self.unit_weight[num], 
                                                                                    self._epsilon[num],
                                                                                    list(self.connect_idx.values())[num], p) 
            self.coupling_variable += [coupling_variable]   
            # this is a herustic function 
            self.omega += [self.w_omega[num]*(1-overlap_area)]
            self._upd_obs[key] = [ma_obs, mb_obs]
            self._upd_R[key] = [ma_R, mb_R]
            self._upd_Q[key] = [ma_Q, mb_Q]


        self.model_obs, self.meta_R, self.meta_Q = {}, {}, {}
        for key in self._upd_obs.keys():
            ma = key.split('_')[0]
            mb = key.split('_')[1]
            self.model_obs[ma] = self._upd_obs[key][0]
            self.model_obs[mb] = self._upd_obs[key][1]
            self.meta_R[ma] = self._upd_R[key][0]
            self.meta_R[mb] = self._upd_R[key][1]
            self.meta_Q[ma] = self._upd_Q[key][0]
            self.meta_Q[mb] = self._upd_Q[key][1]

        # temporal easy repeat, can be elaborated by the user, all step phi changes before coupling ends
        self.phi = self._phi.copy() 
        for key in self.phi:
            self.phi[key] = [item for item in self.phi[key]]
            for num in range(len(self.phi[key])):
                if len(self.phi[key][num]) != 6000: # TBD: comparison of minial timestep, cut down before the model ends
                    self.phi[key][num] = np.repeat(self.phi[key][num], round(self.models[key][num].dt/0.01), axis=0)
