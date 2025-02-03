'''
    Author: Chenxi Wang (chenxi.wang@salilab.org)
    Date: 2022-03-04

    This script defines the coupling scheme for 3 models.
'''

import numpy as np
from re import L
from scipy.stats import norm
from scipy.interpolate import interp1d
from GraphMetamodel.utils import *
import urllib.request
import os

#url='https://raw.githubusercontent.com/python/cpython/3.8/Lib/statistics.py'
#urllib.request.urlretrieve(url, 'GraphMetamodel/statistics_basic.py') 
import GraphMetamodel.statistics_basic as stat


def compute_overlap_of_normal_dist(m1,m2,std1,std2):

    N1 = stat.NormalDist(m1, std1)
    N2 = stat.NormalDist(m2, std2)
    
    return N1.overlap(N2)



def compute_overlap_steps(m1_dt,m2_dt,m1_total_time,m2_total_time,m1_scale,m2_scale):
    
    # bug: maybe wrong due to the float decimal precision
    m1_time_seq = np.around(np.arange(0,m1_total_time*m1_scale,m1_dt*m1_scale,dtype=float), 5)
    m2_time_seq = np.around(np.arange(0,m2_total_time*m2_scale,m2_dt*m2_scale,dtype=float), 5)
    overlap_steps = set(m1_time_seq).intersection(set(m2_time_seq))

    return m1_time_seq, m2_time_seq, overlap_steps


def cal_product(mean1, mean2, std1, std2, p):

    # p = 0.5
    product_mean = p*mean1 + (1-p)*mean2
    product_var = p*std1**2 + (1-p)*std2**2 + p*(1-p)*(mean1-mean2)**2
        
    return product_mean, product_var



class coupling_graph:


    def __init__(self, models, connect_var, unit_weights, model_states, timescale, w_phi=1, w_omega=1, w_epsilon=1):

        ''' 
        connect_var_m1: list for connecting variables 
        phi: list for parameter phi of each connecting variable
        '''

        self.models = models
        self.connect_var = connect_var
        self.unit_weight = unit_weights
        self.model_states = model_states
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
        self.timescale = timescale

        # self.w_phi = np.array([w_phi]*np.sum([len(item) for item in list(self.connect_var.values())]))
        self.w_phi = [[w_phi, w_phi], [w_phi, w_phi]] # n_coupling_var*2
        self.w_omega = np.array([w_omega]*self.n_coupling_var)
        self.w_epsilon = np.array([w_epsilon]*self.n_coupling_var)

        self.model_idx = {}
        for key in self.models.keys():
            ma = key.split('_')[0]
            mb = key.split('_')[1]
            self.model_idx[ma] = self.models[key][0]
            self.model_idx[mb] = self.models[key][1]
        
        self._check_state_shape(ma)
        self._check_state_shape(mb)
    

    def _check_state_shape(self, mx):

        state_shape = len(self.model_states[mx]) 
        expected_shape = len(np.arange(0, self.model_idx[mx].total_time, self.model_idx[mx].dt))
    
        if state_shape != expected_shape:
            print('ERROR: Surrogate model {} state shape is wrong.'.format(mx))
            print('State length: {}'.format(state_shape))
            print('Model definition expected length: {}'.format(expected_shape))
        else:
            print('Surrogate model {} state shape is correct.'.format(mx))


    def _interpolate(self, x, y, xnew):
        
        f = interp1d(x, y, kind='cubic', fill_value="extrapolate")
        ynew = f(xnew)

        return ynew


    def _pair_coupling_graph(self, ma, mb, unit_weight, epsilon, connect_idx, ma_scale, mb_scale, p, verbose=1):

        ''' update the coupling graph for multi-scale model inference '''


        if verbose==1:
            # TBD: generate a graph using pypgm/daft here
            print('===== Sub-coupling graph =====')
            print('model_one_name: {}'.format(ma.modelname))
            print('connecting_variable: {}'.format(ma.state[connect_idx[0]]))
            print('model_two_name: {}'.format(mb.modelname))
            print('connecting_variable: {}'.format(mb.state[connect_idx[1]]))

        # print(list(self.model_idx.keys())[0])
        ma_mean = self.model_states[list(self.model_idx.keys())[0]][:,:,0]
        ma_std = self.model_states[list(self.model_idx.keys())[0]][:,:,1]
        mb_mean = self.model_states[list(self.model_idx.keys())[1]][:,:,0]
        mb_std = self.model_states[list(self.model_idx.keys())[1]][:,:,1]

        if ma.dt > mb.dt:
            print('run this')

            ts_scale = round(ma.dt/mb.dt)
            _, _, overlap_steps = compute_overlap_steps(ma.dt, mb.dt, ma.total_time, mb.total_time,ma_scale,mb_scale)

            if verbose==1:
                print('overlapped time steps: {}'.format(len(overlap_steps)))

            ###### interpolate the missing states ######
            interp_overlap_steps = int(len(overlap_steps)*ts_scale)
            coupling_graph_ma_interp = np.repeat(ma_mean, ts_scale, axis=0)
            # here, the missing states of the observations is interpolated using the values of state variables instead of the observations
            for i in range(ma.n_state):
                coupling_graph_ma_interp[:,i] = self._interpolate(x=np.linspace(0,ma.total_time,num=ma.n_step,endpoint=False),
                                                               y=ma_mean[:,i],
                                                               xnew=np.linspace(0,ma.total_time,num=len(coupling_graph_ma_interp),endpoint=False))
            # here, the interp_ma_std is changed to a number of ratio
            std_ratio = abs(np.mean(ma_std/ma_mean, axis=0))
            interp_ma_state = coupling_graph_ma_interp
            interp_ma_state_std = abs(coupling_graph_ma_interp*std_ratio)
            interp_all_var_state = np.concatenate((interp_ma_state.reshape(interp_overlap_steps,-1,1), 
                                                   interp_ma_state_std.reshape(interp_overlap_steps,-1,1)), axis=2)
            

            ###### compute the coupling variable ######
            interp_ma_connect_state = unit_weight[0]*interp_ma_state[:,connect_idx[0]]
            interp_ma_connect_state_std = unit_weight[0]*interp_ma_state_std[:, connect_idx[0]]
            mb_connect_state = unit_weight[1]*mb_mean[:,connect_idx[1]]
            mb_connect_state_std = unit_weight[1]*mb_std[:,connect_idx[1]]
            coupling_var_mean, coupling_var_std = [],[]
            for ki in range(interp_overlap_steps):
                pd_mean, pd_var = cal_product(interp_ma_connect_state[ki], mb_connect_state[ki], interp_ma_connect_state_std[ki], mb_connect_state_std[ki], p)
                coupling_var_mean += [pd_mean]
                coupling_var_std += [np.sqrt(pd_var)]
            coupling_var_mean = np.array(coupling_var_mean).reshape(-1,1)
            coupling_var_std = np.array(coupling_var_std).reshape(-1,1)
            coupling_variable_state = np.concatenate((coupling_var_mean, coupling_var_std), axis=1)
            print('coupling variable state: ', coupling_variable_state.shape)

            
        else:
            print('run the other')
            

        return coupling_variable_state, interp_all_var_state
        
       

    def get_coupling_graph_multi_scale(self, p, verbose=1):

        if verbose==1:
            print('******** Coupling Graph info ********')

        self.coupling_variable, self.omega = [], []
        ftemp = open('./results/coupling_graph_param_test.csv','w')

        for num,key in enumerate(self.connect_var):

            coupling_variable, interp_connect_var = self._pair_coupling_graph(list(self.models.values())[num][0], 
                                                                        list(self.models.values())[num][1],
                                                                        self.unit_weight[num], 
                                                                        self.w_epsilon[num],
                                                                        list(self.connect_idx.values())[num],
                                                                        ma_scale=self.timescale[key][0],
                                                                        mb_scale=self.timescale[key][1], p=p) 
            self.coupling_variable += [coupling_variable]   
            connect_var_idx = list(self.connect_idx.values())[num]
            m2_state = self.model_states[list(self.model_idx.keys())[1]]

            for ts in range(len(coupling_variable)):

                # the overlap area is computed using the PDF of surrogate model variable states and the coupling variable
                overlap_m1_c_ts = compute_overlap_of_normal_dist(
                                    self.unit_weight[num][0]*interp_connect_var[ts,connect_var_idx[0],0], 
                                    coupling_variable[ts][0], 
                                    self.unit_weight[num][0]*interp_connect_var[ts,connect_var_idx[0],1], 
                                    coupling_variable[ts][1])
                overlap_m2_c_ts = compute_overlap_of_normal_dist(
                                    self.unit_weight[num][1]*m2_state[ts,connect_var_idx[1],0], 
                                    coupling_variable[ts][0], 
                                    self.unit_weight[num][1]*m2_state[ts,connect_var_idx[1],1], 
                                    coupling_variable[ts][1])
                self.omega += [(overlap_m1_c_ts, overlap_m2_c_ts)]


            self.model_states['a'] = interp_connect_var


        ftemp.close()

        if verbose==1:
            print('\n******** Coupling Graph info ********')                                                                       
