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
from concurrent.futures import ThreadPoolExecutor


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


def compute_line_num(file_name):
    
    '''
    get the numer of lines in a file, minus 1 if file with a header
    
    '''

    from itertools import (takewhile, repeat)
    buffer = 1024 * 1024
    with open(file_name) as f:
        buf_gen = takewhile(lambda x: x, (f.read(buffer) for _ in repeat(None)))
        return sum(buf.count('\n') for buf in buf_gen) - 1


def cal_product(mean1, mean2, std1, std2):

    p = 0.5
    product_mean = p*mean1 + (1-p)*mean2
    product_var = p*std1**2 + (1-p)*std2**2 + p*(1-p)*(mean1-mean2)**2
        
    return product_mean, product_var



class coupling_graph:

    
    def __init__(self, models, connect_var, model_states_file, unit_weights, timescale, 
                       batch=10000, w_phi=1, w_omega=1, w_epsilon=1):

        ''' 
        connect_var_m1: list for connecting variables 
        phi: list for parameter phi of each connecting variable


        Example:
            
        '''

        self.models = models
        self.connect_var = connect_var
        self.unit_weight = unit_weights
        self.batchsize = batch
        self.connect_idx = {}
        self.state_length = {}
        self.n_coupling_var = len(self.connect_var)
        self.timescale = timescale
        self.model_states_file = model_states_file
        self.w_phi = [[w_phi, w_phi]]*self.n_coupling_var
        self.w_omega = np.array([w_omega]*self.n_coupling_var)
        self.w_epsilon = np.array([w_epsilon]*self.n_coupling_var)
        self.n_batch = {}
        self.model_idx = {}
        self.coupling_variable = {}
        self.omega = {}

        for pair_c in self.models.keys():
            self.state_length[pair_c] = {}
            ma, mb = list(self.models[pair_c].values())
            ma_idx, mb_idx = pair_c.split('_')
            var1, var2 = self.connect_var[pair_c]
            self.model_idx[ma_idx] = ma
            self.model_idx[mb_idx] = mb
            self.connect_idx[pair_c] = (ma.state.index(var1), mb.state.index(var2))
            if self.model_states_file is not None:
                print(self.model_states_file[pair_c][ma_idx])
                print(self.model_states_file[pair_c][mb_idx])
                self.state_length[pair_c][ma_idx] = compute_line_num(self.model_states_file[pair_c][ma_idx])
                self.state_length[pair_c][mb_idx] = compute_line_num(self.model_states_file[pair_c][mb_idx])
                self._check_state_shape(ma_idx, pair_c)
                self._check_state_shape(mb_idx, pair_c)
                self.n_batch[pair_c] = [int(l / self.batchsize) for l in list(self.state_length[pair_c].values())]
            

    def _check_state_shape(self, m_name, pair_c):

        state_shape = self.state_length[pair_c][m_name]
        expected_shape = len(np.arange(0, self.models[pair_c][m_name].total_time, self.models[pair_c][m_name].dt))

        if abs(state_shape - expected_shape) > 1:
            print(abs(state_shape - expected_shape))
            print(f'State length: {state_shape}')
            print(f'Model definition expected length: {expected_shape}')
            raise ValueError(f'ERROR: Surrogate model {m_name} state shape is wrong.')
        else:
            print(f'Surrogate model {m_name} state shape is correct.')


    def _interpolate(self, x, y, xnew):
        
        f = interp1d(x, y, kind='cubic', fill_value="extrapolate")
        ynew = f(xnew)

        return ynew
    

    def _pair_coupling_graph(self, pair_c, ma, mb, unit_weight, epsilon, connect_idx, ma_scale, mb_scale, verbose=1):

        ''' update the coupling graph for multi-scale model inference '''


        if verbose==1:
            print('===== Sub-coupling graph =====')
            print('model_one_name: {}'.format(ma.modelname))
            print('connecting_variable: {}'.format(ma.state[connect_idx[0]]))
            print('model_two_name: {}'.format(mb.modelname))
            print('connecting_variable: {}'.format(mb.state[connect_idx[1]]))
            print('===== Sub-coupling graph =====')

        ma_idx, mb_idx = list(self.models[pair_c].keys())
        ma_mean = self.model_states[pair_c][ma_idx][:,:,0]
        ma_std = self.model_states[pair_c][ma_idx][:,:,1]
        mb_mean = self.model_states[pair_c][mb_idx][:,:,0]
        mb_std = self.model_states[pair_c][mb_idx][:,:,1]

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
            print(ma.n_step)
            print(len(ma_mean))
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
            # print('this should be (2000000,9,2) ', interp_all_var_state.shape)
            

            ###### compute the coupling variable ######
            interp_ma_connect_state = unit_weight[0]*interp_ma_state[:,connect_idx[0]]
            interp_ma_connect_state_std = unit_weight[0]*interp_ma_state_std[:, connect_idx[0]]
            mb_connect_state = unit_weight[1]*mb_mean[:,connect_idx[1]]
            mb_connect_state_std = unit_weight[1]*mb_std[:,connect_idx[1]]
            coupling_var_mean, coupling_var_std = [],[]
            for ki in range(interp_overlap_steps):
                pd_mean, pd_var = cal_product(interp_ma_connect_state[ki], mb_connect_state[ki], interp_ma_connect_state_std[ki], mb_connect_state_std[ki])
                coupling_var_mean += [pd_mean]
                coupling_var_std += [np.sqrt(pd_var)]
            coupling_var_mean = np.array(coupling_var_mean).reshape(-1,1)
            coupling_var_std = np.array(coupling_var_std).reshape(-1,1)
            coupling_variable_state = np.concatenate((coupling_var_mean, coupling_var_std), axis=1)
            print('coupling variable state: ', coupling_variable_state.shape)

            
        else:
            print('run the other')
            

        return coupling_variable_state, interp_all_var_state


    def _pair_coupling_graph_batch(self, nb, total_batch, pair_c, ma, mb, unit_weight, epsilon, connect_idx, ma_scale, mb_scale, verbose=0):

        ''' update the coupling graph for multi-scale model inference '''

        if verbose==1:
            print('===== Sub-coupling graph =====')
            print('model_one_name: {}'.format(ma.modelname))
            print('connecting_variable: {}'.format(ma.state[connect_idx[0]]))
            print('model_two_name: {}'.format(mb.modelname))
            print('connecting_variable: {}'.format(mb.state[connect_idx[1]]))
            print('===== Sub-coupling graph =====')
            
        ma_idx, mb_idx = list(self.models[pair_c].keys())
        ma_mean = self.batch_model_states[pair_c][ma_idx][:,:,0]
        ma_std = self.batch_model_states[pair_c][ma_idx][:,:,1]
        mb_mean = self.batch_model_states[pair_c][mb_idx][:,:,0]
        mb_std = self.batch_model_states[pair_c][mb_idx][:,:,1]
        
        if ma.dt*ma_scale < mb.dt*mb_scale:
            mtemp = ma
            ma = mb
            mb = mtemp
        else:
            pass

        ma_batch_time = ma.dt*len(ma_mean)
        mb_batch_time = mb.dt*len(mb_mean)
        # print(ma_batch_time)
        # print(mb_batch_time)
        ts_scale = round(ma.dt/mb.dt)
        _, _, overlap_steps = compute_overlap_steps(ma.dt, mb.dt, ma_batch_time, mb_batch_time, ma_scale, mb_scale)
        print('batch {}/{} overlap steps: {}'.format(nb, total_batch, len(overlap_steps)))

        ###### interpolate the missing states ######
        # interp_overlap_steps = int(len(overlap_steps)*ts_scale)
        # coupling_graph_ma_interp = np.repeat(ma_mean, ts_scale, axis=0)
        interp_overlap_steps = len(mb_mean)
        coupling_graph_ma_interp = np.ones((len(mb_mean),ma.n_state)) # this is for non-multipling steps?

        # here, the missing states of the observations is interpolated using the values of state variables instead of the observations
        for i in range(ma.n_state):
            coupling_graph_ma_interp[:,i] = self._interpolate(x=np.linspace(0,ma_batch_time,num=len(ma_mean),endpoint=False),
                                                           y=ma_mean[:,i],
                                                           xnew=np.linspace(0,ma_batch_time,num=len(mb_mean),endpoint=False))

        # print('interp_overlap_steps: ', interp_overlap_steps)
        # print('coupling_graph_ma_interp: ', coupling_graph_ma_interp.shape)
        # print('ma_mean: ', len(ma_mean))
        # print('mb_mean: ', len(mb_mean))

        # here, the interp_ma_std is changed to a number of ratio
        std_ratio = abs(np.mean(ma_std/ma_mean, axis=0))
        interp_ma_state = coupling_graph_ma_interp
        interp_ma_state_std = abs(coupling_graph_ma_interp*std_ratio)
        interp_all_var_state = np.concatenate((interp_ma_state.reshape(interp_overlap_steps,-1,1), 
                                                interp_ma_state_std.reshape(interp_overlap_steps,-1,1)), axis=2)
        

        ###### compute the coupling variable ######
        # print('This should be [172, 2]:' connect_idx) # [172, 2]
        interp_ma_connect_state = unit_weight[0]*interp_ma_state[:,connect_idx[0]]
        interp_ma_connect_state_std = unit_weight[0]*interp_ma_state_std[:, connect_idx[0]]
        mb_connect_state = unit_weight[1]*mb_mean[:,connect_idx[1]]
        mb_connect_state_std = unit_weight[1]*mb_std[:,connect_idx[1]]
        coupling_var_mean, coupling_var_std = [],[]
        for ki in range(interp_overlap_steps):
            pd_mean, pd_var = cal_product(interp_ma_connect_state[ki], mb_connect_state[ki], interp_ma_connect_state_std[ki], mb_connect_state_std[ki])
            coupling_var_mean += [pd_mean]
            coupling_var_std += [np.sqrt(pd_var)]
        coupling_var_mean = np.array(coupling_var_mean).reshape(-1,1)
        coupling_var_std = np.array(coupling_var_std).reshape(-1,1)
        coupling_variable_state = np.concatenate((coupling_var_mean, coupling_var_std), axis=1)
        # print(coupling_var_mean)
        # print('coupling variable state: ', coupling_variable_state.shape)


        # ###### compute omega ######
        # # the overlap area is computed using the PDF of surrogate model variable states and the coupling variable
        # overlap_m1_c_ts = compute_overlap_of_normal_dist(
        #                     self.unit_weight[num][0]*interp_connect_var[ts,connect_var_idx[0],0], 
        #                     coupling_variable[ts][0], 
        #                     self.unit_weight[num][0]*interp_connect_var[ts,connect_var_idx[0],1], 
        #                     coupling_variable[ts][1])
        # overlap_m2_c_ts = compute_overlap_of_normal_dist(
        #                     self.unit_weight[num][1]*m2_state[ts,connect_var_idx[1],0], 
        #                     coupling_variable[ts][0], 
        #                     self.unit_weight[num][1]*m2_state[ts,connect_var_idx[1],1], 
        #                     coupling_variable[ts][1])

        return coupling_variable_state, interp_all_var_state
    
        
    def _get_coupling_graph_multi_scale_VE_ISK(self, filepath, num, pair_c, verbose):

        self.coupling_variable[pair_c], self.omega[pair_c] = [], []
        f_coupling_param = open(filepath,'w')

        self.model_states = {}
        self.model_states[pair_c] = {}
        ma_idx, mb_idx = pair_c.split('_')
        ma_scale, mb_sacle = self.timescale[pair_c]
        print(ma_scale, mb_sacle)
        
        if verbose==1:
            print('======= Reading model state files =======')

        self.model_states[pair_c][ma_idx] = np.genfromtxt(self.model_states_file[pair_c][ma_idx], 
                                            delimiter=',', 
                                            skip_header=1, 
                                            max_rows=self.state_length[pair_c][ma_idx]).reshape(self.state_length[pair_c][ma_idx], -1, 2)
        
        self.model_states[pair_c][mb_idx] = np.genfromtxt(self.model_states_file[pair_c][mb_idx], 
                                            delimiter=',', 
                                            skip_header=1, 
                                            max_rows=self.state_length[pair_c][mb_idx]).reshape(self.state_length[pair_c][mb_idx], -1, 2)
        if verbose==1:
            print('======= DONE - read model state files =======')

        ma, mb = list(self.models[pair_c].values())
        coupling_variable, interp_connect_var = self._pair_coupling_graph(
            pair_c=pair_c, ma=ma, mb=mb,
            unit_weight=self.unit_weight[pair_c], 
            epsilon=self.w_epsilon[num],
            connect_idx=list(self.connect_idx.values())[num],
            ma_scale=ma_scale, 
            mb_scale=mb_sacle)
        
        connect_var_idx = list(self.connect_idx.values())[num]
        # print('this should be [2, 10]: ', connect_var_idx)
        m2_state = self.model_states[pair_c][mb_idx]

        for ts in range(len(coupling_variable)):

            # the overlap area is computed using the PDF of surrogate model variable states and the coupling variable
            overlap_m1_c_ts = compute_overlap_of_normal_dist(
                                self.unit_weight[pair_c][0]*interp_connect_var[ts,connect_var_idx[0],0], 
                                coupling_variable[ts][0], 
                                self.unit_weight[pair_c][0]*interp_connect_var[ts,connect_var_idx[0],1], 
                                coupling_variable[ts][1])
            overlap_m2_c_ts = compute_overlap_of_normal_dist(
                                self.unit_weight[pair_c][1]*m2_state[ts,connect_var_idx[1],0], 
                                coupling_variable[ts][0], 
                                self.unit_weight[pair_c][1]*m2_state[ts,connect_var_idx[1],1], 
                                coupling_variable[ts][1])
            self.omega[pair_c] += [(overlap_m1_c_ts, overlap_m2_c_ts)]


            self.model_states[ma_idx] = interp_connect_var # for VE only

            param = [coupling_variable[ts][0], coupling_variable[ts][1], overlap_m1_c_ts, overlap_m2_c_ts]
            print(*param, file=f_coupling_param, sep=',')
        
        f_coupling_param.close()
        
        self.omega[pair_c] = np.array(self.omega[pair_c])
        self.coupling_variable[pair_c] = np.array(coupling_variable)


    def _get_coupling_graph_multi_scale_IHC_ISK(self, filepath, temp_path, num, pair_c, input_file_num, verbose=0):

        f_coupling_param2 = open(filepath,'w')


        ma, mb = list(self.models[pair_c].values())
        ma_scale, mb_sacle = self.timescale[pair_c]
        ma_idx, mb_idx = pair_c.split('_')
        connect_idx = list(self.connect_idx.values())[num]
        self.omega[pair_c], self.coupling_variable[pair_c] = [], []
        
        if verbose==1:
            print('===== Sub-coupling graph =====')
            print('model_one_name: {}'.format(ma.modelname))
            print('connecting_variable: {}'.format(ma.state[connect_idx[0]]))
            print('model_two_name: {}'.format(mb.modelname))
            print('connecting_variable: {}'.format(mb.state[connect_idx[1]]))
            print('Number of batch: {}'.format(max(self.n_batch[pair_c])))


        for nb in range(max(self.n_batch[pair_c])): # the batch size should be decided use model dt

            self.batch_model_states = {}
            self.batch_model_states[pair_c] = {}

            ma_batchsize = round(self.batchsize*mb.dt*mb_sacle / (ma.dt*ma_scale)) # make sure its integar
            ma_start_line = 1 + int(nb*ma_batchsize) # we know ma_dt is larger
            mb_start_line = 1 + int(nb*self.batchsize)

            if input_file_num == 1:
                self.batch_model_states[pair_c][ma_idx] = np.genfromtxt(self.model_states_file[pair_c][ma_idx], 
                                            delimiter=',', 
                                            skip_header=ma_start_line, 
                                            max_rows=ma_batchsize).reshape(ma_batchsize, -1, 2)
            else:
                temp0 = np.genfromtxt(self.model_states_file[pair_c][ma_idx], 
                                        delimiter=',', 
                                        skip_header=ma_start_line, 
                                        max_rows=ma_batchsize).reshape(ma_batchsize, -1, 2) # (10000, 4, 2)
                for nf in range(1, input_file_num):
                    file_path = f"{self.model_states_file[pair_c][ma_idx].rsplit('_', 1)[0]}_{nf}.csv"
                    temp1 = np.genfromtxt(file_path, delimiter=',', 
                                            skip_header=ma_start_line, 
                                            max_rows=ma_batchsize).reshape(ma_batchsize, -1, 2)
                    # temp0 = np.concatenate((temp0, temp1), axis=1)
                    n_col = nf
                    temp2 = np.concatenate((temp0[:,:n_col,:].reshape(ma_batchsize,n_col,-1), temp1[:,0,:].reshape(ma_batchsize,1,-1)), axis=1)
                    for i in range(1, 4): # IHC only
                        temp3 = np.concatenate((temp0[:,i*n_col:(i+1)*n_col,:].reshape(ma_batchsize,n_col,-1), 
                                                temp1[:,i,:].reshape(ma_batchsize,1,-1)), axis=1)
                        temp2 = np.concatenate((temp2, temp3), axis=1)
                    temp0 = temp2
                    
            self.batch_model_states[pair_c][ma_idx] = temp0
            # print(self.batch_model_states[pair_c][ma_idx].shape)
            self.batch_model_states[pair_c][mb_idx] = np.genfromtxt(self.model_states_file[pair_c][mb_idx], 
                                        delimiter=',', 
                                        skip_header=mb_start_line, 
                                        max_rows=self.batchsize).reshape(self.batchsize, -1, 2)
            
            coupling_variable, interp_connect_var = self._pair_coupling_graph_batch(nb, max(self.n_batch[pair_c]), pair_c, ma, mb,
                                                                        self.unit_weight[pair_c], 
                                                                        self.w_epsilon[num],
                                                                        connect_idx,
                                                                        ma_scale, mb_sacle, verbose)                 
            
            ''' save overlap_area, omega, coupling_var_state, upd_var_obs'''

            # self.coupling_variable += [coupling_variable]   
            connect_var_idx = list(self.connect_idx.values())[num]
            # print('this should be [172, 2]: ', connect_var_idx)
            m2_state = self.batch_model_states[pair_c][mb_idx] # here, m2 is ISK

            # print(interp_connect_var.shape)
            # print(m2_state.shape)

            for ts in range(len(coupling_variable)):

                # the overlap area is computed using the PDF of surrogate model variable states and the coupling variable
                overlap_m1_c_ts = compute_overlap_of_normal_dist(
                                    self.unit_weight[pair_c][0]*interp_connect_var[ts,connect_var_idx[0],0], 
                                    coupling_variable[ts][0], 
                                    self.unit_weight[pair_c][0]*interp_connect_var[ts,connect_var_idx[0],1], 
                                    coupling_variable[ts][1])
                overlap_m2_c_ts = compute_overlap_of_normal_dist(
                                    self.unit_weight[pair_c][1]*m2_state[ts,connect_var_idx[1],0], 
                                    coupling_variable[ts][0], 
                                    self.unit_weight[pair_c][1]*m2_state[ts,connect_var_idx[1],1], 
                                    coupling_variable[ts][1])
                self.omega[pair_c] += [(overlap_m1_c_ts, overlap_m2_c_ts)]
                self.coupling_variable[pair_c] += [(coupling_variable[ts][0], coupling_variable[ts][1])]

                param = [coupling_variable[ts][0], coupling_variable[ts][1], overlap_m1_c_ts, overlap_m2_c_ts]
                print(*param, file=f_coupling_param2, sep=',')

            #     for nc in range(57):
            #         # write the 
            #         exec("output_{} = open('{}/IHC_cell_{}.csv', 'a')".format(nc,temp_path,nc))
            #         cell_i = [*interp_connect_var[ts][nc], *interp_connect_var[ts][nc+57], *interp_connect_var[ts][nc+57*2], *interp_connect_var[ts][nc+57*3]]
            #         # print(*cell_i)
            #         exec("print(*cell_i, file=output_{}, sep=',')".format(nc))

            # for nc in range(57):
            #     exec("output_{}.close()".format(nc))                                                                   

        f_coupling_param2.close()

        self.omega[pair_c] = np.array(self.omega[pair_c])
        self.coupling_variable[pair_c] = np.array(self.coupling_variable[pair_c])
        


    def get_coupling_graph_multi_scale(self, graph_output, temp_path, input_file_num=1, verbose=1):

        if verbose==1:
            print('******** Coupling Graph info ********')

        for num, pair_c in enumerate(self.connect_var):

            if pair_c == 'VE_ISK':
                self._get_coupling_graph_multi_scale_VE_ISK(graph_output[pair_c], num, pair_c, verbose=verbose)
                print(self.coupling_variable[pair_c].shape)

            elif pair_c == 'IHC_ISK':
                self._get_coupling_graph_multi_scale_IHC_ISK(graph_output[pair_c], temp_path, num, pair_c, input_file_num=input_file_num, verbose=verbose)
                print(self.coupling_variable[pair_c].shape)

            else:
                print('ERROR: Unknown coupling graph')

        if verbose==1:
            print('******** Coupling Graph info ********')
