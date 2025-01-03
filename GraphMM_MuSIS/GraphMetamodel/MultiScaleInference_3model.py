'''
    Author: Chenxi Wang (chenxi.wang@salilab.org)
    Date: 2022-04-20
'''

import numpy as np
from GraphMetamodel.utils import *
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
import random
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
import time

class MetaModel:

    def __init__(self, coupling_graph, upd_model_state_files):

        self.coupling_graph = coupling_graph
        self.upd_model_state_files = upd_model_state_files
        self.meta_state, self.total_time_list, self.dt_list = [], [], []
        for pair_c in list(self.coupling_graph.models.keys()):
            ma_idx, mb_idx = pair_c.split('_')
            ma = self.coupling_graph.models[pair_c][ma_idx]
            mb = self.coupling_graph.models[pair_c][mb_idx]
            ma.connect_var_idx, ma.connect_omega, ma.connect_unit_weight = [], [], []
            mb.connect_var_idx, mb.connect_omega, mb.connect_unit_weight = [], [], []


        # ======= parameters in the coupling graph =========
        for pair_c in list(self.coupling_graph.models.keys()):
            ma_idx, mb_idx = pair_c.split('_')
            for m_idx in list(self.coupling_graph.models[pair_c].keys()):
                self.meta_state += self.coupling_graph.models[pair_c][m_idx].state
                self.total_time_list += [self.coupling_graph.models[pair_c][m_idx].total_time]
                self.dt_list += [self.coupling_graph.models[pair_c][m_idx].dt]
            ma = self.coupling_graph.models[pair_c][ma_idx]
            mb = self.coupling_graph.models[pair_c][mb_idx]
            # model_a = self.coupling_graph.models[key][0]
            # model_b = self.coupling_graph.models[key][1]
            ma.connect_var_idx += [self.coupling_graph.connect_idx[pair_c][0]]
            ma.connect_omega += [item[0] for item in self.coupling_graph.omega[pair_c]]
            ma.connect_unit_weight += [self.coupling_graph.unit_weight[pair_c][0]]
            mb.connect_var_idx += [self.coupling_graph.connect_idx[pair_c][1]]
            mb.connect_omega += [item[1] for item in self.coupling_graph.omega[pair_c]]
            mb.connect_unit_weight += [self.coupling_graph.unit_weight[pair_c][1]]

        self.min_dt = min(self.dt_list)
        print('dt_list: ', self.dt_list)
        print('self.min_dt: ', self.min_dt) # 1e-6
        self.max_total_time = max(self.total_time_list)
        self.max_n_step = len(np.arange(0, self.max_total_time, self.min_dt, dtype=float))
        self.upd_coupling_var = []
        self.noise_model_type = 'time-variant'
        
        self.meta_state = self.meta_state[:9] + self.meta_state[21:]
        self.n_meta_state = len(self.meta_state) +2 # +2 if only for VE_IHC_ISK
        
            
    
    def _get_initial_meta_state(self):

        coupling_var_state_mean_t0 = list(self.coupling_graph.coupling_variable[pair_c][0, 0] \
                                      for pair_c in list(self.coupling_graph.coupling_variable.keys())) 
        coupling_var_state_std_t0 = list(self.coupling_graph.coupling_variable[pair_c][0, 1] \
                                     for pair_c in list(self.coupling_graph.coupling_variable.keys()))

        model_var_state_mean_t0, model_var_state_std_t0 = [], []
        for pair_c in list(self.coupling_graph.models.keys()):
            for m_idx in list(self.coupling_graph.models[pair_c].keys()):
                s_model = self.coupling_graph.models[pair_c][m_idx]
                if isinstance(s_model.initial, list): # initial should be a list
                    model_var_state_mean_t0 += s_model.initial 
                else:
                    model_var_state_mean_t0 += list(s_model.initial) 
                model_var_state_std_t0 += np.diag(s_model.initial_noise).tolist() 
                 # by default, the initial covariance matrix of the metamodel is diagnal, 
                 # unless additional info is given           
        
        model_var_state_mean_t0 = model_var_state_mean_t0[:9] + model_var_state_mean_t0[21:]
        model_var_state_std_t0 = model_var_state_std_t0[:9] + model_var_state_std_t0[21:]
        # to eliminate repeated variables

        Meta_mean_t0 = np.array(coupling_var_state_mean_t0 + model_var_state_mean_t0) 
        Meta_cov_t0 = np.diag(np.array(coupling_var_state_std_t0 + model_var_state_std_t0))**2 # assume diagnol

        return Meta_mean_t0, Meta_cov_t0

    
    

    def _get_obs_ts(self, x0, mx, filepath=None):
        
        obs_t_interval = get_observations(x0, mx.fx, mx.dt, mx.dt, mx.measure_std_scale, filepath)
        obs_temp = np.random.normal(loc=obs_t_interval[:,:,0], scale=abs(obs_t_interval[:,:,1]))
        # obs_temp = np.random.normal(loc=x0, scale=abs(x0*0.1))

        return obs_temp


    def _get_meta_obs_ts(self, batch_time, ts): # this should be changed to interpolated input model states
        
        '''
        since observations are considered as perfect estimation to the surrogate model, 
        thus the model observations can use surrogate model states as a substitue,
        this for the convience of computing and parameter delivering
        '''

        pair_c = list(self.batch_model_states.keys())[0]
        ma_idx, mb_idx = pair_c.split('_')
        ma_obs_ts = self.batch_model_states[pair_c][ma_idx][ts, :, 0]
        mb_obs_ts = self.batch_model_states[pair_c][mb_idx][ts, :, 0]

        Meta_obs_mean_ts = np.array([ self.coupling_graph.coupling_variable['IHC_ISK'][batch_time+ts][0] ]\
            + ma_obs_ts.tolist() + mb_obs_ts.tolist())
        # Meta_obs_cov_ts = np.diag(Meta_obs_mean_ts*1e-3)**2
        # Meta_obs_ts = np.random.multivariate_normal(Meta_obs_mean_ts, Meta_obs_cov_ts)

        return Meta_obs_mean_ts


    def _get_Q_ts(self, z_mean, transition_cov_scale, n_step=1):

        '''
        get the transition noise matrix to approximate a transition noise model, 
        '''

        Q_ts = []
        z_mean = np.array(z_mean).reshape(n_step, -1)

        if self.noise_model_type == 'time-invariant':
            Q_ts = [np.diag(abs(np.mean(z_mean, axis=0)*transition_cov_scale)) for i in range(n_step)]

        elif self.noise_model_type == 'time-variant':
            ''' time-invariant means the noise model is relavent to the value of state variable'''
            Q_ts = [np.diag(abs(z_mean[i]*random.gauss(transition_cov_scale, transition_cov_scale*1e-2))) for i in range(n_step)]
        else:
            print('ERROR: Unknown noise model type.')

        return np.array(Q_ts) 

    
    def _get_R_ts(self, x_mean, emission_cov_scale, n_step=1):

        '''
        get the emission noise matrix to approximate a emission noise model
        '''

        R_ts = []
        x_mean = np.array(x_mean).reshape(n_step, -1)

        if self.noise_model_type == 'time-invariant':
            R_ts = [np.diag(abs(np.mean(x_mean, axis=0)*emission_cov_scale)) for i in range(n_step)]

        elif self.noise_model_type == 'time-variant':
            R_ts = [np.diag(abs(x_mean[i]*random.gauss(emission_cov_scale, emission_cov_scale*1e-2))) for i in range(n_step)]
        else:
            print('ERROR: Unknown noise model type.')

        return np.array(R_ts)

    

    def _get_meta_param_ts(self, meta_mean_ts, batch_time, ts): # for multiple models
        
        coupling_var_ts, Meta_Q_ts, Meta_R_ts, con_phi_ts = [], [], [], []

        for num, pair_c in enumerate(self.coupling_graph.connect_var):

            ma, mb = list(self.coupling_graph.models[pair_c].values())
            ma_idx, mb_idx = pair_c.split('_')
            ma_state = [self.meta_state.index(i) for i in self.meta_state if ma_idx in i]
            mb_state = [self.meta_state.index(i) for i in self.meta_state if mb_idx in i]

            ma_start_idx = min(ma_state) + 2
            ma_end_idx = max(ma_state) + 3
            mb_start_idx = min(mb_state) + 2
            mb_end_idx = max(mb_state) + 3

            ma_mean_ts = meta_mean_ts[ma_start_idx:ma_end_idx]
            mb_mean_ts = meta_mean_ts[mb_start_idx:mb_end_idx]

            ma_Q_ts = self._get_Q_ts(ma_mean_ts, ma.transition_cov_scale)[0]
            mb_Q_ts = self._get_Q_ts(mb_mean_ts, mb.transition_cov_scale)[0]  
            ma_R_ts = self._get_R_ts(ma_mean_ts, ma.emission_cov_scale)[0]
            mb_R_ts = self._get_R_ts(mb_mean_ts, mb.emission_cov_scale)[0]

            coupling_var_ts += [self.coupling_graph.coupling_variable[pair_c][batch_time+ts][1]]

            Meta_Q_ts += np.diag(ma_Q_ts).tolist() 
            Meta_Q_ts += np.diag(mb_Q_ts).tolist()
            Meta_R_ts += np.diag(ma_R_ts).tolist() 
            Meta_R_ts += np.diag(mb_R_ts).tolist()
            
            ma_var_idx, mb_var_idx = self.coupling_graph.connect_idx[pair_c]
            phi_v1 = ma_Q_ts[ma_var_idx, ma_var_idx]
            phi_v2 = mb_Q_ts[mb_var_idx, mb_var_idx]

            con_phi_ts += [(phi_v1, phi_v2)]

        Meta_Q_ts = np.diag(np.array(coupling_var_ts + Meta_Q_ts[:9] + Meta_Q_ts[21:]))
        Meta_R_ts = np.diag(np.array(coupling_var_ts + Meta_R_ts[:9] + Meta_R_ts[21:])) 

        return Meta_Q_ts, Meta_R_ts, con_phi_ts

    
    def _fx_metamodel(self, x_ts, dt, ts, coupling_omega_ts, ton_lst, V_list):

        xout = x_ts.copy()

        num = 1
        pair_c = 'IHC_ISK'
        self.min_dt = 1e-4

        xout[num] = self.coupling_graph.coupling_variable[pair_c][ts, 0]

        ma, mb = list(self.coupling_graph.models[pair_c].values())
        ma_idx, mb_idx = pair_c.split('_')
        var1, var2 = self.coupling_graph.connect_var[pair_c]

        units = self.coupling_graph.unit_weight[pair_c]
        omega_ts = coupling_omega_ts[num]

        ma_state = [self.meta_state.index(i) for i in self.meta_state if ma_idx in i]
        mb_state = [self.meta_state.index(i) for i in self.meta_state if mb_idx in i]

        ma_start_idx = min(ma_state) + 2
        ma_end_idx = max(ma_state) + 3
        mb_start_idx = min(mb_state) + 2
        mb_end_idx = max(mb_state) + 3
            
        ma_nc_idx = self.meta_state.index(var1) + 2
        mb_nc_idx = self.meta_state.index(var2) + 2
        
        xout[ma_start_idx:ma_end_idx] = ma.fx(x_ts[ma_start_idx:ma_end_idx], self.min_dt, ts)
        xout[ma_nc_idx] = ((1-omega_ts[0])*xout[ma_nc_idx]*units[0] + omega_ts[0]*xout[num]) / units[0]
        xout[mb_start_idx:mb_end_idx] = mb.fx(x_ts[mb_start_idx:mb_end_idx], self.min_dt, ts, V_list)
        xout[mb_nc_idx] = ((1-omega_ts[1])*xout[mb_nc_idx]*units[1] + omega_ts[1]*xout[num]) / units[1] # 'Ca_ic.ISK'
 
        # print('var1, var2', var1, var2)
        # print('ma_nc_idx', ma_nc_idx) # 184
        # print('mb_nc_idx', mb_nc_idx) # 241
        # print('ma_start_idx:ma_end_idx', ma_start_idx, ma_end_idx) # 11-239
        # print('mb_start_idx:mb_end_idx', mb_start_idx, mb_end_idx) # 239-251

        num = 0
        pair_c = 'VE_ISK'

        xout[num] = self.coupling_graph.coupling_variable[pair_c][ts, 0]

        ma, mb = list(self.coupling_graph.models[pair_c].values())
        ma_idx, mb_idx = pair_c.split('_')
        var1, var2 = self.coupling_graph.connect_var[pair_c]

        units = self.coupling_graph.unit_weight[pair_c]
        omega_ts = coupling_omega_ts[num]

        ma_state = [self.meta_state.index(i) for i in self.meta_state if ma_idx in i]
        mb_state = [self.meta_state.index(i) for i in self.meta_state if mb_idx in i]

        ma_start_idx = min(ma_state) + 2
        ma_end_idx = max(ma_state) + 3
        mb_start_idx = min(mb_state) + 2
        mb_end_idx = max(mb_state) + 3
            
        ma_nc_idx = self.meta_state.index(var1) + 2
        mb_nc_idx = self.meta_state.index(var2) + 2
        
        # self.min_dt = 6e-5 

        xout[ma_start_idx:ma_end_idx] = ma.fx(x_ts[ma_start_idx:ma_end_idx], self.min_dt, ts, ton_lst)
        xout[ma_nc_idx] = ((1-omega_ts[0])*xout[ma_nc_idx]*units[0] + omega_ts[0]*xout[num]) / units[0]
        xout[mb_nc_idx] = ((1-omega_ts[1])*xout[mb_nc_idx]*units[1] + omega_ts[1]*xout[num]) / units[1]

        # print('var1, var2', var1, var2) 
        # print('ma_nc_idx', ma_nc_idx) # 4
        # print('mb_nc_idx', mb_nc_idx) # 249
        # print('ma_start_idx:ma_end_idx', ma_start_idx, ma_end_idx) # 2-11
        # print('mb_start_idx:mb_end_idx', mb_start_idx, mb_end_idx) # 239-251
        
        return xout


    def hx(self, x):

        return x


    def inference_for_ts(self, x_ts, P_ts, batch_time, ts, coupling_omega_ts, ton_lst, V_list):

        sigmas = MerweScaledSigmaPoints(self.n_meta_state, alpha=1e-3, beta=2., kappa=0.)
        surrogate_ts = UKF(dim_x=self.n_meta_state, dim_z=self.n_meta_state, fx=self._fx_metamodel, 
                           hx=self.hx, dt=self.min_dt, points=sigmas)

        surrogate_ts.x = x_ts 
        surrogate_ts.P = P_ts
        meta_mean_ts = x_ts
        Q_meta_ts, R_meta_ts, con_phi_ts = self._get_meta_param_ts(meta_mean_ts, batch_time, ts-batch_time)

        for num, pair_c in enumerate(self.coupling_graph.connect_var):

            var1, var2 = self.coupling_graph.connect_var[pair_c]
            ma_nc_idx = self.meta_state.index(var1) + 2
            mb_nc_idx = self.meta_state.index(var2) + 2
            Q_meta_ts[ma_nc_idx, ma_nc_idx], Q_meta_ts[mb_nc_idx, mb_nc_idx] = con_phi_ts[num]

        surrogate_ts.Q = Q_meta_ts
        surrogate_ts.R = R_meta_ts
        # Meta_obs_ts = self._get_meta_obs_ts(batch_time=batch_time, ts=ts)

        # start_time = time.time()
        surrogate_ts.predict(fx=self._fx_metamodel, ts=ts, coupling_omega_ts=coupling_omega_ts, ton_lst=ton_lst, V_list=V_list)
        # end_time = time.time()
        # time_flag = end_time - start_time
        # print('predict: {}'.format(time_flag))

        Meta_obs_ts = surrogate_ts.x

        # start_time = time.time()
        surrogate_ts.update(Meta_obs_ts)
        # end_time = time.time()
        # time_flag = end_time - start_time
        # print('update: {}'.format(time_flag))

        return surrogate_ts.x, surrogate_ts.P

    
    def inference(self, input_file_num, ton_lst, V_list, filepath=None, verbose=1):

        if verbose==1:
            print('********************************')
            print('******** Metamodel info ********')
            print('********************************')
            for pair_c in list(self.coupling_graph.models.keys()):
                for i,m_idx in enumerate(list(self.coupling_graph.models[pair_c].keys())):
                    model = self.coupling_graph.models[pair_c][m_idx]
                    print('==========================')
                    print('model_{}_name: {}'.format(i+1, model.modelname))
                    print('total_time: {} {}'.format(model.total_time, model.unit))
                    print('time_step: {} {}'.format(model.dt, model.unit))
                    print('==========================')
            print('Metamodel variables: {}'.format(self.n_meta_state))
            print('********************************')
            print('******** Metamodel info ********')
            print('********************************')

        if filepath is not None:
            output = open(filepath, 'w')
            print(*['coupler','coupler']*self.coupling_graph.n_coupling_var+\
                  list(np.repeat(self.meta_state, 2)), file=output, sep=',')
        
        if verbose==1:
            print('-------- Run metamodel ---------') 

        x_ts, P_ts = self._get_initial_meta_state()
            
        for nb in range(max(self.coupling_graph.n_batch[pair_c])):

            print('batch {}/{} metamodel inference'.format(nb, max(self.coupling_graph.n_batch[pair_c])))

            if nb >= 1:
                x_ts = x_ts_batch
                P_ts = P_ts_batch

            self.batch_model_states = {}

            mb_batchsize = self.coupling_graph.batchsize
            # ma_start_line = 1 + int(nb*ma_batchsize) # we know ma_dt is larger
            # mb_start_line = 1 + int(nb*self.coupling_graph.batchsize)
            mb_start_line = int(nb*mb_batchsize) # we know ma_dt is larger
            mc_start_line = int(nb*self.coupling_graph.batchsize)
            ma_start_line = mc_start_line
            
            #############################################################
            # #### This part is for geting observed node states
            # ma_idx = 'VE'
            # mb_idx = 'IHC'
            # mc_idx = 'ISK'

            # if input_file_num == 1:
            #     self.batch_model_states[mb_idx] = np.genfromtxt(self.upd_model_state_files[1], 
            #                                     delimiter=',', 
            #                                     skip_header=mb_start_line, 
            #                                     max_rows=mb_batchsize).reshape(mb_batchsize, -1, 2)
            # else:
            #     temp0 = np.genfromtxt(self.upd_model_state_files[1], 
            #                             delimiter=',', 
            #                             skip_header=mb_start_line, 
            #                             max_rows=mb_batchsize).reshape(mb_batchsize, -1, 1) # (10000, 4, 2)
            #     # print(temp0.shape)
            #     for nf in range(1, input_file_num):
            #         file_path = self.upd_model_state_files[1][:-5] + str(nf) + '.csv'
            #         # print(file_path)
            #         temp1 = np.genfromtxt(file_path, delimiter=',', 
            #                                 skip_header=mb_start_line, 
            #                                 max_rows=mb_batchsize).reshape(mb_batchsize, -1, 1)
            #         n_col = nf
            #         temp2 = np.concatenate((temp0[:,:n_col,:].reshape(mb_batchsize,n_col,-1), 
            #                                 temp1[:,0,:].reshape(mb_batchsize,1,-1)), axis=1)
            #         # print(temp2.shape)
            #         for i in range(1, 4): # IHC only
            #             temp3 = np.concatenate((temp0[:,i*n_col:(i+1)*n_col,:].reshape(mb_batchsize,n_col,-1), 
            #                                     temp1[:,i,:].reshape(mb_batchsize, 1, -1)), axis=1)
            #             temp2 = np.concatenate((temp2, temp3), axis=1)
            #         temp0 = temp2
            # self.batch_model_states[mb_idx] = temp0
            # print(self.batch_model_states[mb_idx].shape)

            # self.batch_model_states[mc_idx] = np.genfromtxt(self.upd_model_state_files[2], 
            #                             delimiter=',', 
            #                             skip_header=mc_start_line, 
            #                             max_rows=self.coupling_graph.batchsize).reshape(self.coupling_graph.batchsize, -1, 2)

            # self.batch_model_states[ma_idx] = np.genfromtxt(self.upd_model_state_files[0], 
            #                             delimiter=',', 
            #                             skip_header=ma_start_line, 
            #                             max_rows=self.coupling_graph.batchsize).reshape(self.coupling_graph.batchsize, -1, 2)
            #############################################################
            
            batch_time = mb_start_line

            for ts in range(self.coupling_graph.batchsize):

                coupling_omega_ts = []
                for num, pair_c in enumerate(self.coupling_graph.connect_var):
                    coupling_omega_ts += [self.coupling_graph.omega[pair_c][batch_time+ts]] 
                    
                x_ts, P_ts = self.inference_for_ts(x_ts, P_ts, batch_time, batch_time+ts, coupling_omega_ts, ton_lst, V_list)
                Meta_mean_ts = x_ts
                Meta_std_ts = np.sqrt(np.diag(P_ts))

                if filepath is not None:
                    print(*list(chain.from_iterable(zip(Meta_mean_ts, Meta_std_ts))), file=output, sep=',')
                    
                # if verbose==1:
                #     time.sleep(1e-20)
                #     process_bar(ts+1, self.max_n_step)

            x_ts_batch = x_ts
            P_ts_batch = P_ts
        
        if filepath is not None:
            output.close()

        print('\n-------- Finished ----------')

