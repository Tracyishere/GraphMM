'''
    Author: Chenxi Wang (chenxi.wang@salilab.org)
    Date: 2022-04-20
'''

import numpy as np
from GraphMetamodel.utils import *
from scipy.stats import norm
import matplotlib.pyplot as plt
from itertools import chain
import random
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
import time
import numba
from functools import lru_cache
from scipy import sparse

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

        self.model_indices = {}
        
        for pair_c in list(self.coupling_graph.models.keys()):
            ma_idx, mb_idx = pair_c.split('_')
            for m_idx in list(self.coupling_graph.models[pair_c].keys()):
                self.meta_state += self.coupling_graph.models[pair_c][m_idx].state
                self.total_time_list += [self.coupling_graph.models[pair_c][m_idx].total_time]
                self.dt_list += [self.coupling_graph.models[pair_c][m_idx].dt]
                
            ma = self.coupling_graph.models[pair_c][ma_idx]
            mb = self.coupling_graph.models[pair_c][mb_idx]
            
            ma.connect_var_idx += [self.coupling_graph.connect_idx[pair_c][0]]
            ma.connect_omega += [item[0] for item in self.coupling_graph.omega[pair_c]]
            ma.connect_unit_weight += [self.coupling_graph.unit_weight[pair_c][0]]
            mb.connect_var_idx += [self.coupling_graph.connect_idx[pair_c][1]]
            mb.connect_omega += [item[1] for item in self.coupling_graph.omega[pair_c]]
            mb.connect_unit_weight += [self.coupling_graph.unit_weight[pair_c][1]]

        self.min_dt = min(self.dt_list)
        print('dt_list: ', self.dt_list)
        print('self.min_dt: ', self.min_dt)
        self.max_total_time = max(self.total_time_list)
        
        self.max_n_step = int(np.ceil(self.max_total_time / self.min_dt))
        self.upd_coupling_var = []
        self.noise_model_type = 'time-variant'
        
        self.meta_state = self.meta_state[:9] + self.meta_state[21:]
        self.n_meta_state = len(self.meta_state) + 2
        
        self._precompute_indices()
            
    
    def _precompute_indices(self):
        self.model_indices = {}
        for pair_c in list(self.coupling_graph.models.keys()):
            ma_idx, mb_idx = pair_c.split('_')
            
            ma_state_indices = np.array([i for i, state in enumerate(self.meta_state) if ma_idx in state])
            mb_state_indices = np.array([i for i, state in enumerate(self.meta_state) if mb_idx in state])
            
            if len(ma_state_indices) > 0 and len(mb_state_indices) > 0:
                self.model_indices[pair_c] = {
                    ma_idx: {
                        'start': min(ma_state_indices) + 2,
                        'end': max(ma_state_indices) + 3,
                        'indices': ma_state_indices
                    },
                    mb_idx: {
                        'start': min(mb_state_indices) + 2,
                        'end': max(mb_state_indices) + 3,
                        'indices': mb_state_indices
                    }
                }
                
        self.connect_var_indices = {}
        for pair_c in self.coupling_graph.connect_var:
            var1, var2 = self.coupling_graph.connect_var[pair_c]
            ma_nc_idx = self.meta_state.index(var1) + 2
            mb_nc_idx = self.meta_state.index(var2) + 2
            self.connect_var_indices[pair_c] = (ma_nc_idx, mb_nc_idx)
    
    def _get_initial_meta_state(self):
        coupling_var_state_mean_t0 = np.array([
            self.coupling_graph.coupling_variable[pair_c][0, 0] 
            for pair_c in self.coupling_graph.coupling_variable
        ])
        
        coupling_var_state_std_t0 = np.array([
            self.coupling_graph.coupling_variable[pair_c][0, 1] 
            for pair_c in self.coupling_graph.coupling_variable
        ])

        model_var_state_mean_t0, model_var_state_std_t0 = [], []
        for pair_c in self.coupling_graph.models:
            for m_idx in self.coupling_graph.models[pair_c]:
                s_model = self.coupling_graph.models[pair_c][m_idx]
                if isinstance(s_model.initial, list):
                    model_var_state_mean_t0.extend(s_model.initial)
                else:
                    model_var_state_mean_t0.extend(list(s_model.initial))
                    
                model_var_state_std_t0.extend(np.diag(s_model.initial_noise).tolist())
        
        model_var_state_mean_t0 = np.array(model_var_state_mean_t0[:9] + model_var_state_mean_t0[21:])
        model_var_state_std_t0 = np.array(model_var_state_std_t0[:9] + model_var_state_std_t0[21:])

        Meta_mean_t0 = np.concatenate([coupling_var_state_mean_t0, model_var_state_mean_t0])
        Meta_cov_t0 = np.diag(np.square(np.concatenate([coupling_var_state_std_t0, model_var_state_std_t0])))

        return Meta_mean_t0, Meta_cov_t0

    def _get_obs_ts(self, x0, mx, filepath=None):
        obs_t_interval = get_observations(x0, mx.fx, mx.dt, mx.dt, mx.measure_std_scale, filepath)
        obs_temp = np.random.normal(loc=obs_t_interval[:,:,0], scale=abs(obs_t_interval[:,:,1]))
        return obs_temp

    def _get_meta_obs_ts(self, batch_time, ts):
        pair_c = list(self.batch_model_states.keys())[0]
        ma_idx, mb_idx = pair_c.split('_')
        
        ma_obs_ts = self.batch_model_states[pair_c][ma_idx][ts, :, 0]
        mb_obs_ts = self.batch_model_states[pair_c][mb_idx][ts, :, 0]

        Meta_obs_mean_ts = np.concatenate([
            [self.coupling_graph.coupling_variable['ICN_ISK'][batch_time+ts][0]],
            ma_obs_ts,
            mb_obs_ts
        ])
        
        return Meta_obs_mean_ts

    @lru_cache(maxsize=128)
    def _get_Q_ts_cached(self, z_mean_tuple, transition_cov_scale):
        z_mean = np.array(z_mean_tuple)
        
        if self.noise_model_type == 'time-invariant':
            Q = np.diag(abs(z_mean * transition_cov_scale))
        elif self.noise_model_type == 'time-variant':
            random_scale = transition_cov_scale * (1 + 1e-2 * np.random.normal())
            Q = np.diag(abs(z_mean * random_scale))
        else:
            print('ERROR: Unknown noise model type.')
            Q = np.zeros((len(z_mean), len(z_mean)))
            
        return Q

    def _get_Q_ts(self, z_mean, transition_cov_scale, n_step=1):
        z_mean = np.array(z_mean).reshape(n_step, -1)
        
        Q_ts = [self._get_Q_ts_cached(tuple(z_mean[i]), transition_cov_scale) for i in range(n_step)]
            
        return np.array(Q_ts)

    @lru_cache(maxsize=128)
    def _get_R_ts_cached(self, x_mean_tuple, emission_cov_scale):
        x_mean = np.array(x_mean_tuple)
        
        if self.noise_model_type == 'time-invariant':
            R = np.diag(abs(x_mean * emission_cov_scale))
        elif self.noise_model_type == 'time-variant':
            random_scale = emission_cov_scale * (1 + 1e-2 * np.random.normal())
            R = np.diag(abs(x_mean * random_scale))
        else:
            print('ERROR: Unknown noise model type.')
            R = np.zeros((len(x_mean), len(x_mean)))
            
        return R
    
    def _get_R_ts(self, x_mean, emission_cov_scale, n_step=1):
        x_mean = np.array(x_mean).reshape(n_step, -1)
        
        R_ts = [self._get_R_ts_cached(tuple(x_mean[i]), emission_cov_scale) for i in range(n_step)]
            
        return np.array(R_ts)

    def _get_meta_param_ts(self, meta_mean_ts, batch_time, ts):
        coupling_var_ts = []
        Meta_Q_diag_values = []
        Meta_R_diag_values = []
        con_phi_ts = []

        for num, pair_c in enumerate(self.coupling_graph.connect_var):
            ma_idx, mb_idx = pair_c.split('_')
            ma, mb = list(self.coupling_graph.models[pair_c].values())
            
            if pair_c in self.model_indices:
                ma_start_idx = self.model_indices[pair_c][ma_idx]['start']
                ma_end_idx = self.model_indices[pair_c][ma_idx]['end']
                mb_start_idx = self.model_indices[pair_c][mb_idx]['start']
                mb_end_idx = self.model_indices[pair_c][mb_idx]['end']
            else:
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

            coupling_var_ts.append(self.coupling_graph.coupling_variable[pair_c][batch_time+ts][1])

            Meta_Q_diag_values.extend(np.diag(ma_Q_ts))
            Meta_Q_diag_values.extend(np.diag(mb_Q_ts))
            Meta_R_diag_values.extend(np.diag(ma_R_ts))
            Meta_R_diag_values.extend(np.diag(mb_R_ts))
            
            ma_var_idx, mb_var_idx = self.coupling_graph.connect_idx[pair_c]
            phi_v1 = ma_Q_ts[ma_var_idx, ma_var_idx]
            phi_v2 = mb_Q_ts[mb_var_idx, mb_var_idx]

            con_phi_ts.append((phi_v1, phi_v2))

        Meta_Q_diag_values = np.array(Meta_Q_diag_values)
        Meta_R_diag_values = np.array(Meta_R_diag_values)
        
        diag_values_q = np.concatenate([coupling_var_ts, Meta_Q_diag_values[:9], Meta_Q_diag_values[21:]])
        diag_values_r = np.concatenate([coupling_var_ts, Meta_R_diag_values[:9], Meta_R_diag_values[21:]])
        
        if len(diag_values_q) > 100:
            Meta_Q_ts = sparse.diags(diag_values_q).toarray()
            Meta_R_ts = sparse.diags(diag_values_r).toarray()
        else:
            Meta_Q_ts = np.diag(diag_values_q)
            Meta_R_ts = np.diag(diag_values_r)

        return Meta_Q_ts, Meta_R_ts, con_phi_ts

    def _fx_metamodel(self, x_ts, dt, ts, coupling_omega_ts, ton_lst, V_list):
        xout = np.copy(x_ts)
        
        pair_configs = [
            ('ICN_ISK', 1, 1e-4),
            ('VE_ISK', 0, 1e-4)
        ]
        
        for pair_name, num, min_dt in pair_configs:
            xout[num] = self.coupling_graph.coupling_variable[pair_name][ts, 0]
            
            ma, mb = list(self.coupling_graph.models[pair_name].values())
            ma_idx, mb_idx = pair_name.split('_')
            var1, var2 = self.coupling_graph.connect_var[pair_name]
            units = self.coupling_graph.unit_weight[pair_name]
            omega_ts = coupling_omega_ts[num]
            
            if pair_name in self.model_indices:
                ma_start_idx = self.model_indices[pair_name][ma_idx]['start']
                ma_end_idx = self.model_indices[pair_name][ma_idx]['end']
                mb_start_idx = self.model_indices[pair_name][mb_idx]['start']
                mb_end_idx = self.model_indices[pair_name][mb_idx]['end']
            else:
                ma_state = np.array([i for i, state in enumerate(self.meta_state) if ma_idx in state])
                mb_state = np.array([i for i, state in enumerate(self.meta_state) if mb_idx in state])
                ma_start_idx = np.min(ma_state) + 2 if len(ma_state) > 0 else 0
                ma_end_idx = np.max(ma_state) + 3 if len(ma_state) > 0 else 0
                mb_start_idx = np.min(mb_state) + 2 if len(mb_state) > 0 else 0
                mb_end_idx = np.max(mb_state) + 3 if len(mb_state) > 0 else 0
            
            if pair_name in self.connect_var_indices:
                ma_nc_idx, mb_nc_idx = self.connect_var_indices[pair_name]
            else:
                ma_nc_idx = self.meta_state.index(var1) + 2
                mb_nc_idx = self.meta_state.index(var2) + 2
            
            if pair_name == 'ICN_ISK':
                xout[ma_start_idx:ma_end_idx] = ma.fx(x_ts[ma_start_idx:ma_end_idx], min_dt, ts)
                factor_a = (1.0 - omega_ts[0]) * units[0]
                xout[ma_nc_idx] = (factor_a * xout[ma_nc_idx] + omega_ts[0] * xout[num]) / units[0]
                xout[mb_start_idx:mb_end_idx] = mb.fx(x_ts[mb_start_idx:mb_end_idx], min_dt, ts, V_list)
                factor_b = (1.0 - omega_ts[1]) * units[1]
                xout[mb_nc_idx] = (factor_b * xout[mb_nc_idx] + omega_ts[1] * xout[num]) / units[1]
            else:
                xout[ma_start_idx:ma_end_idx] = ma.fx(x_ts[ma_start_idx:ma_end_idx], min_dt, ts, ton_lst)
                factor_a = (1.0 - omega_ts[0]) * units[0]
                xout[ma_nc_idx] = (factor_a * xout[ma_nc_idx] + omega_ts[0] * xout[num]) / units[0]
                factor_b = (1.0 - omega_ts[1]) * units[1]
                xout[mb_nc_idx] = (factor_b * xout[mb_nc_idx] + omega_ts[1] * xout[num]) / units[1]
            
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
            if pair_c in self.connect_var_indices:
                ma_nc_idx, mb_nc_idx = self.connect_var_indices[pair_c]
            else:
                var1, var2 = self.coupling_graph.connect_var[pair_c]
                ma_nc_idx = self.meta_state.index(var1) + 2
                mb_nc_idx = self.meta_state.index(var2) + 2
                
            Q_meta_ts[ma_nc_idx, ma_nc_idx], Q_meta_ts[mb_nc_idx, mb_nc_idx] = con_phi_ts[num]

        surrogate_ts.Q = Q_meta_ts
        surrogate_ts.R = R_meta_ts

        surrogate_ts.predict(fx=self._fx_metamodel, ts=ts, coupling_omega_ts=coupling_omega_ts, ton_lst=ton_lst, V_list=V_list)
        Meta_obs_ts = surrogate_ts.x
        surrogate_ts.update(Meta_obs_ts)

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

        output = None
        if filepath is not None:
            output = open(filepath, 'w')
            header = ','.join(['coupler','coupler']*self.coupling_graph.n_coupling_var + 
                             list(np.repeat(self.meta_state, 2)))
            print(header, file=output)
        
        if verbose==1:
            print('-------- Run metamodel ---------') 

        x_ts, P_ts = self._get_initial_meta_state()
        
        max_batch = max(self.coupling_graph.n_batch.get(list(self.coupling_graph.n_batch.keys())[0], [0]))
            
        for nb in range(max_batch):
            print('batch {}/{} metamodel inference'.format(nb, max_batch))

            if nb >= 1:
                x_ts = x_ts_batch
                P_ts = P_ts_batch

            self.batch_model_states = {}

            mb_batchsize = self.coupling_graph.batchsize
            mb_start_line = int(nb*mb_batchsize)
            mc_start_line = int(nb*self.coupling_graph.batchsize)
            ma_start_line = mc_start_line
            
            batch_time = mb_start_line
            
            batchsize = self.coupling_graph.batchsize
            
            for ts in range(batchsize):
                coupling_omega_ts = [
                    self.coupling_graph.omega[pair_c][batch_time+ts] 
                    for pair_c in self.coupling_graph.connect_var
                ]
                    
                x_ts, P_ts = self.inference_for_ts(x_ts, P_ts, batch_time, batch_time+ts, coupling_omega_ts, ton_lst, V_list)
                Meta_mean_ts = x_ts
                Meta_std_ts = np.sqrt(np.diag(P_ts))

                if filepath is not None:
                    interleaved = list(chain.from_iterable(zip(Meta_mean_ts, Meta_std_ts)))
                    print(','.join(map(str, interleaved)), file=output)
                    if ts % 10 == 0:
                        output.flush()
                
            x_ts_batch = x_ts
            P_ts_batch = P_ts
        
        if filepath is not None:
            output.close()

        print('\n-------- Finished ----------')
