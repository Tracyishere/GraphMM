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


class MetaModel:

    def __init__(self, coupling_graph):

        self.coupling_graph = coupling_graph
        self.n_model = len(self.coupling_graph.model_idx)
        self.meta_state = []
        for key in self.coupling_graph.model_idx:
            self.meta_state += self.coupling_graph.model_idx[key].state
        self.meta_n_state = len(self.meta_state) +1
        print(self.meta_n_state)

        self.total_time_list = [self.coupling_graph.model_idx[key].total_time for key in self.coupling_graph.model_idx]
        self.dt_list = [self.coupling_graph.model_idx[key].dt for key in self.coupling_graph.model_idx]
        self.min_dt = min(self.dt_list)
        self.max_total_time = max(self.total_time_list)
        self.max_n_step = len(np.arange(0, self.max_total_time, self.min_dt, dtype=float))
        self.upd_coupling_var = []

        # ======= parameters in the coupling graph =========
        for num,key in enumerate(self.coupling_graph.models):
            
            model_a = self.coupling_graph.models[key][0]
            model_b = self.coupling_graph.models[key][1]
                
            model_a.con_var_idx += [self.coupling_graph.connect_idx[key][0]]
            # model_a.con_omega += [self.coupling_graph.omega] 
            model_a.con_omega += [[item[0] for item in self.coupling_graph.omega]]
            model_a.con_unit_weight += [self.coupling_graph.unit_weight[num][0]]
            
            model_b.con_var_idx += [self.coupling_graph.connect_idx[key][1]]
            # model_b.con_omega += [self.coupling_graph.omega] 
            model_b.con_omega += [[item[1] for item in self.coupling_graph.omega]]
            model_b.con_unit_weight += [self.coupling_graph.unit_weight[num][1]]

        self.noise_model_type = 'time-variant'

    
    def _get_initial_meta_state(self):

        coupling_var_state_mean_t0 = [c[0,0] for c in self.coupling_graph.coupling_variable] 
        coupling_var_state_std_t0 = [c[0,1] for c in self.coupling_graph.coupling_variable] 

        model_var_state_mean_t0, model_var_state_std_t0 = [], []
        for key in self.coupling_graph.model_idx:
            s_model = self.coupling_graph.model_idx[key]
            # initial should be a list
            # print(s_model.initial_noise)
            if isinstance(s_model.initial, list):
                model_var_state_mean_t0 += s_model.initial
            else:
                model_var_state_mean_t0 += s_model.initial.tolist()
            # model_var_state_std_t0 += [np.array(s_model.initial_noise)*np.array(model_var_state_mean_t0)]
            model_var_state_std_t0 += np.diag(s_model.initial_noise).tolist()
            # model_var_state_std_t0 += (np.array(s_model.initial_noise)*s_model.initial).tolist()            
           
        Meta_mean_t0 = np.array(coupling_var_state_mean_t0 + model_var_state_mean_t0)
        # by default, the initial covariance matrix of the metamodel is diagnal, unless additional info is given
        Meta_cov_t0 = np.diag(np.array(coupling_var_state_std_t0 + model_var_state_std_t0))**2

        return Meta_mean_t0, Meta_cov_t0


    def _get_obs_ts(self, x0, mx, filepath=None):
        
        obs_t_interval = get_observations(x0, mx.fx, mx.dt, mx.dt, mx.measure_std_scale, filepath)
        obs_temp = np.random.normal(loc=obs_t_interval[:,:,0], scale=abs(obs_t_interval[:,:,1]))
        # obs_temp = np.random.normal(loc=x0, scale=abs(x0*0.1))

        return obs_temp


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


    def _get_meta_obs_ts(self, ma_mean_ts, mb_mean_ts, ts): # this should be changed to interpolated input model states

        ma_obs_ts = self._get_obs_ts(ma_mean_ts, list(self.coupling_graph.models.values())[0][0])
        mb_obs_ts = self._get_obs_ts(mb_mean_ts, list(self.coupling_graph.models.values())[0][1])
        
        # since observations are considered as perfect estimation to the surrogate model, 
        # thus the model observations can use surrogate model states as a substitue,
        # this for the convience of computing and parameter delivering
        ma_obs_ts = self.coupling_graph.model_states[list(self.coupling_graph.model_states.keys())[0]][ts, :, 0]
        mb_obs_ts = self.coupling_graph.model_states[list(self.coupling_graph.model_states.keys())[1]][ts, :, 0]

        Meta_obs_mean_ts = np.array([ self.coupling_graph.coupling_variable[0][ts, 0] ] + ma_obs_ts.tolist() + mb_obs_ts.tolist())
        Meta_obs_cov_ts = np.diag(Meta_obs_mean_ts*1e-3)**2
        Meta_obs_ts = np.random.multivariate_normal(Meta_obs_mean_ts, Meta_obs_cov_ts)

        return Meta_obs_ts

    
    def _get_meta_param_ts(self, ma_mean_ts, mb_mean_ts, ts): # this is for 2 models only
        
        
        ma_Q_ts = self._get_Q_ts(ma_mean_ts, list(self.coupling_graph.models.values())[0][0].transition_cov_scale)[0]
        mb_Q_ts = self._get_Q_ts(mb_mean_ts, list(self.coupling_graph.models.values())[0][1].transition_cov_scale)[0]  
        Meta_Q_ts = np.diag(np.array([ self.coupling_graph.coupling_variable[0][ts, 1] ]+ np.diag(ma_Q_ts).tolist() + np.diag(mb_Q_ts).tolist()))
        
        ma_R_ts = self._get_R_ts(ma_mean_ts, list(self.coupling_graph.models.values())[0][0].emission_cov_scale)[0]
        mb_R_ts = self._get_R_ts(mb_mean_ts, list(self.coupling_graph.models.values())[0][1].emission_cov_scale)[0]
        Meta_R_ts = np.diag(np.array([ self.coupling_graph.coupling_variable[0][ts, 1] ] + np.diag(ma_R_ts).tolist() + np.diag(mb_R_ts).tolist()))
        
        for num,key in enumerate(self.coupling_graph.models): 
            ma_var_idx, mb_var_idx = self.coupling_graph.connect_idx[key]
            # phi_v1 = self.coupling_graph.w_phi[num][0]*ma_Q_ts[ma_var_idx, ma_var_idx]
            # phi_v2 = self.coupling_graph.w_phi[num][1]*mb_Q_ts[mb_var_idx, mb_var_idx]
            phi_v1 = ma_Q_ts[ma_var_idx, ma_var_idx]
            phi_v2 = mb_Q_ts[mb_var_idx, mb_var_idx]

        con_phi_ts = [phi_v1, phi_v2]

        return Meta_Q_ts, Meta_R_ts, con_phi_ts


    def _fx_metamodel(self, x_ts, dt, ts, omega_ts):

        xout = x_ts.copy()
        units = [1., 1.]

        xout[0] = self.coupling_graph.coupling_variable[0][ts, 0]
        xout[1:3] = self.coupling_graph.models['a_b'][0].fx(x_ts[1:3], self.min_dt)
        xout[2] = ((1-omega_ts[0])*xout[2]*units[0] + omega_ts[0]*xout[0]) / units[0]
        xout[3:] = self.coupling_graph.models['a_b'][1].fx(x_ts[3:], self.min_dt)
        xout[-2] = ((1-omega_ts[1])*xout[-2]*units[1] + omega_ts[1]*xout[0]) / units[1]

        return xout


    def hx(self, x):

        return x


    def inference_for_ts(self, x_ts, P_ts, ts, omega_ts):

        units = [1., 1.]

        sigmas = MerweScaledSigmaPoints(self.meta_n_state, alpha=1e-2, beta=1., kappa=0.)
        surrogate_ts = UKF(dim_x=self.meta_n_state, dim_z=self.meta_n_state, fx=self._fx_metamodel, hx=self.hx, dt=self.min_dt, points=sigmas)
        surrogate_ts.x = x_ts 
        surrogate_ts.P = P_ts
        meta_m1_mean_ts = x_ts[1:3]
        meta_m2_mean_ts = x_ts[3:]
        Q_meta_ts, R_meta_ts, con_phi_ts = self._get_meta_param_ts(meta_m1_mean_ts, meta_m2_mean_ts, ts)
        Q_meta_ts[2, 2] = con_phi_ts[0]
        Q_meta_ts[-2, -2] = con_phi_ts[1]
        surrogate_ts.Q = Q_meta_ts
        surrogate_ts.R = R_meta_ts
        # Meta_obs_ts = self._get_meta_obs_ts(ma_mean_ts=meta_m1_mean_ts, mb_mean_ts=meta_m2_mean_ts, ts=ts)
        Meta_obs_ts = x_ts
        surrogate_ts.predict(fx=self._fx_metamodel, dt=self.min_dt, ts=ts, omega_ts=omega_ts)
        surrogate_ts.update(Meta_obs_ts)

        return surrogate_ts.x, surrogate_ts.P

    
    def inference(self, test_omega=[0.5, 0.5], filepath=None, verbose=1):

        if verbose==1:
            print('******** Metamodel info ********')
            for i,key in enumerate(self.coupling_graph.model_idx):
                model = self.coupling_graph.model_idx[key]
                print('==========================')
                print('model_{}_name: {}'.format(i+1, model.modelname))
                print('total_time: {} {}'.format(model.total_time, model.unit))
                print('time_step: {} {}'.format(model.dt, model.unit))
                print('==========================')
            print('******** Metamodel info ********')

        if filepath is not None:
            output = open(filepath, 'w')
            print(*['coupler','coupler']*self.coupling_graph.n_coupling_var+list(np.repeat(self.meta_state, 2)), file=output, sep=',')
        
        if verbose==1:
            print('-------- Run metamodel ---------') 

        x_ts, P_ts = self._get_initial_meta_state()

        for ts in range(self.max_n_step):
                
            omega_ts = test_omega
            x_ts, P_ts = self.inference_for_ts(x_ts, P_ts, ts, omega_ts)
            Meta_mean_ts = x_ts
            Meta_std_ts = marginal_from_joint(P_ts)

            if filepath is not None:
                print(*list(chain.from_iterable(zip(Meta_mean_ts, Meta_std_ts))), file=output, sep=',')
                
                # # Write x_ts (one row)
                # print(*Meta_mean_ts, file=output, sep=',')
                # # Write P_ts (six rows)
                # for i in range(6):
                #     print(*Meta_std_ts[i], file=output, sep=',')
                
            if verbose==1:
                time.sleep(1e-20)
                process_bar(ts+1, self.max_n_step)
        
        if filepath is not None:
            output.close()

        print('\n-------- Finished ----------')
