'''
    Author: Chenxi Wang (chenxi.wang@salilab.org)
    Date: 2021-10-04

    This script includes functions to couple surrogate models at different timescales.
'''

import numpy as np
from GraphMetamodel.utils import *
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import unscented_transform


class MetaModel:


    def __init__(self, coupling_graph):
        
        self.m1 = coupling_graph.models['a_b'][0]
        self.m2 = coupling_graph.models['a_b'][1]
        self.coupling_graph = coupling_graph
        self.coupler = self.coupling_graph.coupling_variable[0]
        self.state = self.m1.state + self.m2.state
        self.n_state = len(self.state)
        self.initial = np.array(list(self.m1.initial) + list(self.m2.initial))
        self.initial_noise = self.m1.initial_noise
        self.m1_R = self.coupling_graph.meta_R['a']
        self.m1_Q = self.coupling_graph.meta_Q['a']
        self.m2_R = self.coupling_graph.meta_R['b']
        self.m2_Q = self.coupling_graph.meta_Q['b']
        self.m1_obs = self.coupling_graph.model_obs['a']
        self.m2_obs = self.coupling_graph.model_obs['b']
        self.R = [np.diag(list(np.diag(self.m1_R[i]))+list(np.diag(self.m2_R[i]))) for i in range(self.m2.n_step)]
        self.Q = [np.diag(list(np.diag(self.m1_Q[i]))+list(np.diag(self.m2_Q[i]))) for i in range(self.m2.n_step)]


    def _isPD(self, B):
        
        ''' Returns true when input is positive-definite, via Cholesky '''
        
        try:
            _ = np.linalg.cholesky(B)
            return True
        
        except np.linalg.LinAlgError:
            return False



    def _nearestPD(self, A):
        
        ''' Find the nearest positive-definite matrix to input '''

        B = (A + A.T) / 2
        _, s, V = np.linalg.svd(B)
        H = np.dot(V.T, np.dot(np.diag(s), V))
        A2 = (B + H) / 2
        A3 = (A2 + A2.T) / 2

        if self._isPD(A3):
            return A3

        spacing = np.spacing(np.linalg.norm(A))
        I = np.eye(A.shape[0])
        k = 1
        
        while not self._isPD(A3):
            mineig = np.min(np.real(np.linalg.eigvals(A3)))
            A3 += I * (-mineig * k**2 + spacing)
            k += 1

        return A3



    def _sqrt_func(self, x):

        try:
            result = np.linalg.cholesky(x)

        except np.linalg.LinAlgError:
            
            x = self._nearestPD(x)
            result = np.linalg.cholesky(x)

        return result



    def _metamodel_update_X(self, m1, m2, coupler, ts):

        ''' Return the analytical solution of coupling'''

        couple_idx_m1, couple_idx_m2 = self.coupling_graph.connect_idx['a_b']
        omega_1 = self.coupling_graph.omega[0]; omega_2 = self.coupling_graph.omega[0]
        fy_1 = m1[couple_idx_m1] 
        fy_2 = m2[couple_idx_m2]
        m1[couple_idx_m1] = (1-omega_1[ts])*coupler+omega_1[ts]*fy_1
        m2[couple_idx_m2] = (1-omega_2[ts])*coupler+omega_2[ts]*fy_2

        return np.array(list(m1)+list(m2))



    def _metamodel_update_P(self, ts):

        ''' Return the analytical solution of coupling'''

        couple_idx_m1, couple_idx_m2 = self.coupling_graph.connect_idx['a_b']
        epsilon = self.coupling_graph.coupling_variable[0][:,1]
        omega_1 = self.coupling_graph.omega[0]; omega_2 = self.coupling_graph.omega[0]
        phi_1 = self.coupling_graph.phi['a_b'][0]; phi_2 = self.coupling_graph.phi['a_b'][1]
        phi_1_ts = phi_1[ts]; phi_2_ts = phi_2[ts]
        P_meta = np.diag(list(np.diag(self.m1_Q[ts]))+list(np.diag(self.m2_Q[ts])))
        P_meta[couple_idx_m1,couple_idx_m1] = epsilon[ts]**2*(1-omega_1[ts])**2+phi_1_ts**2
        P_meta[couple_idx_m2+self.m1.n_state,couple_idx_m2+self.m1.n_state] = epsilon[ts]**2*(1-omega_2[ts])**2+phi_2_ts**2
        P_meta[couple_idx_m1,couple_idx_m2+self.m1.n_state] = epsilon[ts]**2*(1-omega_1[ts])*(1-omega_2[ts])
        P_meta[couple_idx_m2+self.m1.n_state,couple_idx_m1] = epsilon[ts]**2*(1-omega_1[ts])*(1-omega_2[ts])
        
        return P_meta



    def _fx_couple(self, state, dt, ts, coupler):

        predicted_m1 = self.m1.fx(state[:self.m1.n_state], self.m1.dt)
        predicted_m2 = self.m2.fx(state[-self.m2.n_state:], self.m2.dt)
        state_couple = self._metamodel_update_X(predicted_m1, predicted_m2, coupler, ts)

        return state_couple


    
    def _hx(self, x): 
        '''
        In all our cases, the observation functions are all defined as a Indentity matrix
        '''
        return x



    def inference(self, n_repeats, verbose=0):

        repeated_obs_m1_mean, repeated_obs_m1_std, repeated_obs_m2_mean, repeated_obs_m2_std = [],[],[],[]

        if verbose==1:
            print('******** Metamodel info ********')
            print('model_one_name: {}'.format(self.m1.modelname))
            print('total_time: {} {}'.format(self.m1.total_time, self.m1.unit))
            print('time_step: {} {}'.format(self.m1.dt, self.m1.unit))
            print('==========================')
            print('model_two_name:{}'.format(self.m2.modelname))
            print('total_time: {} {}'.format(self.m2.total_time, self.m2.unit))
            print('time_step: {} {}'.format(self.m2.dt, self.m2.unit))
            print('******** Metamodel info ********')
        
        print('-------- Run metamodel ---------')   

        for i in range(n_repeats):
            
            n_step = len(np.arange(0, max(self.m1.total_time, self.m2.total_time), min(self.m1.dt, self.m2.dt)))
            ''' here, self.m1.dt = self.m2.dt'''

            sigmas_meta = MerweScaledSigmaPoints(self.n_state, alpha=1e-3, beta=2., kappa=0., sqrt_method=self._sqrt_func)
            ukf_meta = UKF(dim_x=self.n_state, dim_z=self.n_state, fx=self._fx_couple, hx=self._hx, dt=self.m1.dt, points=sigmas_meta)
            ukf_meta.x = np.array(self.initial)
            ukf_meta.P *= self.initial_noise
  
            updated_m1, upd_std_m1, updated_m2, upd_std_m2 = [], [], [], []

            obs_m1 = np.random.normal(loc=self.m1_obs[:,:,0], scale=self.m1_obs[:,:,1])
            obs_m2 = np.random.normal(loc=self.m2_obs[:,:,0], scale=self.m2_obs[:,:,1])
                        
            for ts in range(n_step):

                time.sleep(1e-10)
                process_bar(i+1, n_repeats)
                                
                ukf_meta.Q = self.Q[ts]
                ukf_meta.R = self.R[ts]

                ukf_meta.predict(fx=self._fx_couple, dt=self.m1.dt, coupler=self.coupling_graph.coupling_variable[0][ts, 0], ts=ts)
                # update the predict function using metamodel version, update sigma points and weights at the same time
                ukf_meta.sigmas_f = ukf_meta.points_fn.sigma_points(ukf_meta.x, self._metamodel_update_P(ts=ts))
                _, ukf_meta.P = unscented_transform(ukf_meta.sigmas_f, ukf_meta.Wm, ukf_meta.Wc)
                ukf_meta.P_prior = np.copy(ukf_meta.P)

                obs_meta = np.concatenate((obs_m1[ts].reshape(-1,1), obs_m2[ts].reshape(-1,1)), axis=0).reshape(-1,)
                ukf_meta.update(obs_meta)

                meta_m1_mean = ukf_meta.x[:self.m1.n_state]
                meta_m1_cov = ukf_meta.P[:self.m1.n_state,:self.m1.n_state]
                meta_m1_std = marginal_from_joint(meta_m1_cov)
                updated_m1 += [meta_m1_mean]; upd_std_m1 += [meta_m1_std]

                meta_m2_mean = ukf_meta.x[-self.m2.n_state:]
                meta_m2_cov = ukf_meta.P[-self.m2.n_state:,-self.m2.n_state:]
                meta_m2_std = marginal_from_joint(meta_m2_cov)
                updated_m2 += [meta_m2_mean]; upd_std_m2 += [meta_m2_std]

            repeated_obs_m1_mean += [updated_m1]
            repeated_obs_m1_std += [upd_std_m1]
            repeated_obs_m2_mean += [updated_m2]
            repeated_obs_m2_std += [upd_std_m2]

        print('\n-------- Finished ---------')

        self.meta_m1_mean = np.mean(np.array(repeated_obs_m1_mean), axis=0)
        self.meta_m1_std = np.mean(np.array(repeated_obs_m1_std), axis=0) # this is not sure
        self.meta_m2_mean = np.mean(np.array(repeated_obs_m2_mean), axis=0)
        self.meta_m2_std = np.mean(np.array(repeated_obs_m2_std), axis=0) # this is not sure