'''
    Author: Chenxi Wang(chenxi.wang@salilab.org)
    Date: 2021-07-20
'''

from GraphMetamodel.utils import *
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from itertools import chain


class SurrogateInputModel:

    def __init__(self, name, state, initial, initial_noise_scale, fx, dt, input_dt, total_time, 
                       measure_std_scale, transition_cov_scale, emission_cov_scale, noise_model_type, unit):   

        self.modelname = name
        self.state = state
        self.n_state = len(self.state)
        self.initial = initial
        self.initial_noise = initial_noise_scale*abs(np.diag(initial))
        self.fx = fx
        self.dt = dt
        self.input_dt = input_dt
        self.total_time = total_time
        self.unit = unit
        self.n_step = len(np.arange(0,self.total_time,self.dt))
        self.measure_std_scale = measure_std_scale
        self.transition_cov_scale = transition_cov_scale
        self.emission_cov_scale = emission_cov_scale
        self.noise_model_type = noise_model_type


    def hx(self, x): 
        '''
        In all our cases, the observation functions are all defined as a Indentity matrix
        '''
        return x


    def get_Q_ts(self, z_mean, n_step=1):

        '''
        get the transition noise matrix to approximate a transition noise model, 
        '''

        Q_ts = []
        z_mean = np.array(z_mean).reshape(n_step, -1)

        if self.noise_model_type == 'time-invariant':
            Q_ts = [np.diag(abs(np.mean(z_mean, axis=0)*self.transition_cov_scale)) for i in range(n_step)]

        elif self.noise_model_type == 'time-variant':
            ''' time-invariant means the noise model is relavent to the value of state variable'''
            # Q_ts = [np.diag(abs(z_mean[i]*random.gauss(self.transition_cov_scale, self.transition_cov_scale*1e-2))) for i in range(n_step)]
            Q_ts = [np.diag(abs(z_mean[i]*self.transition_cov_scale)) for i in range(n_step)]
        else:
            print('ERROR: Unknown noise model type.')

        return np.array(Q_ts) 

    
    def get_R_ts(self, x_mean, n_step=1):

        '''
        get the emission noise matrix to approximate a emission noise model
        '''

        R_ts = []
        x_mean = np.array(x_mean).reshape(n_step, -1)

        if self.noise_model_type == 'time-invariant':
            R_ts = [np.diag(abs(np.mean(x_mean, axis=0)*self.emission_cov_scale)) for i in range(n_step)]

        elif self.noise_model_type == 'time-variant':
            # R_ts = [np.diag(abs(x_mean[i]*random.gauss(self.emission_cov_scale, self.emission_cov_scale*1e-2))) for i in range(n_step)]
            R_ts = [np.diag(abs(x_mean[i]*self.emission_cov_scale)) for i in range(n_step)]
        else:
            print('ERROR: Unknown noise model type.')

        return np.array(R_ts)


    def get_obs_ts(self, x0, time):
        
        obs_t_interval = get_observations_fx_with_one_step(x0, self.fx, self.dt, time, self.measure_std_scale)
        obs_temp = np.random.normal(loc=obs_t_interval[:, :,0], scale=obs_t_interval[:, :,1])
                
        return obs_temp[0]


    def inference_for_ts(self, x_ts, P_ts, time, V_mem):

        sigmas = MerweScaledSigmaPoints(self.n_state, alpha=1e-3, beta=2., kappa=0.)
        surrogate_ts = UKF(dim_x=self.n_state, dim_z=self.n_state, fx=self.fx, hx=self.hx, dt=self.dt, points=sigmas)
        surrogate_ts.x = x_ts 
        surrogate_ts.P = P_ts
        surrogate_ts.Q = self.get_Q_ts(x_ts)[0]
        # obs_ts = self.get_obs_ts(x_ts, time)
        obs_ts = x_ts
        surrogate_ts.R = 100*self.get_R_ts(obs_ts)[0]
        surrogate_ts.predict(dt=self.dt, fx=self.fx, time=time, V_mem=V_mem)
        surrogate_ts.update(obs_ts)

        return surrogate_ts, obs_ts


    def inference(self, V_mem, obs_filepath=None, output_filepath=None, n_repeat=1, verbose=1):    
        
        '''
        The output 'predicted' here is $P(z_t|O_{1:t-1})$,
        The output 'updated' here is $P(z_t|O_{1:t})$
        '''

        if obs_filepath is not None:
            f_obs = open(obs_filepath, 'w')
                
        if output_filepath is not None:
            output = open(output_filepath, 'w')
            print(*list(np.repeat(self.state, 2)), file=output, sep=',')

        if verbose == 1:
            print('******** Surrogate model info ********')
            print('model_name: {}'.format(self.modelname))
            print('total_sim_time: {} {}'.format(self.total_time, self.unit))
            print('dt: {} {}'.format(self.dt, self.unit))
            print('step: {}'.format(self.n_step))
            print('num_repeat: {}'.format(n_repeat))
            print('******** Surrogate model info ********')
            print('-------- Run surrogate model ---------')

        meta_mean, meta_std = [], []

        for k in range(n_repeat): # repeat for different observations

            meta_mean_r, meta_std_r = [], []

            x_ts = self.initial
            P_ts = self.initial_noise

            for t in range(self.n_step):

                surrogate_ts, obs_ts = self.inference_for_ts(x_ts, P_ts, t, V_mem)
                x_ts = surrogate_ts.x
                P_ts = surrogate_ts.P

                mean_ts = surrogate_ts.x
                std_ts = marginal_from_joint(surrogate_ts.P)
                # std_ts = np.sqrt(np.diag(surrogate_ts.P))
                
                if obs_filepath is not None:
                    print(*list(obs_ts), file=f_obs, sep=',')

                meta_mean_r += [mean_ts]
                meta_std_r += [std_ts]

                if verbose == 1:
                    time.sleep(1e-30)
                    process_bar((k+1)*(t+1), n_repeat*self.n_step)

            meta_mean += [np.array(meta_mean_r)]
            meta_std += [np.array(meta_std_r)]

        if verbose == 1:
            print('\n-------- Finished ---------') 

        if output_filepath is not None:

            meta_mean = np.mean(np.array(meta_mean), axis=0)
            print(meta_mean.shape)
            meta_std = np.mean(np.array(meta_std), axis=0)
            print(meta_std.shape)

            for i in range(len(meta_mean)):
              print(*list(chain.from_iterable(zip(meta_mean[i], meta_std[i]))), file=output, sep=',')

            output.close()

        if obs_filepath is not None:
            f_obs.close()