'''
    Author: Chenxi Wang(chenxi.wang@salilab.org)
    Date: 2021-07-20
'''

from GraphMetamodel.utils import *
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
<<<<<<< Updated upstream
import scipy.stats as stats
from itertools import chain


class SurrogateInputModel:

    def __init__(self, name, state, initial, initial_noise, fx, Q, R, obs, dt, total_time, unit):   
=======
from itertools import chain
import random

np.random.seed(121)
random.seed(121)

class SurrogateInputModel:

    def __init__(self, name, state, initial, initial_noise_scale, fx, dt, input_dt, total_time, 
                       measure_std_scale, transition_cov_scale, emission_cov_scale, noise_model_type, unit):   
>>>>>>> Stashed changes

        self.modelname = name
        self.state = state
        self.n_state = len(self.state)
        self.initial = initial
<<<<<<< Updated upstream
        self.initial_noise = initial_noise
        self.fx = fx
        self.Q = np.array(Q)
        self.R = np.array(R)
        self.obs = obs
        self.dt = dt
        self.total_time = total_time
        self.unit = unit
        self.n_step = len(np.arange(0,self.total_time,self.dt))
        self.mean = 0
        self.std = 0
=======
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
>>>>>>> Stashed changes


    def hx(self, x): 
        '''
        In all our cases, the observation functions are all defined as a Indentity matrix
        '''
        return x


<<<<<<< Updated upstream
=======
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
            Q_ts = [np.diag(abs(z_mean[i]*random.gauss(self.transition_cov_scale, self.transition_cov_scale*1e-2))) for i in range(n_step)]
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
            R_ts = [np.diag(abs(x_mean[i]*random.gauss(self.emission_cov_scale, self.emission_cov_scale*1e-2))) for i in range(n_step)]
        else:
            print('ERROR: Unknown noise model type.')

        return np.array(R_ts)


    def get_obs_ts(self, x0, filepath=None):
        
        obs_t_interval = get_observations(x0, self.fx, self.dt, self.dt, self.measure_std_scale)
        obs_temp = np.random.normal(loc=obs_t_interval[:, :,0], scale=obs_t_interval[:, :,1])
                
        return obs_temp


    def inference_for_ts(self, x_ts, P_ts):

        sigmas = MerweScaledSigmaPoints(self.n_state, alpha=1e-3, beta=2., kappa=0.)
        surrogate_ts = UKF(dim_x=self.n_state, dim_z=self.n_state, fx=self.fx, hx=self.hx, dt=self.dt, points=sigmas)
        surrogate_ts.x = x_ts 
        surrogate_ts.P = P_ts
        surrogate_ts.Q = self.get_Q_ts(x_ts)[0]
        obs_ts = self.get_obs_ts(x_ts)[0]
        surrogate_ts.R = self.get_R_ts(obs_ts)[0]
        surrogate_ts.predict(dt=self.dt, fx=self.fx)
        surrogate_ts.update(obs_ts)

        return surrogate_ts


>>>>>>> Stashed changes
    def inference(self, filepath=None, n_repeat=1, verbose=1):    
        
        '''
        The output 'predicted' here is $P(z_t|O_{1:t-1})$,
        The output 'updated' here is $P(z_t|O_{1:t})$
        '''
<<<<<<< Updated upstream
        
        surrogate_mean, surrogate_std = [], []
        
        if filepath is not None:
            output = open(filepath, 'w')
            print(*list(np.repeat(self.state, 2)), file=output, sep=',')
=======
                
        # avg_state_var = ['V', 'n', 's', 'Ca']
        if filepath is not None:
            output = open(filepath, 'w')
            print(*list(np.repeat(self.state, 2)), file=output, sep=',')
            # print(*list(np.repeat(avg_state_var, 2)), file=output, sep=',')
>>>>>>> Stashed changes

        if verbose == 1:
            print('******** Metamodel info ********')
            print('model_name: {}'.format(self.modelname))
            print('total_sim_time: {} {}'.format(self.total_time, self.unit))
            print('time_step: {} {}'.format(self.dt, self.unit))
            print('num_repeat: {}'.format(n_repeat))
            print('******** Metamodel info ********')
            print('-------- Run surrogate model ---------')

<<<<<<< Updated upstream

        for k in range(n_repeat): # repeat for different observations

            obs_temp = np.random.normal(loc=self.obs[:,:,0], scale=self.obs[:,:,1])

            sigmas = MerweScaledSigmaPoints(self.n_state, alpha=1e-3, beta=2., kappa=0.)
            surrogate = UKF(dim_x=self.n_state, dim_z=self.n_state, fx=self.fx, hx=self.hx, dt=self.dt, points=sigmas)
            surrogate.x = np.array([self.initial]) # initial state
            surrogate.P *= self.initial_noise # noise of initial state

            updated, marginal_upd_std = [], []    
            
            for i,z in enumerate(obs_temp):
                
                surrogate.Q = self.Q[i]
                surrogate.R = self.R[i]

                surrogate.predict(dt=self.dt, fx=self.fx)
                surrogate.update(z)
                
                mean = surrogate.x
                updated += [mean]
                std = marginal_from_joint(surrogate.P)
                marginal_upd_std += [std]

                if filepath is not None:
                    print(*list(chain.from_iterable(zip(mean, std))), file=output, sep=',')

                if verbose == 1:
                    time.sleep(1e-20)
                    process_bar((k+1)*(i+1), n_repeat*len(obs_temp))

        if verbose == 1:
            print('\n -------- Finished ---------')

            surrogate_mean += [updated]
            surrogate_std += [marginal_upd_std]

        if filepath is not None:
            output.close()

        self.mean = np.mean(np.array(surrogate_mean), axis=0)
        self.std = np.mean(np.array(surrogate_std), axis=0)

   

    def new_inference(model, start_point, new_state, filepath=None, n_repeat=1, verbose=1):    
        
        '''
        The output 'predicted' here is $P(z_t|O_{1:t-1})$,
        The output 'updated' here is $P(z_t|O_{1:t})$
        '''
        
        surrogate_mean, surrogate_std = [], []
        
        if filepath is not None:
            output = open(filepath, 'w')
            print(*list(np.repeat(model.state, 2)), file=output, sep=',')

        for k in range(n_repeat): # repeat for different observations

            obs_temp = np.random.normal(loc=model.obs[start_point:,:,0], scale=model.obs[start_point:,:,1])

            sigmas = MerweScaledSigmaPoints(model.n_state, alpha=1e-3, beta=2., kappa=0.)
            surrogate = UKF(dim_x=model.n_state, dim_z=model.n_state, fx=model.fx, hx=model.hx, dt=model.dt, points=sigmas)
            surrogate.x = np.array([new_state[0]]) # initial state
            surrogate.P *= new_state[1] # noise of initial state

            updated, marginal_upd_std = [], []    
            
            for i,z in enumerate(obs_temp):
                
                surrogate.Q = model.Q[i]
                surrogate.R = model.R[i]

                surrogate.predict(dt=model.dt, fx=model.fx)
                surrogate.update(z)
                
                mean = surrogate.x
                updated += [mean]
                std = marginal_from_joint(surrogate.P)
                marginal_upd_std += [std]

                if filepath is not None:
                    print(*list(chain.from_iterable(zip(mean, std))), file=output, sep=',')

        if verbose == 1:
            print('\n -------- Finished ---------')

            surrogate_mean += [updated]
            surrogate_std += [marginal_upd_std]

        if filepath is not None:
            output.close()

        return np.mean(np.array(surrogate_mean), axis=0), np.mean(np.array(surrogate_std), axis=0)
=======
        for k in range(n_repeat): # repeat for different observations

            x_ts = self.initial
            P_ts = self.initial_noise

            for t in range(self.n_step):

                surrogate_ts = self.inference_for_ts(x_ts, P_ts)
                x_ts = surrogate_ts.x
                P_ts = surrogate_ts.P

                mean_ts = surrogate_ts.x
                std_ts = marginal_from_joint(surrogate_ts.P)
                
                if filepath is not None:
                    print(*list(chain.from_iterable(zip(mean_ts, std_ts))), file=output, sep=',')
                    # output_mean_ts = [np.mean(mean_ts[:57]), np.mean(mean_ts[57:2*57]), np.mean(mean_ts[2*57:3*57]), np.mean(mean_ts[3*57:])]
                    # output_std_ts = [np.mean(std_ts[:57]), np.mean(std_ts[57:2*57]), np.mean(std_ts[2*57:3*57]), np.mean(std_ts[3*57:])]
                    # print(*list(chain.from_iterable(zip(output_mean_ts, output_std_ts))), file=output, sep=',')
                elif self.input_dt < self.dt:
                    print('ERROR: Input model dt is smaller than the surrogate model.')

                if verbose == 1:
                    time.sleep(1e-20)
                    process_bar((k+1)*(t+1), n_repeat*self.n_step)

        if verbose == 1:
            print('\n-------- Finished ---------')

        if filepath is not None:
            output.close()
>>>>>>> Stashed changes
