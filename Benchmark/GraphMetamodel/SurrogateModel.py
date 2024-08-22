'''
    Author: Chenxi Wang(chenxi.wang@salilab.org)
    Date: 2021-07-20
'''

from GraphMetamodel.utils import *
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
import scipy.stats as stats
from itertools import chain


class SurrogateInputModel:

    def __init__(self, name, state, initial, initial_noise, fx, Q, R, obs, dt, total_time, unit):   

        self.modelname = name
        self.state = state
        self.n_state = len(self.state)
        self.initial = initial
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


    def hx(self, x): 
        '''
        In all our cases, the observation functions are all defined as a Indentity matrix
        '''
        return x


    def inference(self, filepath=None, n_repeat=1, verbose=1):    
        
        '''
        The output 'predicted' here is $P(z_t|O_{1:t-1})$,
        The output 'updated' here is $P(z_t|O_{1:t})$
        '''
        
        surrogate_mean, surrogate_std = [], []
        
        if filepath is not None:
            output = open(filepath, 'w')
            print(*list(np.repeat(self.state, 2)), file=output, sep=',')

        if verbose == 1:
            print('******** Metamodel info ********')
            print('model_name: {}'.format(self.modelname))
            print('total_sim_time: {} {}'.format(self.total_time, self.unit))
            print('time_step: {} {}'.format(self.dt, self.unit))
            print('num_repeat: {}'.format(n_repeat))
            print('******** Metamodel info ********')
            print('-------- Run surrogate model ---------')


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