'''
    Author: Chenxi Wang(chenxi.wang@salilab.org)
    Date: 2021-07-20
'''

from GraphMetamodel.utils import *
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from itertools import chain
import time
import glob
import os

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
        

    def _get_obs_from_input(self, obs_from_input):
        if os.path.isfile(obs_from_input):
            # Single cell file method
            self.obs_from_input = np.genfromtxt(obs_from_input, delimiter=' ')
        elif os.path.isdir(obs_from_input):
            # Multiple cell files method
            inputfiles = glob.glob(os.path.join(obs_from_input, '*.csv'))
            sorted_inputfiles = [os.path.join(obs_from_input, f'input_ICN_cell_{i}.csv') for i in range(57)]
            obs = []
            for f in sorted_inputfiles:
                f_obs = np.genfromtxt(f, delimiter=',', skip_header=1)
                obs.append(f_obs)

            stacked_arrays = np.stack(obs)
            transposed_arrays = np.transpose(stacked_arrays, (1, 0, 2))  # (100, 57, 4)
            self.obs_from_input = np.empty((len(transposed_arrays), self.n_state))
            for i in range(4):
                self.obs_from_input[:, i*57:(i+1)*57] = transposed_arrays[:, :, i]
        else:
            raise ValueError("Invalid input: obs_from_input must be a file or directory")

    def hx(self, x): 
        '''
        In all our cases, the observation functions are all defined as a Indentity matrix
        '''
        return x
    
    def is_positive_definite(self, matrix):

        eigvals = np.linalg.eigvals(matrix)

        return np.all(eigvals > 0)
    
    def svd_sqrt(self, matrix):

        U, s, V = np.linalg.svd(matrix)
        D = np.diag(np.sqrt(s))

        return U @ D @ V
    

    def make_positive_definite(self, matrix):
        
        min_eig = np.min(np.real(np.linalg.eigvals(matrix)))

        if min_eig <= 0:

            identity = np.eye(matrix.shape[0])

            increment = abs(min_eig) + 1e-10

            matrix += increment * identity

        return matrix


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
            R_ts = [np.diag(abs(x_mean[i]*self.emission_cov_scale)) for i in range(n_step)]
        else:
            print('ERROR: Unknown noise model type.')

        return np.array(R_ts)


    # def get_obs_ts(self, x0, time):

    #     obs_t_interval = get_observations_fx_with_one_step(x0, self.fx, self.dt, time, self.measure_std_scale)
    #     obs_temp = np.random.normal(loc=obs_t_interval[:, :,0], scale=obs_t_interval[:, :,1])
        
    #     return obs_temp[0]
    

    def get_obs_ts(self, x0, time):
        
        return self.obs_from_input[time]


    def inference_for_ts(self, x_ts, P_ts, ts):

        sigmas = MerweScaledSigmaPoints(self.n_state, alpha=1e-3, beta=2., kappa=0.)
        surrogate_ts = UKF(dim_x=self.n_state, dim_z=self.n_state, fx=self.fx, hx=self.hx, dt=self.dt, points=sigmas)
        surrogate_ts.x = x_ts 
        surrogate_ts.P = P_ts
        surrogate_ts.Q = self.get_Q_ts(x_ts)[0]
        obs_ts = self.get_obs_ts(x_ts, ts)
        surrogate_ts.R = self.get_R_ts(obs_ts)[0]
        surrogate_ts.predict(dt=self.dt, fx=self.fx, time=ts)
        surrogate_ts.update(obs_ts)

        return surrogate_ts, obs_ts


    def inference(self, obs_from_input, filepath=None, n_repeat=1, verbose=1):    
        
        '''
        The output 'predicted' here is $P(z_t|O_{1:t-1})$,
        The output 'updated' here is $P(z_t|O_{1:t})$
        '''
                
        avg_state_var = ['V', 'n', 's', 'Ca']
        
        if filepath is not None:
            # output = open(filepath, 'w')
            # print(*list(np.repeat(self.state, 2)), file=output, sep=',')
            # print(*list(np.repeat(avg_state_var, 2)), file=output, sep=',')
            for i in range(57):
                exec("output_{} = open(filepath+'surrogate_ICN_cell_{}.csv', 'w')".format(i,i))
                exec("print(*list(np.repeat(avg_state_var, 2)), file=output_{}, sep=',')".format(i))

        # if filepath is not None:
        #     for i in range(57):
        #         exec("obs_output_{} = open(filepath+'obs_surrogate_ICN_cell_{}.csv', 'w')".format(i,i))
        #         exec("print(*list(np.repeat(avg_state_var, 2)), file=obs_output_{}, sep=',')".format(i))

        if verbose == 1:
            print('******** Metamodel info ********')
            print('model_name: {}'.format(self.modelname))
            print('total_sim_time: {} {}'.format(self.total_time, self.unit))
            print('dt: {} {}'.format(self.dt, self.unit))
            print('steps: {}'.format(self.n_step))
            print('num_repeat: {}'.format(n_repeat))
            print('******** Metamodel info ********')
            print('-------- Run surrogate model ---------')

        self._get_obs_from_input(obs_from_input)

        for k in range(n_repeat): # repeat for different observations

            x_ts = self.initial
            P_ts = self.initial_noise

            for t in range(self.n_step):
                
                surrogate_ts, obs_ts = self.inference_for_ts(x_ts, P_ts, ts=t)
                x_ts = surrogate_ts.x
                if not self.is_positive_definite(surrogate_ts.P):
                    P_ts = self.make_positive_definite(surrogate_ts.P)
                else:
                    P_ts = surrogate_ts.P

                mean_ts = surrogate_ts.x
                # std_ts = marginal_from_joint(surrogate_ts.P)
                std_ts = np.sqrt(np.diag(surrogate_ts.P))
                
                if filepath is not None:
                    # output_mean_ts = [np.mean(mean_ts[:57]), np.mean(mean_ts[57:2*57]), np.mean(mean_ts[2*57:3*57]), np.mean(mean_ts[3*57:])]
                    # output_std_ts = [np.mean(std_ts[:57]), np.mean(std_ts[57:2*57]), np.mean(std_ts[2*57:3*57]), np.mean(std_ts[3*57:])]
                    for i in range(57):
                        output_mean_ts = [mean_ts[i], mean_ts[i+57], mean_ts[i+57*2], mean_ts[i+57*3]]
                        output_std_ts = [std_ts[i], std_ts[i+57], std_ts[i+57*2], std_ts[i+57*3]]
                        exec("print(*list(chain.from_iterable(zip(output_mean_ts, output_std_ts))), file=output_{}, sep=',')".format(i))
                        # obs_cell_ts = [obs_ts[i], obs_ts[i+57], obs_ts[i+57*2], obs_ts[i+57*3]]
                        # exec("print(*obs_cell_ts, file=obs_output_{}, sep=',')".format(i))
                elif self.input_dt < self.dt:
                    print('ERROR: Input model dt is smaller than the surrogate model.')

                if verbose == 1:
                    time.sleep(1e-100)
                    process_bar((k+1)*(t+1), n_repeat*self.n_step)

        if verbose == 1:
            print('\n-------- Finished ---------')

        if filepath is not None:
            for i in range(57):
                exec("output_{}.close()".format(i))