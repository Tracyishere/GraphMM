'''
    Author: Chenxi Wang (chenxi.wang@salilab.org)
    Date: 2022-04-20
'''

import numpy as np
from GraphMetamodel.utils import *
from GraphMetamodel.DefineCouplingGraph import *
import GraphMetamodel.SequentialImportanceSampling_v3 as SIS
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints



def new_inference(model, start_point, new_state, filepath=None, n_repeat=1):    

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
        surrogate.P *= new_state[1]**2 # noise of initial state

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

        surrogate_mean += [updated]
        surrogate_std += [marginal_upd_std]

    if filepath is not None:
        output.close()

    return np.mean(np.array(surrogate_mean), axis=0), np.mean(np.array(surrogate_std), axis=0)




def make_metamodel(meta_state, coupling_graph, meta_obs_mean_ts, meta_obs_cov_ts, R_meta_ts, ts, joint_particles, particle_weights, test_omega):

    test_omega = [0.5, 0.5]

    model_dim = []
    for key in coupling_graph.model_idx:
        model = coupling_graph.model_idx[key]
        model_dim += [(coupling_graph.n_coupling_var+meta_state.index(model.state[0]), 
                       coupling_graph.n_coupling_var+meta_state.index(model.state[-1])+1)]
    
    # ======= state prediction =========
    n_particles, n_states = joint_particles.shape
    particles = joint_particles
    for p in range(n_particles):
        for num in range(coupling_graph.n_coupling_var):
            particles[p,num] = coupling_graph.coupling_variable[num][ts,0] + np.random.normal(0, coupling_graph.coupling_variable[num][ts,1])

        for i,key in enumerate(coupling_graph.model_idx):
            model = coupling_graph.model_idx[key]
            coupling_var_particles = [particles[p,k] for k in range(len(model.con_var_idx))]
            particles[p,model_dim[i][0]:model_dim[i][1]], particle_weights[p] = SIS.predict_each_model(particle=particles[p,model_dim[i][0]:model_dim[i][1]], 
                                                                                                weights=particle_weights[p],
                                                                                                model=model, n_state=model.n_state, 
                                                                                                connect_idx=model.con_var_idx, 
                                                                                                coupler=coupling_var_particles,
                                                                                                m_Q_ts=coupling_graph.meta_Q[key][ts], 
                                                                                                phi_ts=[con_phi[ts] for con_phi in model.con_phi], 
                                                                                                # omega_ts=[con_omega[ts] for con_omega in model.con_omega], 
                                                                                                omega_ts=test_omega,
                                                                                                units=model.con_unit_weight)                                                             
    # ======= state update =========
    particles, particle_weights = SIS.update_and_resample(particles, particle_weights, n_particles, 
                                                      meta_obs_mean_ts, meta_obs_cov_ts, R_meta_ts, 
                                                      model_dim, coupling_graph.n_coupling_var)                                                               

    return particles, particle_weights




class MetaModel:

    def __init__(self, coupling_graph):

        self.coupling_graph = coupling_graph
        self.n_model = len(self.coupling_graph.model_idx)
        self.meta_state = []
        for key in self.coupling_graph.model_idx:
            self.meta_state += self.coupling_graph.model_idx[key].state
        self.meta_n_state = len(self.meta_state)

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
            model_a.con_phi += [self.coupling_graph.phi[key][0]]
            model_a.con_omega += [self.coupling_graph.omega[num]] 
            model_a.con_unit_weight += [self.coupling_graph.unit_weight[num][0]]
            
            model_b.con_var_idx += [self.coupling_graph.connect_idx[key][1]]
            model_b.con_phi += [self.coupling_graph.phi[key][1]]
            model_b.con_omega += [self.coupling_graph.omega[num]] 
            model_b.con_unit_weight += [self.coupling_graph.unit_weight[num][1]]

    
    def _get_initial_meta_state(self):

        coupling_var_state_mean_t0 = [c[0,0] for c in self.coupling_graph.coupling_variable] 
        coupling_var_state_std_t0 = [c[0,1] for c in self.coupling_graph.coupling_variable] 

        model_var_state_mean_t0, model_var_state_std_t0 = [], []
        for key in self.coupling_graph.model_idx:
            s_model = self.coupling_graph.model_idx[key]
            # initial should be a list
            model_var_state_mean_t0 += list(s_model.initial)
            if isinstance(s_model.initial_noise, float):
                model_var_state_std_t0 += [s_model.initial_noise]*s_model.n_state
            else:
                 model_var_state_std_t0 += s_model.initial_noise            
           
        Meta_mean_t0 = np.array(coupling_var_state_mean_t0 + model_var_state_mean_t0)
        # by default, the initial covariance matrix of the metamodel is diagnal, unless additional info is given
        Meta_cov_t0 = np.diag(np.array(coupling_var_state_std_t0 + model_var_state_std_t0))**2

        return Meta_mean_t0, Meta_cov_t0


    def _get_meta_obs_ts(self, ts):

        Meta_obs_mean_ts, Meta_obs_std_ts, Meta_R_ts = [], [], []
        for key in self.coupling_graph.model_obs:
            model_obs = self.coupling_graph.model_obs[key]
            Meta_obs_mean_ts += model_obs[ts,:,0].tolist()
            Meta_obs_std_ts += model_obs[ts,:,1].tolist()
            Meta_R_ts += np.diag(self.coupling_graph.meta_R[key][ts,:,:]).tolist()

        Meta_obs_mean_ts = np.array(Meta_obs_mean_ts)
        Meta_obs_cov_ts = np.diag(np.array(Meta_obs_std_ts)**2)
        Meta_R_ts = np.diag(np.array(Meta_R_ts))  

        return Meta_obs_mean_ts, Meta_obs_cov_ts, Meta_R_ts


    
    def inference(self, n_particles, test_omega, filepath=None, verbose=1):

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
            # print(*['coupler','coupler']*self.coupling_graph.n_coupling_var+list(np.repeat(self.meta_state, 2)), file=output, sep=',')
        
        # ============= update coupling graph =============
        # self.coupling_graph = self.coupling_graph.get_upd_coupling_graph_multi_scale()
        Meta_mean_t0, Meta_cov_t0 = self._get_initial_meta_state()
        meta_mean, meta_std = [],[]

        if verbose==1:
            print('-------- Run metamodel ---------') 

        particles = np.random.multivariate_normal(Meta_mean_t0, Meta_cov_t0, n_particles)
        particle_weights = np.ones(n_particles)/n_particles
       
        for ts in range(self.max_n_step):

            if ts<=800: # this is because model one run more time
                meta_obs_mean_ts, meta_obs_cov_ts, R_meta_ts = self._get_meta_obs_ts(ts=ts)
                particles, particle_weights = make_metamodel(self.meta_state, self.coupling_graph, meta_obs_mean_ts, meta_obs_cov_ts, R_meta_ts, 
                                                             ts, particles, particle_weights, test_omega)
                Meta_mean_ts, Meta_cov_ts = SIS.estimate(n_particles, particles, particle_weights)
                print(*list(Meta_mean_ts), file=output, sep=',')
                for line in Meta_cov_ts:
                    print(*list(line), file=output, sep=',')
                Meta_std_ts = np.sqrt(Meta_cov_ts)

                # for var in range(self.coupling_graph.n_coupling_var):
                #     self.upd_coupling_var += [[Meta_mean_ts[var], Meta_std_ts[var]]]

            else:
                continue

            # if filepath is not None:
                # this print is for all steps
                # print(*list(chain.from_iterable(zip(Meta_mean_ts, Meta_std_ts))), file=output, sep=',')

            meta_mean += [Meta_mean_ts]
            meta_std += [Meta_std_ts]

            if verbose==1:
                time.sleep(1e-20)
                process_bar(ts+1, self.max_n_step)

        self.upd_coupling_var = np.array(self.upd_coupling_var)
        self.meta_mean = np.array(meta_mean)
        self.meta_std = np.array(meta_std)
        
        if filepath is not None:
            output.close()

        print('\n-------- Finished ----------')