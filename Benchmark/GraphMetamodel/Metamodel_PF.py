'''
    Author: Chenxi Wang (chenxi.wang@salilab.org)
    Date: 2021-11-20

    This script is for the alternating inference method for two adjacent time slice when coupling. 
    We try 2 methods, 1) particle filter 2) alternate updating

    The following is only for the coupling of the toy example in the tutorial.
'''


import numpy as np
from MetaModel.utils import *
import MetaModel.SequentialImportanceSampling as SIS
from scipy.stats import norm
import numpy as np

np.random.seed(2) 
sys.float_info.min


class MetaModel:

    def __init__(self, coupling_graph):

        self.m1 = coupling_graph.m1
        self.m2 = coupling_graph.m2
        self.coupling_graph = coupling_graph
        self.meta_state = self.m1.state + self.m2.state
        self.meta_n_state = len(self.meta_state)
        self.meta_n_step = max(self.m1.n_step,self.m2.n_step)
        
        ''' initial states '''
        self.coupler = [self.coupling_graph.coupling_variable[0]]
        self.meta_m1_mean = [np.array(self.m1.initial)]
        self.meta_m1_std = [np.array([self.m1.initial_noise]*self.m1.n_state)]
        self.meta_m2_mean = [np.array(self.m2.initial)]
        self.meta_m2_std = [np.array([self.m1.initial_noise]*self.m2.n_state)]


    def inference(self, n_particles, verbose=1):

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
        
        if verbose==1:
            print('-------- Run metamodel ---------') 
        
        Mmeta_t0_mean = np.array([self.coupling_graph.coupling_variable[0][0]]+list(self.meta_m1_mean[0])+list(self.meta_m2_mean[0]))
        Mmeta_t0_cov = np.diag(np.array([self.coupling_graph.coupling_variable[0][1]]+list(self.meta_m1_std[0])+list(self.meta_m2_std[0])))
        particles = np.random.multivariate_normal(Mmeta_t0_mean, Mmeta_t0_cov, n_particles)
        particle_weights = np.ones(n_particles)/n_particles

        for ts in range(self.meta_n_step):

            particles, particle_weights = SIS.particle_sampling(self.coupling_graph, particles, particle_weights, ts)
            Mmeta_t_mean, Mmeta_t_cov = SIS.estimate(n_particles, particles, particle_weights)
            Mmeta_t_std = np.sqrt(Mmeta_t_cov)
            self.coupler += [[Mmeta_t_mean[0], Mmeta_t_std[0]]]
            self.meta_m1_mean += [Mmeta_t_mean[1:1+self.m1.n_state]]
            self.meta_m1_std += [Mmeta_t_std[1:1+self.m1.n_state]]
            self.meta_m2_mean += [Mmeta_t_mean[1+self.m1.n_state:]]
            self.meta_m2_std += [Mmeta_t_std[1+self.m1.n_state:]]

            if verbose==1:
                time.sleep(1e-10)
                process_bar(ts+1, self.meta_n_step)

        self.coupler = np.array(self.coupler)[1:]
        self.meta_m1_mean = np.array(self.meta_m1_mean)[1:]; self.meta_m1_std = np.array(self.meta_m1_std)[1:]
        self.meta_m2_mean = np.array(self.meta_m2_mean)[1:]; self.meta_m2_std = np.array(self.meta_m2_std)[1:]
        
        print('\n-------- Finished ----------')