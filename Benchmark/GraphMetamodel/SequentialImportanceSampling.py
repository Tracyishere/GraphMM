
'''
    Author: Chenxi Wang (chenxi.wang@salilab.org)
    Date: 2022-04-20
'''

import numpy as np
from GraphMetamodel.utils import *
from scipy.stats import norm
import numpy as np
import random


np.random.seed(2) 
sys.float_info.min


def effective_n(weights):
    
    '''
    Calculates effective N, an approximation for the number of particles contributing meaningful information determined by their weight. 
    When this number is small, the particle filter should resample the particles to redistribute the weights among more particles.
    '''
    
    return 1. / np.sum(np.square(weights))
    
    
def resample(weights):

    '''
    We desire an algorithm that has several properties. 
    It should preferentially select particles that have a higher probability. 
    It should select a representative population of the higher probability particles to avoid sample impoverishment. 
    It should include enough lower probability particles to give the filter a chance of detecting strongly nonlinear behavior.
    '''
    
    N = len(weights)
    random_offset = random.random()
    random_partition = [(x + random_offset)/N for x in list(range(N))]
    cumulative_sum = np.cumsum(weights)
    particle_indexes = np.zeros(N, 'i')

    i, j = 0, 0
    while (i < N) and (j < N):
        if random_partition[i] < cumulative_sum[j]:
            particle_indexes[i] = j
            i += 1 
        else:
            j += 1  
            
    return particle_indexes


def simple_resample(particles, weights):
    
    N = len(particles)
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1. # avoid round-off error
    indexes = np.searchsorted(cumulative_sum, random.random())

    # resample according to indexes
    particles[:] = particles[indexes]
    weights = weights.fill(1.0 / N)
    
    return particles, weights


def systematic_resample(weights):
    
    N = len(weights)

    # make N subdivisions, choose positions with a consistent random offset
    positions = (np.arange(N) + random.random()) / N
    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
            
    return indexes
    

def compute_likelihood_each_model(sample, measurement, measurement_noise):

    dm = sample-measurement
    dm = dm.reshape(-1,1)
    p_o_given_m = np.exp(-1/2 * np.dot(np.dot(dm.T, np.linalg.inv(measurement_noise)), dm))

    return p_o_given_m


def compute_likelihood(sample, measurement, measurement_noise, model_dim, n_coupler):

    ''' Here, we assume the likelihood for each state in each model is independent and gaussian.  '''
    
    likelihood_sample = 1.0

    for i in range(len(model_dim)):
        
        mi_idx0 = model_dim[i][0]
        mi_idx1 = model_dim[i][1]
        sample_mi = sample[mi_idx0:mi_idx1]
        measurement_mi = measurement[mi_idx0-n_coupler:mi_idx1-n_coupler]
        measurement_noise_mi = measurement_noise[mi_idx0-n_coupler:mi_idx1-n_coupler, mi_idx0-n_coupler:mi_idx1-n_coupler]
        p_o_given_mi = compute_likelihood_each_model(sample_mi, measurement_mi, measurement_noise_mi)

        likelihood_sample *= p_o_given_mi

    return likelihood_sample


def update(particles, weights, observation, observation_noise, model_dim, n_coupler):

    for i, par in enumerate(particles):
        weights[i] *= compute_likelihood(par, observation[i], observation_noise, model_dim, n_coupler)
        
    # Check if weights are non-zero
    if sum(weights) < 1e-20:
        print("\n Weight normalization failed: sum of all weights is {} (weights will be reinitialized)".format(sum(weights)))
        weights = np.array([1.0/len(weights)]*len(weights))
    
    weights += 1.e-300 # avoid round-off to zero
    weights /= sum(weights) # normalize
    
    return weights


def estimate(n_particles, particles, weights):
    
    ''' returns mean and variance of the weighted particles,  
        we should return the joint prob here, we assume the variables are independent '''

    mean = np.average(particles, weights=weights, axis=0)
    weighted_par = np.array([weights[i]*particles[i] for i in range(n_particles)])
    cov = np.cov(weighted_par, rowvar=False)  
    # cov = np.average((particles - mean)**2, weights=weights, axis=0)

    return mean, cov


def predict_each_model(particle, weights, model, n_state, connect_idx, coupler, m_Q_ts, phi_ts, omega_ts, units):

    if (particle > 0).all():
        try: 
            fy_model = model.fx(np.array(particle), model.dt)
        except RuntimeWarning:
            weights = 1e-300

        for idx in range(n_state):
            if idx in connect_idx:
                k = connect_idx.index(idx)
                particle[idx] = omega_ts[k]*fy_model[idx]*units[k] + (1-omega_ts[k])*coupler[k] + np.random.normal(0,phi_ts[k]*units[k])
                particle[idx] /= units[k]
            else:
                particle[idx] = fy_model[idx] + np.random.normal(0,np.diag(m_Q_ts)[idx])
    else:
         weights = 1e-300
                
    return particle, weights


def update_and_resample(particles, weights, n_particles, obs_mean_ts, obs_cov_ts, R_meta_ts, model_dim, n_coupler):

    observation_ts = np.random.multivariate_normal(obs_mean_ts, obs_cov_ts, n_particles)
    weights = update(particles, weights, observation_ts, R_meta_ts, model_dim, n_coupler)

    if effective_n(weights) < n_particles/2:
        indexes = systematic_resample(weights)
        particles[:] = particles[indexes]
        weights[:] = weights[indexes]

    return particles, weights