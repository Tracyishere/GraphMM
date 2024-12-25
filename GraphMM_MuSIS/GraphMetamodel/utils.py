
''' Help functions for metamodeling. '''

import numpy as np
import sys,time
from numpy.linalg.linalg import LinAlgError
import warnings


def process_bar(num, total):
    
    rate = float(num)/total
    ratenum = int(100*rate)
    r = '\r[{}{}]{}%'.format('/'*ratenum,' '*(100-ratenum), ratenum)
    sys.stdout.write(r)
    sys.stdout.flush()


def get_observations(X, fx, dt, total_time, measure_std, file):
    
    X = [np.array(X)] 
    sim_time = np.arange(0, total_time, dt)
    for t in range(len(sim_time)): 
        temp = fx(X[t], dt)
        X += [temp]
        if file is not None:
            print(*list(temp), file=file, sep=',')

    obs_mean = np.array(X[1:])
    obs_std = abs(measure_std*np.array(X[1:]))

    return np.dstack((obs_mean, obs_std))


def get_observations_fx_with_one_step(X, fx, dt, time, measure_std, Vm=None):
    
    if Vm is None:
        if time==-1:
            temp = fx(np.array(X), dt)
        else:
            temp = fx(np.array(X), dt, time)
        obs_mean = np.array(temp)
        obs_std = abs(measure_std*obs_mean)
    else:
        if time==-1:
            temp = fx(np.array(X), dt, Vm)
        else:
            temp = fx(np.array(X), dt, time, Vm)
        obs_mean = np.array(temp)
        obs_std = abs(measure_std*obs_mean)

    return np.dstack((obs_mean, obs_std))


def marginal_from_joint(cov):
            
        '''
        The outcome of the package is joint probability which needs to be marginalized out to get marginal results.
        '''
        
        N = cov.shape[0]
        cov_list = []
        try:
            pre_mat = np.linalg.pinv(cov)
        
        except LinAlgError:
            
            warnings.warn('Fail to get marginal probability')
            
            return np.sqrt(np.diag(cov)).reshape(-1,)
        
        final_std = []
        
        for k in range(N):
            pre_aa = pre_mat[:k, :k]
            pre_ab = pre_mat[:k, k:]
            pre_ba = pre_mat[k:, :k]
            pre_bb = pre_mat[k:, k:]

            cov_a = np.linalg.pinv(pre_aa - np.dot(np.dot(pre_ab,np.linalg.pinv(pre_bb)),pre_ba))
            cov_b = np.linalg.pinv(pre_bb - np.dot(np.dot(pre_ba,np.linalg.pinv(pre_aa)),pre_ab))

            pre_new = np.linalg.pinv(cov_b)
            pre_kk = pre_new[:1, :1]
            pre_kc = pre_new[:1, 1:]
            pre_ck = pre_new[1:, :1]
            pre_cc = pre_new[1:, 1:]

            cov_k = np.linalg.pinv(pre_kk - np.dot(np.dot(pre_kc,np.linalg.pinv(pre_cc)),pre_ck))            

            final_std += [np.sqrt(np.abs(cov_k))] 

        return np.array(final_std).reshape(-1,)
