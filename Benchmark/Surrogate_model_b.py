# %%

from GraphMetamodel.utils import *
from InputModel.Subsystem import *
from GraphMetamodel.SurrogateModel import *
import numpy as np
import random

np.random.seed(42)
# define the forward function
def fx_model_b(x, dt):
    
    xout = np.empty(x.shape)
    
    para = [-0.02764259292818268, -3.2929482000130115, 21.06470961431193,
            -0.38572678573975416, -4.37583184690527, 53.88775990530528,
            0.001192339928239537, -0.14355227056573305]
    
    xout[2] = x[2] + dt*(para[0]*x[1] + para[1]*x[0]**(2/3) + para[2])
    xout[1] = x[1] + dt*(para[3]*x[0] + para[4]*x[2] + para[5])
    xout[0] = x[0] + dt*(para[6]*x[1]**2 + para[7])
    
    return xout

def run_surrogate_model_b(method, mean_scale, transition_cov_scale, emission_cov_scale, save_path=None):

    if method == 'MultiScale':
        # run at the timescale of each input model

        observations_b = get_observations([gamma_0, I_0, G_0], fx_model_b, dt_b, sim_time_b, 0.001)
        observations_b[:,:,0] *= mean_scale

        surrogate_b = SurrogateInputModel(name='model_b',
                                          state=model_var_b, 
                                          initial=np.array([gamma_0, I_0, G_0]),
                                          initial_noise_scale=0.01,
                                          measure_std_scale=0.01,
                                          fx=fx_model_b, dt=dt_b, input_dt=dt_b, total_time=sim_time_b, 
                                          transition_cov_scale=transition_cov_scale,
                                          emission_cov_scale=emission_cov_scale, noise_model_type='time-variant',
                                          unit='min')
        

        surrogate_b.inference(n_repeat=1, filepath=save_path)
        # plot_surrogatemodel(surrogate_b, inputmodel=input_b, inputmodelerror=input_b_std)

    return surrogate_b


# %%
if __name__ == "__main__":

    surrogate_b = run_surrogate_model_b(method='MultiScale', mean_scale=1,
                                        transition_cov_scale=0.01, emission_cov_scale=1,
                                        save_path='./results/surrogate_model_b.csv')

# %%
