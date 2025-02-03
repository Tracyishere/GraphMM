# %%

from GraphMetamodel.utils import *
from InputModel.Subsystem import *
from GraphMetamodel.SurrogateModel import *
import numpy as np
import random

np.random.seed(42)
# define the forward function
def fx_model_a(x, dt):
    
    xout = np.empty(x.shape)
    
    # for 8 min fitting
    para = [1.47702822040744, 
            -0.6461834269007786, 
            0.07434489524163206, 
            -2.937179498275664, 
            22.219341907981303]
    
    xout[0] = x[0] + dt*(para[0]*np.log(x[1]) + para[1])
    xout[1] = x[1] + dt*(para[2]*x[0]**2 + para[3]*x[0] + para[4])
    
    return xout



def run_surrogate_model_a(method, mean_scale, transition_cov_scale, emission_cov_scale, save_path=None):

    ''' Run the input model at different timescales gets similar results, but it's different when run surrogate models. '''


    if method == 'MultiScale':
        # run at the timescale of each input model

        observations_a = get_observations([D_0, I_0], fx_model_a, dt_a, sim_time_a, 0.001)
        observations_a[:,:,0] *= mean_scale

        surrogate_a = SurrogateInputModel(name='model_a',
                                        state=model_var_a, 
                                        initial=np.array([D_0, I_0]),
                                        initial_noise_scale=0.01,
                                        measure_std_scale=0.01,
                                        fx=fx_model_a, dt=dt_a, input_dt=dt_a, total_time=sim_time_a, 
                                        transition_cov_scale=transition_cov_scale,
                                        emission_cov_scale=emission_cov_scale, noise_model_type='time-variant',
                                        unit='min')
        

        surrogate_a.inference(n_repeat=1, filepath=save_path)
        # plot_surrogatemodel(surrogate_a, inputmodel=input_a, inputmodelerror=input_a_std)

    return surrogate_a

# %%

if __name__ == "__main__":
    surrogate_a = run_surrogate_model_a(method='MultiScale', mean_scale=1,
                                        transition_cov_scale=0.01, emission_cov_scale=10, 
                                        save_path='./results/surrogate_model_a.csv')

# %%
