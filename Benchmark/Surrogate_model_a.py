# %%

from GraphMetamodel.utils import *
from InputModel.Subsystem import *
<<<<<<< Updated upstream
from GraphMetamodel.SurrogateModel_new import *
=======
from GraphMetamodel.SurrogateModel import *
>>>>>>> Stashed changes
import numpy as np
import random

np.random.seed(12)
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



def run_surrogate_model_a(method, mean_scale, std_scale):

    ''' Run the input model at different timescales gets similar results, but it's different when run surrogate models. '''


    if method == 'MultiScale':
        # run at the timescale of each input model

        measure_std_scale = 0.01
        observations_a = get_observations([D_0, I_0], fx_model_a, dt_a, sim_time_a, measure_std_scale)
        observations_a[:,:,0] *= mean_scale
        # observations_a = random.gauss(input_a_for_surrogate,input_a_for_surrogate*0.1)

        # time-variant noise
        # Q_a = [np.diag(abs(input_a[i]*random.gauss(0.1,std_scale))) for i in range(n_step_a)]
        # R_a = [np.diag(abs(observations_a[i,:,0]*random.gauss(1,std_scale))) for i in range(n_step_a)]
        # time-invariant noise
        # Q_a = [np.diag(np.mean(input_a, axis=0)*0.1) for i in range(n_step_a)]
        # R_a = [np.diag(np.mean(input_a, axis=0)*1) for i in range(n_step_a)]

        initial_noise_scale = 0.01
        transition_cov_scale = std_scale
        # emission_cov_scale = std_scale
        emission_cov_scale = 0.1

        surrogate_a = SurrogateInputModel(name='model_a',
                                        state=model_var_a, 
                                        initial=np.array([D_0, I_0]), initial_noise_scale=initial_noise_scale, 
                                        fx=fx_model_a, dt=dt_a, input_dt=dt_a, total_time=sim_time_a, 
                                        measure_std_scale=measure_std_scale, transition_cov_scale=transition_cov_scale,
                                        emission_cov_scale=emission_cov_scale, noise_model_type='time-variant',
                                        unit='min')
        

        surrogate_a.inference(n_repeat=1, filepath=f'./results/surrogate_model_a.csv')
        # plot_surrogatemodel(surrogate_a, inputmodel=input_a, inputmodelerror=input_a_std)

    return surrogate_a

surrogate_a = run_surrogate_model_a(method='MultiScale', mean_scale=1, std_scale=0.01)

<<<<<<< Updated upstream
# # %%

# a = np.genfromtxt('./results/surrogate_model_a_new.csv', delimiter=',', skip_header=1)
# b = np.genfromtxt('./results/surrogate_model_a.csv', delimiter=',')
# for i in range(2):
#     plt.plot(a[:, i*2])
#     plt.fill_between(np.arange(0, len(a[:, i*2]), 1), a[:, i*2]-a[:, i*2+1], a[:, i*2]+a[:, i*2+1], alpha=0.1)
#     plt.plot(b[:, i*2])
#     plt.fill_between(np.arange(0, len(b[:, i*2]), 1), b[:, i*2]-b[:, i*2+1], b[:, i*2]+b[:, i*2+1], alpha=0.1)
#     plt.show()
=======
>>>>>>> Stashed changes

# %%
