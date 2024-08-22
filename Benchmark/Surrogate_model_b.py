# %%

from GraphMetamodel.utils import *
import numpy as np
from InputModel.Subsystem import *
<<<<<<< Updated upstream
from GraphMetamodel.SurrogateModel_new import *
=======
from GraphMetamodel.SurrogateModel import *
>>>>>>> Stashed changes
import random

np.random.seed(12)
# define the forward function

def fx_model_b(x, dt):

    xout = np.empty(x.shape)

    para = [-0.02764259292818268, -3.2929482000130115, 21.06470961431193,
            -0.38572678573975416, -4.37583184690527, 53.88775990530528,
            0.001192339928239537, -0.14355227056573305]

    # xout[2] = x[2] + dt*(para[0]*x[1] + para[1]*x[0]**(2/3) + para[2])
    xout[2] = x[2] + dt*(para[0]*x[1] + para[1]*x[0]**(2/3) + para[2])
    xout[1] = x[1] + dt*(para[3]*x[0] + para[4]*x[2] + para[5])
    xout[0] = x[0] + dt*(para[6]*x[1]**2 + para[7])

    return xout

measure_std_scale = 0.001
observations_b = get_observations([gamma_0, I_0, G_0], fx_model_b, dt_b, sim_time_b, measure_std_scale)

# time-variant noise
transition_cov_scale = 0.1
emission_cov_scale = 1
# Q_b = [np.diag(abs(input_b[i]*random.gauss(0.1,0.01))) for i in range(n_step_b)]
# R_b = [np.diag(abs(observations_b[i,:,0]*random.gauss(1,0.01))) for i in range(n_step_b)]

# time-invariant noise
Q_b = [np.diag(np.mean(input_b, axis=0)*0.1) for i in range(n_step_b)]
R_b = [np.diag(np.mean(input_b, axis=0)*1) for i in range(n_step_b)]

surrogate_b = SurrogateInputModel(name='model_b',
                                  state=model_var_b,
                                  initial=[gamma_0, I_0, G_0],
                                  initial_noise_scale=input_b[1][0],
                                  fx=fx_model_b, input_dt=dt_b, measure_std_scale=measure_std_scale,
                                  transition_cov_scale=transition_cov_scale, emission_cov_scale=emission_cov_scale,
                                  noise_model_type = 'time-invariant',
                                  dt=dt_b, total_time=sim_time_b, unit='min')

surrogate_b.inference(n_repeat=1, filepath='./results/surrogate_model_b.csv')
# plot_surrogatemodel(surrogate_b, inputmodel=input_b, inputmodelerror=input_b_std)

<<<<<<< Updated upstream
# # %%

# a = np.genfromtxt('./results/surrogate_model_b_new1.csv', delimiter=',', skip_header=1)
# b = np.genfromtxt('./results/surrogate_model_b.csv', delimiter=',')
# for i in range(3):
#     plt.plot(a[:, i*2])
#     plt.fill_between(np.arange(0, len(a[:, i*2]), 1), a[:, i*2]-a[:, i*2+1], a[:, i*2]+a[:, i*2+1], alpha=0.1)
#     plt.plot(b[:, i*2])
#     plt.fill_between(np.arange(0, len(b[:, i*2]), 1), b[:, i*2]-b[:, i*2+1], b[:, i*2]+b[:, i*2+1], alpha=0.1)
#     plt.show()
=======
>>>>>>> Stashed changes

# %%
