from InputModel.Groundtruth import *
from InputModel.Subsystem import *
# from Surrogate_model_b import *
# from Surrogate_model_a import *
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties


def cal_product(mean1, mean2, std1, std2):

    p = 0.6
    product_mean = p*mean1 + (1-p)*mean2
    product_var = p*std1**2 + (1-p)*std2**2 + p*(1-p)*(mean1-mean2)**2
        
    return product_mean, product_var


# surrogate_a = run_surrogate_model_a(method='MultiScale')
surrogate_a_mean = np.genfromtxt('./results/surrogate_model_a.csv', delimiter=',', skip_header=1).reshape(-1, 2, 2)[:, :, 0]
surrogate_a_std = np.genfromtxt('./results/surrogate_model_a.csv', delimiter=',', skip_header=1).reshape(-1, 2, 2)[:, :, 1]
surrogate_b_mean = np.genfromtxt('./results/surrogate_model_b.csv', delimiter=',', skip_header=1).reshape(-1, 3, 2)[:, :, 0]
surrogate_b_std = np.genfromtxt('./results/surrogate_model_b.csv', delimiter=',', skip_header=1).reshape(-1, 3, 2)[:, :, 1]
surrogate_b_mean_sparse = np.array([surrogate_b_mean[i] for i in range(800) if i % 5 ==0])
surrogate_b_std_sparse = np.array([surrogate_b_std[i] for i in range(800) if i % 5 ==0])
data = np.concatenate((surrogate_a_mean, surrogate_b_mean_sparse), axis=1)


coup_mean, coup_std = [], []
for i in range(160):
    prod_mean, prod_var = cal_product(surrogate_a_mean[i, 0], surrogate_b_mean_sparse[i, 0], surrogate_a_std[i, 0], surrogate_b_std_sparse[i, 0])
    coup_mean += [prod_mean]
    coup_std += [np.sqrt(prod_var)]

# coup_mean, coup_std = [], []
# for i in range(160):
#     prod_mean, prod_var = cal_product(surrogate_a_mean[i, 1], surrogate_b_mean_sparse[i, 1], surrogate_a_std[i, 1], surrogate_b_std_sparse[i, 1])
#     coup_mean += [prod_mean]
#     coup_std += [np.sqrt(prod_var)]

# coup_ = np.genfromtxt('./results/coupling_graph_param_test.csv', delimiter=' ')
# print(coup_.shape)    
# coup_mean = coup_[800:, 0]
# coup_std = coup_[800:, 1]   
# coup_mean = coup_mean[::5]                                                                                                 
# coup_std = coup_std[::5]  

meta = np.genfromtxt('./results/test_2pair_omega_raw2/toy_metamodel_joint_test_coupling_prior_0.6_0.1.csv',delimiter=',',skip_header=1)
meta = np.array([meta[i] for i in range(800) if i % 5 ==0])



font_path = '/wynton/home/sali/chenxi_wang/.local/lib/python3.9/site-packages/matplotlib/mpl-data/fonts/ttf/ARIALN.ttf'
font_prop = FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['font.size'] = 17
plt.yticks(fontproperties = font_prop, size = 17)
plt.xticks(fontproperties = font_prop, size = 17)
plt.xlabel('Time [min]',fontsize = 20, fontproperties = font_prop)
ax = plt.gca()  
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

time = np.linspace(0,8,len(data))

plt.plot(time, data[:, 0], label='PC1')
plt.fill_between(time, data[:, 0]-surrogate_a_std[:, 0], data[:, 0]+surrogate_a_std[:, 0], alpha=0.15)

plt.plot(time, data[:, -3], label='PC3')
plt.fill_between(time, data[:, -3]-surrogate_b_std_sparse[:, 0], data[:, -3]+surrogate_b_std_sparse[:, 0], alpha=0.15)

plt.plot(time, coup_mean, color='grey', label='prior coupling variable')
plt.fill_between(time, np.array(coup_mean)-np.array(coup_std), np.array(coup_mean)+np.array(coup_std), alpha=0.3, color='grey')

plt.plot(time, meta[:, 2], color='red', label='posterior coupling variable')
plt.fill_between(time, np.array(meta[:, 2])-np.array(meta[:, 3]), np.array(meta[:, 2])+np.array(meta[:, 3]), alpha=0.3, color='red')



# plt.plot(time, data[:, 1], label='PC2')
# plt.fill_between(time, data[:, 1]-surrogate_a_std[:, 1], data[:, 1]+surrogate_a_std[:, 1], alpha=0.15)

# plt.plot(time, data[:, -2], label='PC4')
# plt.fill_between(time, data[:, -2]-surrogate_b_std_sparse[:, 1], data[:, -2]+surrogate_b_std_sparse[:, 1], alpha=0.15)

# plt.plot(time, coup_mean, color='grey', label='prior coupling variable')
# plt.fill_between(time, np.array(coup_mean)-np.array(coup_std), np.array(coup_mean)+np.array(coup_std), alpha=0.3, color='grey')

# plt.plot(time, meta[:, 0], color='red', label='posterior coupling variable')
# plt.fill_between(time, np.array(meta[:, 0])-np.array(meta[:, 1]), np.array(meta[:, 0])+np.array(meta[:, 1]), alpha=0.3, color='red')

plt.legend(loc='best', prop=font_prop)
plt.savefig('./results/coupling_variable.png', dpi=600, bbox_inches='tight')

