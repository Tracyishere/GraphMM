# %%

from InputModel.Groundtruth import *
from InputModel.Subsystem import *
<<<<<<< Updated upstream
from GraphMetamodel.SurrogateModel_new import *
from GraphMetamodel.DefineCouplingGraph import *
import statistics_basic as stat
import GraphMetamodel.MultiScaleInference_v12 as MSI


# for i in range(1, 12):

#     # mean_scale = 1+i/10-0.6
#     mean_scale = 1+i/5 - 0.7
    
#     # for j in range(1, 12):

#     #     std_scale = j*0.1*0.1
#     for j in np.linspace(2, -1, 11):
#         std_scale = 10**(-j)

#         print(mean_scale,std_scale)
        
#         from Surrogate_model_a import *
#         try:
#             surrogate_a = run_surrogate_model_a(method='MultiScale', mean_scale=mean_scale, std_scale=std_scale)
#         except ValueError:
#             pass



# %%
    
=======
from GraphMetamodel.SurrogateModel import *
import statistics_basic as stat
>>>>>>> Stashed changes
import glob
import numpy as np
from scipy.stats import *
import matplotlib.pyplot as plt
import math
from scipy.stats import entropy
import scipy.spatial.distance as dist
from scipy.stats import multivariate_normal
import statistics_basic as stat

def entropy_1(X):
    probs = [np.mean(X == c) for c in set(X)]
    return np.sum(-p * np.log2(p) for p in probs)


def compute_model_entropy(mean, std):

    time = mean.shape[0]
    num_var = mean.shape[1]
    model_entropy = []

    for t in range(time):
        model_entropy_t = 0
        for var_idx in range(num_var):
            u = mean[t, var_idx]
            sig = std[t, var_idx]
            # model_entropy_t += norm(u, sig).entropy()
            
            x = np.linspace(u - 3*sig, u + 3*sig, 500)
            y = np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2*math.pi)*sig)
            model_entropy_t += entropy_1(y)
        
        model_entropy.append(model_entropy_t)
        
    return np.array(model_entropy)


def compute_overlap_of_normal_dist(m1,m2,std1,std2):

    if std1==0: std1 += 1e-10
    if std2==0: std2 += 1e-10
    
    N1 = stat.NormalDist(m1, std1)
    N2 = stat.NormalDist(m2, std2)

    return N1.overlap(N2)


# %%


############# sample multiple #############


from InputModel.Subsystem import *

surrogate_a_files = glob.glob('*.csv')

<<<<<<< Updated upstream
filepath = '/Users/tracy/PhD/Projects/Ongoing/2GraphMM/Benchmark/results/enumerate_surrogate_model_v8/'
=======
filepath = './results/enumerate_surrogate_model/'
>>>>>>> Stashed changes
files_sorted = []
enumerate_mean = sorted(np.unique(np.array([item.split('/')[-1].split('_')[-2] for item in glob.glob(filepath+'*.csv')])),key=float)
enumerate_cov = sorted(np.unique(np.array([item.split('/')[-1].split('_')[-1][:-4] for item in glob.glob(filepath+'*.csv*')])),key=float)

for mean in enumerate_mean:
    for cov in enumerate_cov:    
        files_sorted += ['surrogate_model_a_'+mean+'_'+cov+'.csv']

model_a_entropys,model_a_consistency = [], []
for file in files_sorted:
    model_a = np.genfromtxt(filepath+file, delimiter=',')[1:]
    model_a_mean = np.array([model_a[:,0], model_a[:,2]]).transpose()
    model_a_std = np.array([model_a[:,1], model_a[:,3]]).transpose()
    model_a_entropy = compute_model_entropy(model_a_mean, model_a_std)
    model_a_entropys += [np.mean(np.array(model_a_entropy))]
    model_consistency =[]
    for i in range(len(model_a_mean)):
        a_dist_ts0 = compute_overlap_of_normal_dist(model_a_mean[i,0], input_a[i,0], model_a_std[i,0], input_a_std[i,0])
        a_dist_ts1 = compute_overlap_of_normal_dist(model_a_mean[i,1], input_a[i,1], model_a_std[i,1], input_a_std[i,1])
        model_consistency += [a_dist_ts0, a_dist_ts1]
    model_a_consistency += [np.mean(np.array(model_consistency))]



# %%
    
# ############# single best #############

model_a = np.genfromtxt('./results/surrogate_model_a_new.csv', delimiter=',')[1:]

model_a_entropys, model_a_consistency = [], []
# model_a = np.genfromtxt(filepath+file, delimiter=',')[1:]
model_a_mean = np.array([model_a[:,0], model_a[:,2]]).transpose()
model_a_std = np.array([model_a[:,1], model_a[:,3]]).transpose()

model_a_entropys = []
enumerate_mean, enumerate_cov = [],[]
for i in range(1, 12):

    mean_scale = 1 + i/10 - 0.6
    enumerate_mean += [round(mean_scale, 2)]
    for j in np.linspace(0.8, -0.8, 11):
        std_scale = 10**(-j)
        enumerate_cov += [round(std_scale, 2)]
    
        model_a_entropy = compute_model_entropy(model_a_mean*mean_scale, model_a_std*std_scale)
        model_a_entropys += [np.mean(np.array(model_a_entropy))]
        model_consistency = []
        for i in range(len(model_a_mean)):
            a_dist_ts0 = compute_overlap_of_normal_dist(model_a_mean[i,0]*mean_scale, input_a[i,0], model_a_std[i,0]*std_scale, input_a_std[i,0])
            a_dist_ts1 = compute_overlap_of_normal_dist(model_a_mean[i,1]*mean_scale, input_a[i,1], model_a_std[i,1]*std_scale, input_a_std[i,1])
            model_consistency += [a_dist_ts0, a_dist_ts1]
        model_a_consistency += [np.mean(np.array(model_consistency))]


# %%
        
enumerate_mean = enumerate_mean[:12]
enumerate_cov = enumerate_cov[:11]

import numpy as np
from os import listdir
from os.path import isfile, join
from matplotlib import ticker, cm
from matplotlib.pyplot import MultipleLocator
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib as mpl

font = {'family' : 'Arial narrow', 'size'   : 25}
COLOR = '#202020'
mpl.rc('font', **font)
mpl.rc('xtick', labelsize=20) 
mpl.rc('ytick', labelsize=20) 
mpl.rcParams['text.color'] = COLOR
mpl.rcParams['axes.labelcolor'] = COLOR
mpl.rcParams['xtick.color'] = COLOR
mpl.rcParams['ytick.color'] = COLOR


x = np.linspace(0, 10, 11)
y = np.linspace(0, 10, 11)
X, Y = np.meshgrid(x, y)
Z = np.array(model_a_entropys).reshape(11,11).transpose()
Z = np.nan_to_num(Z, nan=-1)

fig, ax = plt.subplots(figsize=(5, 4))

levels = np.arange(np.min(Z), np.max(Z), (np.max(Z)-np.min(Z))/1000)
cs = ax.contourf(X, Y, Z, levels, cmap=plt.get_cmap('OrRd'))
cbar = fig.colorbar(cs, pad=0.01)
cbar.ax.ticklabel_format(style='sci',scilimits=(0,0),axis='x')
cbar.locator = ticker.MaxNLocator(nbins=6)
cbar.ax.tick_params(labelsize=20)
cbar.update_ticks()
plt.yticks(np.arange(0, 11, 2), [round(i, 1) for i in [float(i) for i in enumerate_cov][::2]])
plt.xticks(np.arange(0, 11, 2), [round(i, 1) for i in [float(i)-1 for i in enumerate_mean][::2]])
ax.set_xlabel('$\overline{err}$',fontsize = 25, fontname = 'Arial Narrow')
ax.set_ylabel('$\sigma^2$',fontsize = 25, fontname = 'Arial Narrow', labelpad=0.1)
ax.set_title('Surrogate model entropy',fontsize=25)
plt.savefig('./surrogate_model_a_entropy.png', dpi=600, bbox_inches = 'tight')
plt.show()

# %%

Z = np.array(model_a_consistency).reshape(11,11).transpose()
Z = np.nan_to_num(Z, nan=0.0)

fig, ax = plt.subplots(figsize=(5, 4))

levels = np.arange(np.min(Z), np.max(Z), (np.max(Z)-np.min(Z))/1000)
cs = ax.contourf(X, Y, Z, levels, cmap=plt.get_cmap('OrRd'))
cbar = fig.colorbar(cs, pad=0.01)
cbar.ax.ticklabel_format(style='sci',scilimits=(0,0),axis='x')
cbar.locator = ticker.MaxNLocator(nbins=6)
cbar.ax.tick_params(labelsize=20)
cbar.update_ticks()
plt.yticks(np.arange(0, 11, 2), [round(i, 1) for i in [float(i) for i in enumerate_cov][::2]])
plt.xticks(np.arange(0, 11, 2), [round(i, 1) for i in [float(i)-1 for i in enumerate_mean][::2]])
ax.set_xlabel('$\overline{err}$',fontsize = 25, fontname = 'Arial Narrow')
ax.set_ylabel('$\sigma^2$',fontsize = 25, fontname = 'Arial Narrow', labelpad=0.1)
ax.set_title('Model consistency',fontsize=25)
plt.savefig('./surrogate_model_a_consistency.png', dpi=600, bbox_inches = 'tight')
plt.show()






# %%

from InputModel.Subsystem import *
from InputModel.Groundtruth import *
from matplotlib.pyplot import MultipleLocator
import matplotlib.ticker 
from Surrogate_model_a import *

 
# surrogate_a = run_surrogate_model_a(method='MultiScale', mean_scale=1, std_scale=0.01)

sim_time = np.arange(0, surrogate_a.total_time, surrogate_a.dt)
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(1, 1, 1)

plt.yticks(fontproperties = 'Arial Narrow', size = 25)
plt.xticks(fontproperties = 'Arial Narrow', size = 25)
plt.ylabel(r'$I_{islet}^a\ $[pg/islet]',fontsize = 25, fontname = 'Arial')
plt.xlabel('Time [{}]'.format(surrogate_a.unit),fontsize = 25, fontname = 'Arial')
ax = plt.gca()  
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.tick_params(labelsize=25)
x_major_locator=MultipleLocator(2)
ax.xaxis.set_major_locator(x_major_locator)
ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.f'))
x_minor_locator=MultipleLocator(1)
ax.xaxis.set_minor_locator(x_minor_locator)

plt.plot(time_gt, groundtruth[:,2], color='grey', linewidth=2, label='Groundtruth') 
plt.fill_between(time_gt, 
                    (groundtruth[:,2] - groundtruth_std[:,2]).reshape(-1,), 
                    (groundtruth[:,2] + groundtruth_std[:,2]).reshape(-1,), 
                    alpha=0.2, color='grey')

plt.plot(sim_time, input_a[:,1], color='C0', label = 'Input model', linewidth=1.8)
plt.fill_between(sim_time, input_a[:,1]-input_a_std[:,1], 
                                input_a[:,1]+input_a_std[:,1],
                                color='C0', alpha = 0.1)

plt.plot(sim_time, surrogate_a.mean[:,1], color='red', label = 'Surrogate model', linewidth=2)
plt.fill_between(sim_time, 
                    surrogate_a.mean[:,1]-surrogate_a.std[:,1], 
                    surrogate_a.mean[:,1]+surrogate_a.std[:,1],
                    color='lightcoral', alpha = 0.3)

plt.legend(loc='best', prop={'size': 16, 'family': 'Arial Narrow'}, frameon=False)   
# plt.savefig('./surrogate_input_model_a.png', dpi=600, bbox_inches = 'tight')


# %%

# from Surrogate_model_a import *
from scipy.stats import multivariate_normal

# surrogate_a = run_surrogate_model_a(method='MultiScale', mean_scale=1, std_scale=0.01)
surrogate_a_mean = np.genfromtxt('./results/surrogate_model_a_new.csv', delimiter=',', skip_header=1).reshape(-1,2,2)[:, :, 0]
surrogate_a_std =  np.genfromtxt('./results/surrogate_model_a_new.csv', delimiter=',', skip_header=1).reshape(-1,2,2)[:, :, 1]

fig = plt.figure(figsize=(4.5, 4))
ax = fig.add_subplot(1, 1, 1)
ts = 80
# Generate grid of points
x1, y1 = np.meshgrid(np.linspace(10, 23, 100), np.linspace(16, 40, 100))
pos1 = np.dstack((x1, y1))
rv1 = multivariate_normal([input_a[ts, 0], input_a[ts, 1]], 
                          [[input_a_std[ts, 0]**2, 0], 
                           [0, input_a_std[ts, 1]**2]])
z1 = rv1.pdf(pos1)

x2, y2 = np.meshgrid(np.linspace(10, 23, 100), np.linspace(16, 40, 100))
pos2 = np.dstack((x2, y2))
rv2 = multivariate_normal([surrogate_a_mean[ts, 0], surrogate_a_mean[ts, 1]], 
                          [[input_a_std[ts, 0]**2, 0], 
                           [0, input_a_std[ts, 1]**2]])
z2 = rv2.pdf(pos2)

plt.contour(x1, y1, z1, colors='C1', linestyles='dashed')
plt.contour(x2, y2, z2, colors='red')
plt.plot([], [], color='C1', linestyle='--', label='Input model')
plt.plot([], [], color='red', label='Surrogate model')

ax.tick_params(labelsize=20)
x_major_locator=MultipleLocator(2)
ax.xaxis.set_major_locator(x_major_locator)
ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.f'))
x_minor_locator=MultipleLocator(1)
ax.xaxis.set_minor_locator(x_minor_locator)

ax.tick_params(labelsize=20)
y_major_locator=MultipleLocator(2)
ax.xaxis.set_major_locator(y_major_locator)
ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.d'))
y_minor_locator=MultipleLocator(1)
ax.xaxis.set_minor_locator(y_minor_locator)

plt.legend()
plt.title('Body model joint distribution',fontsize = 23, fontname = 'Arial Narrow')
plt.xlabel(r'$I_{pl}^a\ $[pg/islet]',fontsize = 22, fontname = 'Arial Narrow')
plt.ylabel(r'$D^a$ [mM]',fontsize = 22, fontname = 'Arial Narrow')
plt.legend(loc='best', prop={'size': 16, 'family': 'Arial Narrow'}, frameon=False)
plt.savefig('./surrogate_input_model_a_joint.png', dpi=600, bbox_inches = 'tight')
plt.show()



# %%
