# %%

from InputModel.Groundtruth import *
from InputModel.Subsystem import *
from GraphMetamodel.SurrogateModel import *
<<<<<<< Updated upstream
from GraphMetamodel.DefineCouplingGraph_new import *
import statistics_basic as stat
import GraphMetamodel.MultiScaleInference_new as MSI
=======
from GraphMetamodel.DefineCouplingGraph import *
import statistics_basic as stat
import GraphMetamodel.MultiScaleInference as MSI
>>>>>>> Stashed changes
from InputModel.Subsystem import *
from InputModel.Groundtruth import *
from matplotlib.pyplot import MultipleLocator
import matplotlib.ticker 
from Surrogate_model_a import *
from Surrogate_model_b import *
 

# %%

surrogate_a_states = np.genfromtxt('./results/surrogate_model_a_new.csv', delimiter=',', skip_header=1).reshape(-1,2,2)
surrogate_b_states = np.genfromtxt('./results/surrogate_model_b.csv', delimiter=',', skip_header=1).reshape(-1,3,2)

# for omg1 in range(0, 11):
#     for omg2 in range(0, 11):

for omg1 in range(0, 11, 1):
    for omg2 in range(0, 11, 1):
        Coupling_graph = coupling_graph(models = {'a_b':(surrogate_a, surrogate_b)}, 
                                        connect_var = {'a_b':('$I_{islet}^a\ [pg/islet]$', 
                                                              '$I_{cell}^b [pg/islet]$')}, 
                                        unit_weights = [[1,1]],  
                                        model_states = {'a':surrogate_a_states, 
                                                        'b':surrogate_b_states},
                                        timescale = {'a_b':[1, 5]},
                                        w_phi=1, w_omega=1, w_epsilon=1) # the parameters are adjustable

        Coupling_graph.get_coupling_graph_multi_scale(p=0.5)

        metamodel = MSI.MetaModel(Coupling_graph) 
        metamodel.inference(test_omega=[omg1*0.1, omg2*0.1], 
<<<<<<< Updated upstream
                            filepath=f'./results/enumerate_metamodel_v2/metamodel_joint_{omg1*0.1:.2f}_{omg2*0.1:.2f}.csv')
=======
                            filepath=f'./results/enumerate_metamodel/metamodel_joint_{omg1*0.1:.2f}_{omg2*0.1:.2f}.csv')
>>>>>>> Stashed changes



# %%

import glob

<<<<<<< Updated upstream
filepath = '/Users/tracy/PhD/Projects/Ongoing/1GraphMM/Benchmark/results/enumerate_metamodel_v3/'
=======
filepath = './results/enumerate_metamodel/'
>>>>>>> Stashed changes
files_sorted = []
enumerate_mean = sorted(np.unique(np.array([item.split('/')[-1].split('_')[-2] for item in glob.glob(filepath+'*.csv')])),key=float)
enumerate_cov = sorted(np.unique(np.array([item.split('/')[-1].split('_')[-1][:-4] for item in glob.glob(filepath+'*.csv*')])),key=float)

for mean in enumerate_mean:
    for cov in enumerate_cov:    
        files_sorted += ['metamodel_joint_'+mean+'_'+cov+'.csv']



# %%

for file in files_sorted:
    try:
        metamodel = np.genfromtxt(filepath + file, delimiter=',', skip_header=1)
        print(file)
        
        plt.figure(figsize=(15, 5))  # Adjust the size as needed
        for i in range(1, 6):
            plt.subplot(1, 5, i)  # 2 rows, 3 columns, i-th subplot
            plt.plot(metamodel[:, i*2])
        
        plt.tight_layout()
        plt.show()
    except:
        print(f'Error in {file}')

# %%


for file in files_sorted:
    try:
        metamodel = np.genfromtxt(filepath+file, delimiter=',', skip_header=1).reshape(-1, 6, 2)
        print(file)
        plt.plot(metamodel[:, 1, 0])
    except:
        print(f'Error in {file}')

# %%
