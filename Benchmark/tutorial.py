
from InputModel.Groundtruth import *
from InputModel.Subsystem import *
from GraphMetamodel.SurrogateModel import *
from GraphMetamodel.DefineCouplingGraph import *
import statistics_basic as stat
import GraphMetamodel.MultiScaleInference as MSI
from InputModel.Subsystem import *
from InputModel.Groundtruth import *
from matplotlib.pyplot import MultipleLocator
import matplotlib.ticker 
from Surrogate_model_a import *
from Surrogate_model_b import *
 

surrogate_a = run_surrogate_model_a(method='MultiScale', mean_scale=1, 
                                    transition_cov_scale=0.01, emission_cov_scale=10)    

surrogate_b = run_surrogate_model_b(method='MultiScale', mean_scale=1, 
                                    transition_cov_scale=0.01, emission_cov_scale=1)

surrogate_a_states = np.genfromtxt('./results/surrogate_model_a.csv', delimiter=',', skip_header=1).reshape(-1,2,2)
surrogate_b_states = np.genfromtxt('./results/surrogate_model_b.csv', delimiter=',', skip_header=1).reshape(-1,3,2)

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
metamodel.inference(test_omega=[0.8, 0.2], 
                    filepath=f'./results/metamodel_results.csv')

