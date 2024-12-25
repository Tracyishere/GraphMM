import run_surrogate_ISK_active as ISK_s
import run_surrogate_IHC_active as IHC_s
import run_surrogate_VE_active as VE_s
from GraphMetamodel.DefineCouplingGraph_3model import *
import GraphMetamodel.MultiScaleInference_3model as MSI

def run(cell_number):
    ############ get surrogate model states ############

    surrogate_VE_state_path = f'./results/surrogate_VE_600s_35hub/surrogate_VE_16.7mM_IHC_pm_potential_10min_cell_{cell_number}.csv'
    surrogate_IHC_state_path = f'./results/surrogate_IHC_600s_35hub/surrogate_IHC_cell_{cell_number}.csv'
    surrogate_ISK_state_path = f'./results/surrogate_ISK_600s_35hub/surrogate_ISK_600s_cell_{cell_number}.csv'

    ############ define coupling graph ############

    VE_s.surrogate_VE.create_model(cell_number)

    Coupling_graph_VE_IHC_ISK = coupling_graph(models = {'VE_ISK': {'VE': VE_s.surrogate_VE.model, 'ISK': ISK_s.surrogate_ISK}, 
                                                         'IHC_ISK': {'IHC': IHC_s.surrogate_IHC, 'ISK': ISK_s.surrogate_ISK}}, 
                                    connect_var = {'VE_ISK': ['F.VE', 'NR.ISK'],
                                                   'IHC_ISK': ['Ca1.IHC', 'Ca_ic.ISK']}, 
                                    model_states_file = {'VE_ISK': {'VE': surrogate_VE_state_path, 'ISK': surrogate_ISK_state_path}, 
                                                         'IHC_ISK': {'IHC': surrogate_IHC_state_path, 'ISK': surrogate_ISK_state_path}},
                                    unit_weights = {'VE_ISK':[30,0.02*60],
                                                    'IHC_ISK':[1, 1]},
                                    timescale = {'VE_ISK': [1, 1],
                                                 'IHC_ISK': [1, 60]})

    graph_path = './results/Metamodel_VE_IHC_ISK_600s_35hub/coupling_graph/'
    graph_VE_ISK = graph_path + f'coupling_graph_param_VE_ISK_600s_{cell_number}.csv'
    graph_IHC_ISK = graph_path + f'coupling_graph_param_ISK_IHC_600s_{cell_number}.csv'
    temp_path = graph_path
    Coupling_graph_VE_IHC_ISK.get_coupling_graph_multi_scale(graph_output={'VE_ISK': graph_VE_ISK,
                                                                           'IHC_ISK': graph_IHC_ISK},
                                                            temp_path = temp_path,
                                                            input_file_num=57)


    ############ model inference ############
    obs_VE_state_path = f'./results/obs_surrogate_VE_600s_{cell_number}.csv'
    obs_IHC_state_path = temp_path + f'obs_IHC_cell_{cell_number}.csv'
    obs_ISK_state_path = f'./results/obs_surrogate_ISK_600s_{cell_number}.csv'

    Metamodel = MSI.MetaModel(Coupling_graph_VE_IHC_ISK,
                              upd_model_state_files = [obs_VE_state_path,
                                                       obs_IHC_state_path,
                                                       obs_ISK_state_path])

    cell_potential = np.genfromtxt(f'./InputModel/Input_IHC/input_IHC_600s_35hub/input_IHC_cell_{cell_number}.csv', 
                                   delimiter=',', max_rows=20000, skip_header=1).reshape(20000, -1)[:, 0]

    VE_ton_lst = VE_s.VE.Vmem2Ksti_time(20000, cell_potential)
    steps = int(ISK_s.total_time // ISK_s.dt) + 1
    ton_lst = ISK_s.Vmem2Ksti_time(steps, cell_potential)
    ISK_V_mem = np.ones(steps)*(-70)
    for s,e in ton_lst: ISK_V_mem[s:e] += 50

    result_path = f'./results/Metamodel_VE_IHC_ISK_600s_35hub/'
    Metamodel.inference(input_file_num=57, ton_lst=VE_ton_lst, V_list=ISK_V_mem, 
                        filepath=result_path+f'Metamodel_VE_IHC_ISK_600s_35hub_{cell_number}.csv')

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cell_number = int(sys.argv[1])
        run(cell_number)
    else:
        print("Please provide a cell number as an argument.")
