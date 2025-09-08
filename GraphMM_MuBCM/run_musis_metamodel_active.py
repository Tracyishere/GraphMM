import run_surrogate_ISK_active as ISK_s
import run_surrogate_ICN_active as ICN_s
import run_surrogate_VE_active as VE_s
from run_surrogate_VE_active import SurrogateVE
from run_surrogate_ISK_active import SurrogateISK
from run_surrogate_ICN_active import SurrogateICN
from GraphMetamodel.DefineCouplingGraph_3model import *
import GraphMetamodel.MultiScaleInference_3model as MSI
import yaml
import os
import numpy as np


def run(cell_number):
    ############ get surrogate model states ############

    surrogate_VE_state_path = f'./results/600s/surrogate_VE_600s_nohub/surrogate_VE_16.7mM_ICN_pm_potential_10min_cell_{cell_number}.csv'
    surrogate_ICN_state_path = f'./results/600s/surrogate_ICN_nohub_600s/surrogate_ICN_cell_{cell_number}.csv'
    surrogate_ISK_state_path = f'./results/600s/surrogate_ISK_600s_nohub/surrogate_ISK_600s_cell_{cell_number}.csv'

    ############ define coupling graph ############

    # Load config
    with open('config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Create and initialize SurrogateVE instance
    ve_instance = SurrogateVE()
    ve_instance.create_model(config['VE'])

    # Create and initialize SurrogateISK instance
    isk_instance = SurrogateISK()
    isk_instance.create_model(config['ISK'])

    # Create and initialize SurrogateICN instance
    ICN_instance = SurrogateICN()
    ICN_instance.create_model(config['ICN'])


    Coupling_graph_VE_ICN_ISK = coupling_graph(models = {'VE_ISK': {'VE': ve_instance.model, 'ISK': isk_instance.model},
                                                         'ICN_ISK': {'ICN': ICN_instance.model, 'ISK': isk_instance.model}},
                                    connect_var = {'VE_ISK': ['F.VE', 'NR.ISK'],
                                                   'ICN_ISK': [f'Ca{cell_number}.ICN', f'Ca_ic.ISK']},
                                    model_states_file = {'VE_ISK': {'VE': surrogate_VE_state_path, 'ISK': surrogate_ISK_state_path},
                                                         'ICN_ISK': {'ICN': surrogate_ICN_state_path, 'ISK': surrogate_ISK_state_path}},
                                    unit_weights = {'VE_ISK':[30,0.02*60],
                                                    'ICN_ISK':[1, 1]},
                                    timescale = {'VE_ISK': [1, 1],
                                                 'ICN_ISK': [1, 60]})
    
    graph_path = './results/coupling_graph/'

    if not os.path.exists(graph_path):
        os.mkdir(graph_path)

    graph_VE_ISK = graph_path + f'coupling_graph_param_VE_ISK_600s_{cell_number}.csv'
    graph_ICN_ISK = graph_path + f'coupling_graph_param_ISK_ICN_600s_{cell_number}.csv'
    temp_path = graph_path
    Coupling_graph_VE_ICN_ISK.get_coupling_graph_multi_scale(graph_output={'VE_ISK': graph_VE_ISK,
                                                                           'ICN_ISK': graph_ICN_ISK},
                                                            temp_path = temp_path,
                                                            input_file_num=57)


    ############ model inference ############
    obs_VE_state_path = f'./results/obs_surrogate_VE_600s_{cell_number}.csv'
    obs_ICN_state_path = temp_path + f'obs_ICN_cell_{cell_number}.csv'
    obs_ISK_state_path = f'./results/obs_surrogate_ISK_600s_{cell_number}.csv'

    Metamodel = MSI.MetaModel(Coupling_graph_VE_ICN_ISK,
                              upd_model_state_files = [obs_VE_state_path,
                                                       obs_ICN_state_path,
                                                       obs_ISK_state_path])

    cell_potential = np.genfromtxt(f'./InputModel/Input_ICN/input_ICN_600s_nohub/input_ICN_cell_{cell_number}.csv', 
                                   delimiter=',', max_rows=20000, skip_header=1).reshape(20000, -1)[:, 0]

    VE_ton_lst = VE_s.VE.Vmem2Ksti_time(20000, cell_potential)
    VE_ton_lst = np.array(VE_ton_lst)
    steps = int(isk_instance.model.total_time // isk_instance.model.dt) + 1
    ton_lst = ISK_s.Vmem2Ksti_time(steps, cell_potential)
    ISK_V_mem = np.ones(steps)*(-70)
    for s,e in ton_lst: ISK_V_mem[s:e] += 50

    result_path = f'./results/'
    Metamodel.inference(input_file_num=57, ton_lst=VE_ton_lst, V_list=ISK_V_mem, 
                        filepath=result_path+f'Metamodel_VE_ICN_ISK_600s_nohub_{cell_number}.csv')

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cell_number = int(sys.argv[1])
        run(cell_number)
    else:
        print("Please provide a cell number as an argument.")
