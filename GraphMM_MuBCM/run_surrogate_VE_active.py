'''    
	Author: Chenxi Wang (chenxi.wang@salilab.org) 
	Date: 2021.03.22
'''

import numpy as np
from GraphMetamodel.utils import *
from GraphMetamodel.DefineSurrogateModel_VE import *
import InputModel.Input_VE.VesicleExocytosis_forward_function_G167_raw_potential as VE
import logging
import yaml
import argparse
import os

def fx(x, dt, time, V_mem):

    G_ex = 16.7
    sim_state =  'ICN_pm_potential'

    I,V,F,R,D,D_IR,gamma,rho,ISR = x
    
    I_t = VE._f_I(I,V,dt)
    V_t = VE._f_V(V,I,F,dt)
    F_t = VE._f_F(F,rho,D_IR,G_ex,dt,time,sim_state,V_mem)
    R_t = VE._f_R(R,I,V,gamma,G_ex,dt)
    D_t = VE._f_D(D,gamma,R,D_IR,G_ex,dt)
    D_IR_t = VE._f_D_IR(D_IR,D,rho,G_ex,dt,time,sim_state,V_mem)
    gamma_t = VE._f_gamma(gamma,G_ex,dt)
    rho_t = VE._f_rho(rho,gamma,dt,time,sim_state,V_mem)
    ISR_t = VE._f_ISR(F,G_ex)
    
    return np.array([I_t, V_t, F_t, R_t, D_t, D_IR_t, gamma_t, rho_t, ISR_t])

class SurrogateVE:
    def __init__(self):
        self.model = None

    def create_model(self, config):
        self.model = SurrogateInputModel(name='Vesicle Exocytosis Model',
                                         state=config['state_variables'], 
                                         initial=VE.VesicleExocytosisInit(config['G_ex']), 
                                         initial_noise_scale=config['initial_noise_scale'], 
                                         fx=fx, dt=config['dt'], total_time=config['total_time'], 
                                         input_dt=config['input_dt'],
                                         measure_std_scale=config['measure_std_scale'], 
                                         transition_cov_scale=config['transition_cov_scale'], 
                                         emission_cov_scale=config['emission_cov_scale'], 
                                         noise_model_type='time-variant', unit='min')

def run(cell_number, config):
    try:
        output_dir = config['output_filepath']
        os.makedirs(output_dir, exist_ok=True)
        
        path = config['input_path']
        V_mem = np.genfromtxt(path+f'/input_ICN_cell_{cell_number}.csv',
                            delimiter=',',skip_header=1, max_rows=10000).reshape(10000, -1)[:, 0]
        ton_lst = VE.Vmem2Ksti_time(10000, V_mem)
        
        surrogate_VE = SurrogateVE()
        surrogate_VE.create_model(config)
        surrogate_VE.model.inference(n_repeat=config['n_repeat'], verbose=config['verbose'],
                                     output_filepath=f"{output_dir}/surrogate_VE_{config['G_ex']}mM_ICN_pm_potential_{config['total_time']}min_cell_{cell_number}.csv",
                                     V_mem=ton_lst)
        
        logging.info(f"Successfully completed {__name__} for cell {cell_number}")
    except Exception as e:
        logging.error(f"Error in {__name__} for cell {cell_number}: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run surrogate VE model")
    parser.add_argument("cell_number", type=int, help="Cell number to process")
    args = parser.parse_args()
    
    with open('config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)['VE']
    
    run(args.cell_number, config)