'''    
	Author: Chenxi Wang (chenxi.wang@salilab.org) 
	Date: 2021.03.22
'''

import numpy as np
from GraphMetamodel.utils import *
from GraphMetamodel.DefineSurrogateModel_ISK import *
import InputModel.Input_ISK.InsulinSecretionKinetic_forward_function_pm_ode as ISK  
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import logging
import yaml
import argparse
import os

def interpolate(x, y, xnew):
    f = interp1d(x, y, kind='cubic', fill_value="extrapolate")
    ynew = f(xnew)
    return ynew

def Vmem2Ksti_time(k_timelength, Vmem):
    data = Vmem

    if (data > -35).any():
        threshold = 2  
        first_diff = [data[i] - data[i - 1] for i in range(1, len(data))]
        key_points = [i for i, diff in enumerate(first_diff) if abs(diff) > threshold]

        data = key_points
        threshold = 200
        first_diff = [data[i] - data[i - 1] for i in range(1, len(data))]
        key_key_points = [key_points[0]] + \
            [[key_points[i], key_points[i+1]] for i, diff in enumerate(first_diff) if abs(diff) > threshold] + \
                [key_points[-1]]
        
        combined_list = []
        for i in range(len(key_key_points) - 1):
            start = key_key_points[i]
            end = key_key_points[i + 1]
            if isinstance(start, list):
                start = start[1]
            if isinstance(end, list):
                end = end[0]
            combined_list.append([start, end])
        combined_list[-1][-1] = len(Vmem)
        
        K_sti_time = np.array(combined_list)
        ratio = k_timelength / len(Vmem)
        K_sti_time = np.round(K_sti_time*ratio,0).astype(int)
    
    else:
        K_sti_time = []

    return K_sti_time

def fx_ISK(x, dt, step, V_list):
    SimTime = np.linspace(step, step+1, 2)
    V_t = V_list[step]
    xin = x[1:]
    ode_out = odeint(ISK._f_input_ode_ISK, xin, SimTime, args=(V_t, ))
    xout = [V_t] + ode_out[-1].tolist()
    return np.array(xout)

class SurrogateISK:
    def __init__(self):
        self.model = None

    def create_model(self, config):
        self.model = SurrogateInputModel(name='Insulin Secretion Kinetic Model',
                                         state=config['state_variables'], 
                                         initial=ISK.InsulinSecretionKineticInit(), 
                                         initial_noise_scale=config['initial_noise_scale'], 
                                         fx=fx_ISK, dt=config['dt'], total_time=config['total_time'], 
                                         input_dt=config['input_dt'],
                                         measure_std_scale=config['measure_std_scale'], 
                                         transition_cov_scale=config['transition_cov_scale'], 
                                         emission_cov_scale=config['emission_cov_scale'], 
                                         noise_model_type='time-variant', unit='min')

def run(cell_number, config):
    try:
        output_dir = config['output_filepath']
        os.makedirs(output_dir, exist_ok=True)
        
        cell_potential = np.genfromtxt(config['input_path']+f'/input_ICN_cell_{cell_number}.csv',
                                       delimiter=',',skip_header=1, max_rows=10000).reshape(10000, -1)[:, 0]
        steps = int(float(config['total_time'] // float(config['dt'])))+ 1
        print(type(config['measure_std_scale']))
        
        ton_lst = Vmem2Ksti_time(steps, cell_potential)
        V_mem = np.ones(steps)*(-70)
        for s,e in ton_lst: V_mem[s:e] += 50
        
        surrogate_ISK = SurrogateISK()
        surrogate_ISK.create_model(config)
        surrogate_ISK.model.inference(n_repeat=config['n_repeat'], verbose=config['verbose'],
                                      output_filepath=f'{output_dir}/surrogate_ISK_300s_cell_{cell_number}.csv',
                                      Vm=V_mem)
        logging.info(f"Successfully completed {__name__} for cell {cell_number}")
    except Exception as e:
        logging.error(f"Error in {__name__} for cell {cell_number}: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run surrogate ISK model")
    parser.add_argument("cell_number", type=int, help="Cell number to process")
    args = parser.parse_args()
    
    with open('config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)['ISK']
    
    run(args.cell_number, config)
