from GraphMetamodel.utils import *
from GraphMetamodel.DefineSurrogateModel_ICN import *
from InputModel.Input_ICN.IsletHubCell_parameter import *
import InputModel.Input_ICN.IsletHubCell_forward_function_sec as ICN
import os
import logging
import yaml
import argparse

def create_state_variables(config):
    state_variables = []
    num_cells = config.get('num_cells', 57)  # Default to 57 if not specified
    for name in ['V', 'n', 's', 'Ca']:
        for i in range(num_cells):
            state_variables.append(f'{name}{i}.ICN')
    return state_variables

class SurrogateICN:
    def __init__(self):
        self.model = None

    def create_model(self, config):
        state_variables = create_state_variables(config)
        
        self.model = SurrogateInputModel(name='Islet Hub Cell', 
                                         state=state_variables,
                                         initial=fig6Init(),
                                         initial_noise_scale=config['initial_noise_scale'],
                                         fx=ICN._fx_ICN, 
                                         dt=config['dt'], 
                                         input_dt=config['input_dt'], 
                                         total_time=config['total_time'],
                                         measure_std_scale=config['measure_std_scale'],
                                         transition_cov_scale=config['transition_cov_scale'],
                                         emission_cov_scale=config['emission_cov_scale'],
                                         noise_model_type='time-variant', 
                                         unit='sec')

def run(config):
    try:
        output_dir = config['output_filepath']
        os.makedirs(output_dir, exist_ok=True)
        
        surrogate_ICN = SurrogateICN()
        surrogate_ICN.create_model(config)
        surrogate_ICN.model.inference(n_repeat=config['n_repeat'], 
                                      verbose=config['verbose'],
                                      obs_from_input=config['obs_from_input'],
                                      filepath=output_dir)
        logging.info(f"Successfully completed {__name__}")
    except Exception as e:
        logging.error(f"Error in {__name__}: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run surrogate ICN model")
    args = parser.parse_args()
    
    with open('config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)['ICN']
    
    run(config)
