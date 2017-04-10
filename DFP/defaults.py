'''
Some default values of parameters
'''

import numpy as np
import time

## Target maker
target_maker_args = {}
target_maker_args['min_num_targs'] = 3	
target_maker_args['rwrd_schedule_type'] = 'exp'
target_maker_args['gammas'] = []
target_maker_args['invalid_targets_replacement'] = 'nan'

## Simulator
simulator_args = {}
simulator_args['resolution'] = (160,120)
simulator_args['frame_skip'] = 4
simulator_args['color_mode'] = 'GRAY'	
simulator_args['maps'] = ['MAP01']
simulator_args['switch_maps'] = False
simulator_args['num_simulators'] = 8
simulator_args['game_args'] = ""

## Experience
# Train experience
train_experience_args = {}
train_experience_args['memory_capacity'] = 30000
train_experience_args['history_length'] = 4
train_experience_args['history_step'] = 1
train_experience_args['shared'] = False
train_experience_args['meas_statistics_gamma'] = 0.
train_experience_args['num_prev_acts_to_return'] = 0

# Test policy experience
test_policy_experience_args = train_experience_args.copy()
test_policy_experience_args['memory_capacity'] = 20000
	
## Agent	
agent_args = {}

# agent type
agent_args['agent_type'] = 'all_actions_at_once_advantage'

# preprocessing
agent_args['preprocess_input_images'] = lambda x: x / 255. - 0.5
agent_args['preprocess_input_measurements'] = lambda x: x / 100. - 0.5
agent_args['preprocess_input_targets'] = lambda x: x
agent_args['postprocess_predictions'] = lambda x: x
agent_args['discrete_controls_manual'] = []
agent_args['opposite_button_pairs'] = []
	
# agent properties
agent_args['new_memories_per_batch'] = 8
agent_args['add_experiences_every'] = 1
agent_args['random_objective_coeffs'] = False
agent_args['objective_coeffs_distribution'] = 'none'

# net parameters
agent_args['conv_params']     = None 
agent_args['fc_img_params']   = None
agent_args['fc_meas_params']  = None
agent_args['fc_obj_params']   = None  
agent_args['fc_joint_params'] = None
agent_args['weight_decay'] = 0.00000

# optimization parameters
agent_args['batch_size'] = 64
agent_args['init_learning_rate'] = 0.0002
agent_args['lr_step_size'] = 300000
agent_args['lr_decay_factor'] = 0.3
agent_args['adam_beta1'] = 0.95
agent_args['adam_epsilon'] = 1e-4		
agent_args['optimizer'] = 'Adam'
agent_args['reset_iter_count'] = True
agent_args['clip_gradient'] = 0

# directories		
agent_args['checkpoint_dir'] = 'checkpoints'
agent_args['log_dir'] = 'logs'
agent_args['init_model'] = ''
agent_args['model_name'] = "predictor.model"
agent_args['model_dir'] = time.strftime("%Y_%m_%d_%H_%M_%S")		

# logging and testing
agent_args['print_err_every'] = 50
agent_args['detailed_summary_every'] = 1000
agent_args['test_pred_every'] = 0
agent_args['test_policy_every'] = 10000
agent_args['num_batches_per_pred_test'] = 0
agent_args['num_steps_per_policy_test'] = test_policy_experience_args['memory_capacity'] / simulator_args['num_simulators']
agent_args['checkpoint_every'] = 10000
agent_args['save_param_histograms_every'] = 5000
agent_args['test_policy_in_the_beginning'] = True				

# experiment arguments
experiment_args = {}
experiment_args['num_train_iterations'] = 1000000
experiment_args['test_random_prob'] = 0.
experiment_args['test_init_policy_prob'] = 0.
experiment_args['test_policy_num_steps'] = 2000
experiment_args['show_predictions'] = False
experiment_args['multiplayer'] = False
experiment_args['meas_for_manual'] = [] # expected to be [AMMO2 AMMO3 AMMO4 AMMO5 AMMO6 AMMO7 WEAPON2 WEAPON3 WEAPON4 WEAPON5 WEAPON6 WEAPON7]
experiment_args['results_file'] = 'results.txt'
experiment_args['net_name'] = 'unknown_net'
experiment_args['num_predictions_to_show'] = 10
experiment_args['args_file'] = None


