'''
FutureTargetMaker makes targets for future prediction
'''

from __future__ import print_function
import numpy as np
import random
import matplotlib.pyplot as plt
import time


class FutureTargetMaker:

    def __init__(self, args):
        ''' Object which makes targets of future measurements  
            Parameters:
                future_steps - list of future steps to use as targets (e.g [1,2,4] means we'll use steps [i+1, i+2, i+4] as targets for step i)
                min_num_targs - minimum number of targets we want to have (matters when close to the end of the episode), non-available targets replaced by NANs 
                meas_to_predict - which of the measurements should go to the target
        '''

        # params
        self.future_steps = np.array(sorted(args['future_steps'])).astype(np.uint32)
        self.meas_to_predict = np.array(args['meas_to_predict']).astype(np.uint32)
        self.min_num_targs = args['min_num_targs']
        self.rwrd_schedule_type = args['rwrd_schedule_type']
        self.invalid_targets_replacement = args['invalid_targets_replacement']
        
        if self.rwrd_schedule_type == 'exp':
            gammas = np.array(args['gammas'])
            self.rwrd_schedules = np.power(np.reshape(gammas, (-1,1)), np.reshape(np.arange(self.future_steps[-1]), (1,-1)))
        else:
            print('Unknown rwrd_schedule_type', self.rwrd_schedule_type)
            raise
        
        self.num_reward_targets = self.rwrd_schedules.shape[0]
        self.num_targets = len(self.meas_to_predict) + self.num_reward_targets
        self.target_dim = self.num_targets * len(self.future_steps)
        
        self.target_names = []
        for meas in self.meas_to_predict:
            for step in self.future_steps:
                self.target_names.append('meas%.1d_%.3d' % (meas, step))
                
        for nsched in range(self.rwrd_schedules.shape[0]):
            for step in self.future_steps:
                self.target_names.append('rwrd%.1d_%.3d' % (nsched, step))
        
        # experience memory needs to know this to know which indices are valid
        if self.min_num_targs == 0:
            self.min_future_frames = 0
        else:
            self.min_future_frames = self.future_steps[self.min_num_targs-1] 
    
    def make_targets(self, indices, meas, rwrds, n_episode, meas_mean = None, meas_std = None):
        ''' Make targets of future values 
            Args:
            
            So we only do targets if at least min_num_targs future steps are available, and we replace with nan's any unavailable observations
            
            Known bug: if experience memory is comparable in size with our future horizon for predicting, bad things may happen when going through the end of memory
        '''
        capacity = meas.shape[0]
        targets = np.zeros((len(indices), self.num_targets, len(self.future_steps)), dtype='float32')
        for ns, sample in enumerate(indices):
            # measurement targets
            curr_future_steps = (sample + self.future_steps) % capacity
            if isinstance(meas_mean, np.ndarray) and isinstance(meas_std, np.ndarray):
                targets[ns, :len(self.meas_to_predict), :] = ((meas[curr_future_steps][:, self.meas_to_predict] - meas[None,sample][:, self.meas_to_predict])/meas_std[:, self.meas_to_predict]).transpose()
            else:
                targets[ns, :len(self.meas_to_predict), :] = (meas[curr_future_steps][:, self.meas_to_predict] - meas[None,sample][:, self.meas_to_predict]).transpose()
            invalid_samples = (n_episode[curr_future_steps] != n_episode[sample])
            if self.invalid_targets_replacement == 'nan':
                targets[ns, :len(self.meas_to_predict), invalid_samples] = np.nan # make invalid the entries which fall into the following episodes
            elif self.invalid_targets_replacement == 'last_valid':
                targets[ns, :len(self.meas_to_predict), invalid_samples] = targets[ns, :len(self.meas_to_predict), np.argmax(invalid_samples)-1] # make invalid the entries which fall into the following episodes
            else:
                raise ArgumentException('Unknown invalid_targets_replacement')
            
            # reward targets
            curr_future_window = np.arange(sample+1,sample+self.future_steps[-1]+1) % capacity
            rwrds_cumsum = np.cumsum(np.reshape(rwrds[curr_future_window], (1,-1)) * self.rwrd_schedules, axis=1)
            invalid_samples = (n_episode[curr_future_window] != n_episode[sample])
            end_of_episode = np.argmax(invalid_samples) - 1
            rwrds_cumsum[:, invalid_samples] = rwrds_cumsum[:,end_of_episode,np.newaxis]
            targets[ns, len(self.meas_to_predict):, :] = rwrds_cumsum[:,self.future_steps - 1]
            
        return np.reshape(targets, (len(indices), self.target_dim))
        
        
        
        