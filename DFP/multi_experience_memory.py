'''
Class for experience replay with multiple actors
'''

from __future__ import print_function
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import gridspec
import time
import os
from . import util as my_util

class MultiExperienceMemory:

    def __init__(self, args, multi_simulator = None, target_maker = None):
        ''' Initialize emtpy experience dataset. 
            Assumptions:
                - observations come in sequentially, and there is a terminal state in the end of each episode
                - every episode is shorter than the memory
        '''

        # params
        self.capacity = int(args['memory_capacity'])
        self.history_length = int(args['history_length'])
        self.history_step = int(args['history_step'])
        self.shared = args['shared']
        self.obj_shape = args['obj_shape']
        
        self.num_heads = int(multi_simulator.num_simulators)
        self.target_maker = target_maker
        self.head_offset = int(self.capacity/self.num_heads)
        
        self.img_shape = (multi_simulator.num_channels, multi_simulator.resolution[1], multi_simulator.resolution[0])
        self.meas_shape = (multi_simulator.num_meas,)
        self.action_shape = (multi_simulator.action_len,)
        self.state_imgs_shape = (self.history_length*self.img_shape[0],) +  self.img_shape[1:]
        self.state_meas_shape = (self.history_length*self.meas_shape[0],)

        # initialize dataset
        self.reset()
        
    def reset(self):
        self._images = my_util.make_array(shape=(self.capacity,) + self.img_shape, dtype=np.uint8, shared=self.shared, fill_val=0)
        self._measurements =  my_util.make_array(shape=(self.capacity,) + self.meas_shape, dtype=np.float32, shared=self.shared, fill_val=0.)
        self._rewards =  my_util.make_array(shape=(self.capacity,), dtype=np.float32, shared=self.shared, fill_val=0.)
        self._terminals =  my_util.make_array(shape=(self.capacity,), dtype=np.bool, shared=self.shared, fill_val=1)
        self._actions = my_util.make_array(shape=(self.capacity,) + self.action_shape, dtype=np.int, shared=self.shared, fill_val=0)
        self._objectives = my_util.make_array(shape=(self.capacity,) + self.obj_shape, dtype=np.float32, shared=self.shared, fill_val=0.)
        self._n_episode = my_util.make_array(shape=(self.capacity,), dtype=np.uint64, shared=self.shared, fill_val=0) # this is needed to compute future targets efficiently
        self._n_head = my_util.make_array(shape=(self.capacity,), dtype=np.uint64, shared=self.shared, fill_val=0) # this is needed to compute future targets efficiently

        self._curr_indices = np.arange(self.num_heads) * int(self.head_offset)
        self._episode_counts = np.zeros(self.num_heads)


    def add(self, imgs, meass, rwrds, terms, acts, objs=None, preds=None):
        ''' Add experience to dataset.

        Args:
            img: single observation frame           
            meas: extra measurements from the state
            rwrd: reward
            term: terminal state
            act: action taken
        '''

        self._images[self._curr_indices] = imgs
        self._measurements[self._curr_indices] = meass
        self._rewards[self._curr_indices] = rwrds
        self._terminals[self._curr_indices] = terms
        self._actions[self._curr_indices] = np.array(acts)
        if isinstance(objs, np.ndarray):
            self._objectives[self._curr_indices] = objs
        if isinstance(preds, np.ndarray):
            self._predictions[self._curr_indices] = preds
        self._n_episode[self._curr_indices] = self._episode_counts
        # this is a hack to simulate our version of ViZDoom which gives the agent the first post-mortem measurement. This turns out to matter a bit for learning
        terminated = np.where(np.array(terms) == True)[0]
        term_inds = self._curr_indices[terminated]
        self._measurements[term_inds] = self._measurements[(term_inds-1)%self.capacity]
        if self.meas_shape[0] == 1:
            for ti in term_inds:
                if self._measurements[ti,0] < 8.:
                    self._measurements[ti,0] -= 8.
        if self.meas_shape[0] == 3:
            for ti in term_inds:
                if self._measurements[ti,1] < 12.:
                    self._measurements[ti,1] -= 12.
        # in case there are 2 terminals in a row - not sure this actually can happen, but just in case
        prev_terminals = np.where(self._terminals[(term_inds-1)%self.capacity])[0]
        if len(prev_terminals) > 0:
            print('Prev terminals', prev_terminals)
        self._measurements[term_inds[prev_terminals]] = 0.
        # end of hack
        # self._n_episode[term_inds] = self._episode_counts[terminated]+100# so that the terminal step of the episode is not used as a target by target_maker
        self._n_head[self._curr_indices] = np.arange(self.num_heads)    
        
        self._episode_counts = self._episode_counts + (np.array(terms) == True)
        self._curr_indices = (self._curr_indices + 1) % self.capacity            
        self._terminals[self._curr_indices] = True # make the following state terminal, so that our current episode doesn't get stitched with the next one when sampling states
            
    def add_step(self, multi_simulator, acts = None, objs=None, preds=None):
        if acts == None:
            acts = multi_simulator.get_random_actions()
        self.add(*(multi_simulator.step(acts) +  (acts,objs,preds)))
        
    def add_n_steps_with_actor(self, multi_simulator, num_steps, actor, verbose=False, write_predictions=False, write_logs = False, global_step=0):
        ns = 0
        last_meas = np.zeros((multi_simulator.num_simulators,) + self.meas_shape)
        if write_predictions and not hasattr(self,'_predictions'):
            self._predictions = my_util.make_array(shape=(self.capacity,) + actor.predictions_shape, dtype=np.float32, shared=self.shared, fill_val=0.)
        if verbose or write_logs:
            start_time = time.time()
        #write_logs = False
        if write_logs:
            log_dir = os.path.dirname(self.log_prefix)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            log_brief = open(self.log_prefix + '_brief.txt','a')
            log_detailed = open(self.log_prefix + '_detailed.txt','a')
            log_detailed.write('Step {0}\n'.format(global_step))
            start_times = time.time() * np.ones(multi_simulator.num_simulators)
            num_episode_steps = np.zeros(multi_simulator.num_simulators)
            accum_rewards = np.zeros(multi_simulator.num_simulators)
            accum_meas = np.zeros((multi_simulator.num_simulators,) + self.meas_shape)
            total_final_meas = np.zeros(self.meas_shape)
            total_avg_meas = np.zeros(self.meas_shape)
            total_accum_reward = 0
            total_start_time = time.time()
            num_episodes = 0
            meas_dim = np.prod(self.meas_shape)
            log_brief_format = ' '.join([('{' + str(n) + '}') for n in range(5)]) + ' | ' + \
                       ' '.join([('{' + str(n+5) + '}') for n in range(meas_dim)]) + ' | ' + \
                       ' '.join([('{' + str(n+5+meas_dim) + '}') for n in range(meas_dim)]) + '\n'
            log_detailed_format = ' '.join([('{' + str(n) + '}') for n in range(4)]) + ' | ' + \
                          ' '.join([('{' + str(n+4) + '}') for n in range(meas_dim)]) + ' | ' + \
                          ' '.join([('{' + str(n+4+meas_dim) + '}') for n in range(meas_dim)]) + '\n'
        for ns in range(int(num_steps)):
            if verbose and time.time() - start_time > 2:
                print('%d/%d' % (ns * multi_simulator.num_simulators, num_steps * multi_simulator.num_simulators))
                start_time = time.time()

            curr_act = actor.act_with_multi_memory(self)
            
            # actor has to return a np array of bools
            invalid_states = np.logical_not(np.array(self.curr_states_with_valid_history()))
            if actor.random_objective_coeffs:
                actor.reset_objective_coeffs(np.where(invalid_states)[0].tolist())
            curr_act[invalid_states] = actor.random_actions(np.sum(invalid_states))
            
            if write_predictions:
                self.add_step(multi_simulator, curr_act.tolist(), actor.objectives_to_write(), actor.curr_predictions)
            else:
                self.add_step(multi_simulator, curr_act.tolist(), actor.objectives_to_write())
            if write_logs:
                last_indices = np.array(self.get_last_indices())
                last_rewards = self._rewards[last_indices]
                prev_meas = last_meas
                last_meas = self._measurements[last_indices]
                last_terminals = self._terminals[last_indices]
                last_meas[np.where(last_terminals)[0]] = 0
                accum_rewards += last_rewards
                accum_meas += last_meas
                num_episode_steps = num_episode_steps + 1
                terminated_simulators = list(np.where(last_terminals)[0])
                for ns in terminated_simulators:
                    num_episodes += 1
                    episode_time = time.time() - start_times[ns]
                    avg_meas = accum_meas[ns]/float(num_episode_steps[ns])
                    total_avg_meas += avg_meas
                    total_final_meas += prev_meas[ns]
                    total_accum_reward += accum_rewards[ns]
                    start_times[ns] = time.time()
                    log_detailed.write(log_detailed_format.format(*([num_episodes, num_episode_steps[ns], episode_time, accum_rewards[ns]] + list(prev_meas[ns]) + list(avg_meas))))
                    accum_meas[ns] = 0
                    accum_rewards[ns] = 0
                    num_episode_steps[ns] = 0
                    start_times[ns] = time.time()
        if write_logs:  
            if num_episodes == 0:
                num_episodes = 1
            log_brief.write(log_brief_format.format(*([global_step, time.time(), time.time() - total_start_time, num_episodes, total_accum_reward/float(num_episodes)] +
                                 list(total_final_meas / float(num_episodes)) + list(total_avg_meas / float(num_episodes)))))
            log_brief.close()
            log_detailed.close()

    def get_states(self, indices):
        
        frames = np.zeros(len(indices)*self.history_length, dtype=np.int64)
        for (ni,index) in enumerate(indices):
            frame_slice = np.arange(int(index) - self.history_length*self.history_step + 1, (int(index) + 1), self.history_step) % self.capacity
            frames[ni*self.history_length:(ni+1)*self.history_length] = frame_slice
            
        state_imgs = np.transpose(np.reshape(np.take(self._images, frames, axis=0), (len(indices),) + self.state_imgs_shape), [0,2,3,1]).astype(np.float32)
        state_meas = np.reshape(np.take(self._measurements, frames, axis=0), (len(indices),) + self.state_meas_shape).astype(np.float32)
        return state_imgs, state_meas
            
    def get_current_state(self):
        '''  Return most recent observation sequence '''
        return self.get_states(list((self._curr_indices-1)%self.capacity))
    
    def get_last_indices(self):
        '''  Return most recent indices '''
        return list((self._curr_indices-1)%self.capacity)
    
    def get_targets(self, indices):
        # TODO this 12345678 is a hack, but should be good enough
        return self.target_maker.make_targets(indices, self._measurements, self._rewards, self._n_episode + 12345678*self._n_head)      
    
    def has_valid_history(self, index):
        return (not self._terminals[np.arange(int(index) - self.history_length*self.history_step + 1, int(index)+1) % self.capacity].any())
    
    def curr_states_with_valid_history(self):
        return [self.has_valid_history((ind - 1)%self.capacity) for ind in list(self._curr_indices)]
    
    def has_valid_target(self, index):
        return (not self._terminals[np.arange(index, index+self.target_maker.min_future_frames+1) % self.capacity].any())
    
    def is_valid_state(self, index):
        return self.has_valid_history(index) and self.has_valid_target(index)
    
    def get_observations(self, indices):
        indices_arr = np.array(indices)
        state_imgs, state_meas = self.get_states((indices_arr - 1) % self.capacity)
        rwrds = self._rewards[indices_arr]
        acts = self._actions[indices_arr]       
        terms = self._terminals[indices_arr].astype(int)
        targs = self.get_targets((indices_arr - 1) % self.capacity)
        if isinstance(self._objectives, np.ndarray):
            objs = self._objectives[indices_arr]
        else:
            objs = None     
        
        return state_imgs, state_meas, rwrds, terms, acts, targs, objs

    def get_random_batch(self, batch_size):
        ''' Sample minibatch of experiences for training '''

        samples = [] # indices of the end of each sample

        while len(samples) < batch_size:
            index = random.randrange(self.capacity)
            # check if there is enough history to make a state and enough future to make targets
            if self.is_valid_state(index):
                samples.append(index)
            else:
                continue

        # create batch
        return self.get_observations(np.array(samples))
    
    def compute_avg_meas_and_rwrd(self, start_idx, end_idx):
        # compute average measurement values per episode, and average cumulative reward per episode
        curr_num_obs = 0.
        curr_sum_meas = self._measurements[0] * 0
        curr_sum_rwrd = self._rewards[0] * 0
        num_episodes = 0.
        total_sum_meas = self._measurements[0] * 0
        total_sum_rwrd = self._rewards[0] * 0
        for index in range(int(start_idx), int(end_idx)):
            curr_sum_rwrd += self._rewards[index]
            if self._terminals[index]:
                if curr_num_obs:
                    total_sum_meas += curr_sum_meas / curr_num_obs
                    total_sum_rwrd += curr_sum_rwrd
                    num_episodes += 1
                curr_sum_meas = self._measurements[0] * 0
                curr_sum_rwrd = self._rewards[0] * 0
                curr_num_obs = 0.
            else:               
                curr_sum_meas += self._measurements[index]
                curr_num_obs += 1
                
        if num_episodes == 0.:
            total_avg_meas = curr_sum_meas / curr_num_obs
            total_avg_rwrd = curr_sum_rwrd
        else:
            total_avg_meas = total_sum_meas / num_episodes
            total_avg_rwrd = total_sum_rwrd / num_episodes
        
        return total_avg_meas, total_avg_rwrd
            
            
    def show(self, start_index=0, end_index=None, display=True, write_imgs=False, write_video = False, preprocess_targets=None, show_predictions=0, net_discrete_actions = []):
        if show_predictions:
            assert(hasattr(self,'_predictions'))
        curr_index = start_index
        if not end_index:
            end_index = start_index
        inp = ''
        if write_imgs:
            os.makedirs('imgs')
            prev_time = time.time()
        if write_video:
            print(self.img_shape)
            vw = VideoWriter('vid.avi', (self.img_shape[2],self.img_shape[1]), framerate=24, rgb=(self.img_shape[0]==3), mode='replace')
        print('Press ENTER to go to the next observation, type "quit" or "q" or "exit" and press ENTER to quit')

        if display or write_imgs:
            fig_img = plt.figure(figsize=(10, 7), dpi=50, tight_layout=True)
            ax_img = plt.gca()
        if display:
            fig_img.show()
        ns = 0
        if show_predictions and len(net_discrete_actions):
            action_labels = []
            for act in net_discrete_actions:
                action_labels.append(''.join(str(int(i)) for i in act))
                
        while True:
            curr_img = np.transpose(self._images[curr_index], (1,2,0))
            if curr_img.shape[2] == 1:
                curr_img = np.tile(curr_img, (1,1,3))
            if show_predictions:
                preds = self._predictions[curr_index]
                objs = np.sum(preds, axis=1)
                objs_argsort = np.argsort(-objs)
                curr_preds = np.transpose(preds[objs_argsort[:show_predictions]])
                curr_labels = [action_labels[i] for i in objs_argsort[:show_predictions]]
                
            if curr_index == start_index:
                if display or write_imgs:
                    im = ax_img.imshow(curr_img)
                    txt = ax_img.text(self.img_shape[2] - 10*len(self._measurements[curr_index]) , self.img_shape[1] - 5, str(self._measurements[curr_index]), fontsize=20, color='red')
                if show_predictions:
                    all_objs = np.sum(self._predictions, axis=2)
                    sbp = my_util.StackedBarPlot(curr_preds, ylim=[np.min(all_objs), np.max(all_objs)], labels=curr_labels)
                    del all_objs
                    if display:
                        sbp.show()
            else:
                if display or write_imgs:
                    im.set_data(curr_img)
                    txt.set_text(str(self._measurements[curr_index]))
                if show_predictions:
                    sbp.set_data(curr_preds, labels = curr_labels)
            if write_imgs:
                plt.savefig('imgs/%.5d.png' % curr_index, dpi=50)
                if time.time() - prev_time > 2:
                    print('Wrote %d images' % ns)
                    prev_time = time.time()
            if write_video:
                vw.add_frame(curr_img[:,:,0])
            if display:
                fig_img.canvas.draw()
                if show_predictions:
                    sbp.draw()
                print('Index', curr_index)
                print('Measurements:', self._measurements[curr_index])
                print('Rewards:', self._rewards[curr_index])
                print('Action:', self._actions[curr_index])
                print('Terminal:', self._terminals[curr_index])
                inp = input()
                
            curr_index = (curr_index + 1) % self.capacity
            if curr_index == end_index:
                if write_video:
                    vw.close()
                break
            if inp == 'q' or inp == 'quit' or inp == 'exit':
                break
            ns += 1
            

        
        
        
        
