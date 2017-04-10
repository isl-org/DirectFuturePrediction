'''
Base class for agents
'''

from __future__ import print_function
import numpy as np
import time
import tensorflow as tf
import os
import re
import itertools as it
from . import util as my_util

class Agent:

    def __init__(self, sess, args):
        '''Agent - powered by neural nets, can infer, act, train, test.
        '''
        self.sess = sess
        
        # input data properties
        self.state_imgs_shape = args['state_imgs_shape']
        self.state_meas_shape = args['state_meas_shape']
        self.obj_shape = args['obj_shape']
        self.num_simulators = args['num_simulators']
        self.meas_for_net = args['meas_for_net']
        self.meas_for_manual = args['meas_for_manual']
        self.target_dim = args['target_dim']
        self.target_names = args['target_names']
        self.discrete_controls = args['discrete_controls']
        self.discrete_controls_manual = args['discrete_controls_manual'] # controls to be set "manually" (that is, by a user-specified function)
        self.opposite_button_pairs = args['opposite_button_pairs']
        self.prepare_controls_and_actions()
        
        # preprocessing
        self.preprocess_input_images = args['preprocess_input_images']
        self.preprocess_input_measurements = args['preprocess_input_measurements']
        self.preprocess_input_targets = args['preprocess_input_targets']
        self.postprocess_predictions = args['postprocess_predictions']
        
        # agent properties
        self.objective_coeffs_temporal = args['objective_coeffs_temporal']
        self.objective_coeffs_meas = args['objective_coeffs_meas']
        self.objective_indices = args['objective_indices']
        self.objective_coeffs = args['objective_coeffs']
        self.random_exploration_schedule = args['random_exploration_schedule']
        self.new_memories_per_batch = args['new_memories_per_batch']
        self.add_experiences_every = args['add_experiences_every']
        self.random_objective_coeffs = args['random_objective_coeffs']
        self.objective_coeffs_temporal = args['objective_coeffs_temporal']
        self.objective_coeffs_distribution = args['objective_coeffs_distribution']
        
        # net parameters
        self.conv_params = args['conv_params']
        self.fc_img_params = args['fc_img_params']
        self.fc_meas_params = args['fc_meas_params']
        self.fc_joint_params = args['fc_joint_params']
        self.fc_obj_params = args['fc_obj_params']
        self.weight_decay = args['weight_decay']
        
        # optimization parameters
        self.batch_size = args['batch_size']        
        self.init_learning_rate = args['init_learning_rate']
        self.lr_step_size = args['lr_step_size']
        self.lr_decay_factor = args['lr_decay_factor']
        self.adam_beta1 = args['adam_beta1']
        self.adam_epsilon = args['adam_epsilon']        
        self.optimizer = args['optimizer']      
        self.reset_iter_count = args['reset_iter_count']
        self.clip_gradient = args['clip_gradient']
        
        # directories       
        self.checkpoint_dir = args['checkpoint_dir']
        self.log_dir = args['log_dir']
        self.init_model = args['init_model']
        self.model_dir = args['model_dir']
        self.model_name = args['model_name']        
        
        # logging and testing
        self.print_err_every = args['print_err_every']
        self.detailed_summary_every = args['detailed_summary_every']
        self.checkpoint_every = args['checkpoint_every']
        self.test_policy_every = args['test_policy_every']          
        self.num_steps_per_policy_test = args['num_steps_per_policy_test']
        self.save_param_histograms_every = args['save_param_histograms_every']          
        
        if self.reset_iter_count or (not self.set_init_step(self.init_model)):
            self.curr_step = 0
        
        self.curr_objectives = None
        self.curr_predictions = None
            
        self.build_model()      
        
    def prepare_controls_and_actions(self):
        # generate the list of available actions
        self.discrete_controls_to_net = np.array([i for i in range(len(self.discrete_controls)) if not i in self.discrete_controls_manual])
        self.num_manual_controls = len(self.discrete_controls_manual)
        
        self.net_discrete_actions = []
        
        if not self.opposite_button_pairs:
            # the set of actions is 2^{the set of buttons}
            for perm in it.product([False, True], repeat=len(self.discrete_controls_to_net)):
                self.net_discrete_actions.append(list(perm))
        else:
            # if given the indices of opposite buttons (like left and right), remove actions where both opposite buttons are pressed 
            for perm in it.product([False, True], repeat=len(self.discrete_controls_to_net)):
                act = list(perm)
                valid = True
                for bp in self.opposite_button_pairs:
                    if act[bp[0]] == act[bp[1]] == True:
                        valid=False
                if valid:
                    self.net_discrete_actions.append(act)
                    
        self.num_net_discrete_actions = len(self.net_discrete_actions)
        self.action_to_index = {tuple(val):ind for (ind,val) in enumerate(self.net_discrete_actions)}
        self.net_discrete_actions = np.array(self.net_discrete_actions)
        self.onehot_discrete_actions = np.eye(self.num_net_discrete_actions)
        
    def preprocess_actions(self, acts):
        #Preprocess the action before feeding it to the net (remove "manual" controls and turn to one-hot)
        to_net_acts = acts[:,self.discrete_controls_to_net]
        return self.onehot_discrete_actions[np.array([self.action_to_index[tuple(act)] for act in to_net_acts.tolist()])]
        
    def postprocess_actions(self, acts_net, acts_manual=[]):
        # make a full action from two: the one suggested by the net and the "manual" one
        out_actions = np.zeros((acts_net.shape[0], len(self.discrete_controls)), dtype=np.int)
        out_actions[:,self.discrete_controls_to_net] = self.net_discrete_actions[acts_net]
        if len(acts_manual):
            out_actions[:,self.discrete_controls_manual] = acts_manual
        return out_actions
    
    def random_actions(self, num_samples):
        # return random actions
        acts_net = np.random.randint(0, self.num_net_discrete_actions, num_samples)
        acts_manual = np.zeros((num_samples, self.num_manual_controls), dtype=np.bool)
        return self.postprocess_actions(acts_net, acts_manual)
        
    def make_net(self, input_images, input_measurements, input_actions, input_objective_coeffs):
        # Build the net which will perform the future prediction. Returns pred_all (a vector of predicted future values for all actions) and pred_relevant (a vector of predicted future values for the action which was actually taken)
        raise NotImplementedError( "Agent should implement make_net" )
    
    def make_losses(self, pred_relevant, targets_preprocessed, objective_indices, objective_coeffs):
        # makes the loss function
        raise NotImplementedError( "Agent should implement make_losses" )
    
    def build_model(self):
        # prepare the data
        self.input_images = tf.placeholder(tf.float32, [None] + [self.state_imgs_shape[1], self.state_imgs_shape[2], self.state_imgs_shape[0]],
                                    name='input_images')
        self.input_measurements = tf.placeholder(tf.float32, [None] + list(self.state_meas_shape),
                                    name='input_measurements')
        self.input_targets = tf.placeholder(tf.float32, [None, self.target_dim],
                                    name='input_targets')
        self.input_actions = tf.placeholder(tf.float32, [None, self.num_net_discrete_actions],
                                    name='input_actions')
        self.input_objective_coeffs = tf.placeholder(tf.float32, [None] + list(self.obj_shape),
                                    name='input_objective_coeffs')
        
        if self.preprocess_input_images:
            self.input_images_preprocessed = self.preprocess_input_images(self.input_images)
        if self.preprocess_input_measurements:
            self.input_measurements_preprocessed = self.preprocess_input_measurements(self.input_measurements)
        if self.preprocess_input_targets:
            self.input_targets_preprocessed = self.preprocess_input_targets(self.input_targets)
        if self.postprocess_predictions:
            test_arr = 1.23*np.ones((2,self.target_dim))
            assert(np.all(abs(self.postprocess_predictions(self.preprocess_input_targets(test_arr)) - test_arr) < 1e-8))
        
        # make the actual net
        self.pred_all, self.pred_relevant = self.make_net(self.input_images_preprocessed, self.input_measurements_preprocessed, self.input_actions, self.input_objective_coeffs) 
        self.full_loss, self.errs_to_print, self.short_summary, self.detailed_summary = self.make_losses(self.pred_relevant, self.input_targets_preprocessed, self.objective_indices, self.objective_coeffs)
        
        # make the saver, lr and param summaries
        self.saver = tf.train.Saver()
        if not hasattr(self, 'saver_init'):
            self.saver_init = self.saver
        
        self.tf_step = tf.Variable(self.curr_step, trainable=False)
        self.tf_learning_rate = tf.train.exponential_decay(self.init_learning_rate, self.tf_step, self.lr_step_size, self.lr_decay_factor, staircase=True)

        if self.optimizer == 'Adam':
            self.tf_optim = tf.train.AdamOptimizer(self.tf_learning_rate, beta1=self.adam_beta1, epsilon=self.adam_epsilon)
        elif self.optimizer == 'SGD':
            self.tf_optim = tf.train.GradientDescentOptimizer(self.tf_learning_rate)
        else:
            print('Unknown optimizer', self.optimizer)
            raise
        
        if not hasattr(self, 't_vars'):
            self.t_vars = tf.trainable_variables()
            
        # weight decay
        self.weights_norms = tf.reduce_sum(input_tensor = self.weight_decay * tf.stack([tf.nn.l2_loss(w) for w in self.t_vars]), name='weights_norm')        
        self.loss_with_weight_decay = self.full_loss + self.weights_norms
        
        if self.clip_gradient:
            grads, grad_norm = tf.clip_by_global_norm(tf.gradients(self.loss_with_weight_decay, self.t_vars), self.clip_gradient)
            self.detailed_summary += [tf.summary.scalar("gradient norm", grad_norm)]
            grads_and_vars = zip(grads,self.t_vars)
        else:
            grads_and_vars = self.tf_optim.compute_gradients(self.loss_with_weight_decay, var_list=self.t_vars)
        
        self.tf_minim = self.tf_optim.apply_gradients(grads_and_vars, global_step=self.tf_step)
        
        param_hists = [tf.summary.histogram(gv[1].name, gv[1]) for gv in grads_and_vars]
        grad_hists = [tf.summary.histogram(gv[1].name + '/gradients', gv[0]) for gv in grads_and_vars]
        
        self.short_summary = tf.summary.merge(self.short_summary)
        self.detailed_summary = tf.summary.merge(self.detailed_summary + [tf.summary.scalar("learning rate", self.tf_learning_rate)])
        self.param_summary = tf.summary.merge(param_hists + grad_hists)
            
        tf.global_variables_initializer().run(session=self.sess)
    
    def act(self, state_imgs, state_meas, objective_coeffs):
        return self.postprocess_actions(self.act_net(state_imgs, state_meas[:,self.meas_for_net], objective_coeffs), self.act_manual(state_meas[:,self.meas_for_manual]))
    
    def act_net(self, state_imgs, state_meas, objective_coeffs):
        raise NotImplementedError( "Agent should implement act_net, which takes the input state and outputs an action" )
    
    def act_manual(self, state_meas):
        return []
    
    class Actor:
        # a small interface which will actually be used for taking actions
        def __init__(self, agent, objective_coeffs, random_prob, random_objective_coeffs):
            if objective_coeffs is None:
                assert(random_prob == 1.)
            self.agent = agent
            self.random_objective_coeffs = random_objective_coeffs
            if self.random_objective_coeffs:
                self.objective_coeffs = np.zeros((self.agent.num_simulators,) + self.agent.obj_shape)
                self.objective_coeffs_temporal = self.agent.objective_coeffs_temporal
                self.num_objective_coeffs_meas = len(self.agent.objective_coeffs_meas)
                self.reset_objective_coeffs(range(self.objective_coeffs.shape[0]))
            else:
                self.objective_coeffs = objective_coeffs

            self.random_prob = random_prob
            self.predictions_shape = (self.agent.num_net_discrete_actions,) + self.agent.obj_shape
            
        def reset_objective_coeffs(self,indices):
        # reset objective coeffs to random values
            if self.random_objective_coeffs:
                if self.agent.objective_coeffs_distribution == 'uniform_pos':
                    for ind in indices:
                        _, self.objective_coeffs[ind] = my_util.make_objective_indices_and_coeffs(self.objective_coeffs_temporal, np.random.rand(self.num_objective_coeffs_meas))
                elif self.agent.objective_coeffs_distribution == 'uniform_pos_neg':
                    for ind in indices:
                        _, self.objective_coeffs[ind] = my_util.make_objective_indices_and_coeffs(self.objective_coeffs_temporal, 2*(np.random.rand(self.num_objective_coeffs_meas)-0.5))
                else:
                    raise Exception('Unknown objective_coeffs_distribution', self.objective_coeffs_distribution)
            else:
                print('Not resetting the objectives')
            #if len(indices) > 0:
            #    print(self.objective_coeffs)
                    
        def objectives_to_write(self):
            if self.random_objective_coeffs:
                return self.objective_coeffs
            else:
                return None
                
        def act(self, state_imgs, state_meas):
            if np.random.rand() < self.random_prob:
                self.curr_predictions = None
                return self.agent.random_actions(1)
            else:
                curr_act  = self.agent.act(state_imgs, state_meas, self.objective_coeffs)
                self.curr_predictions = self.agent.curr_predictions
                return curr_act
            
        def act_with_multi_memory(self, multi_memory):
            # this is more efficient because reads the state only if it is needed (for a non-random action)
            if np.random.rand() < self.random_prob:
                self.curr_predictions = None
                curr_act = self.agent.random_actions(multi_memory.num_heads)
            else:
                state_imgs, state_meas = multi_memory.get_current_state()
                curr_act = self.agent.act(state_imgs, state_meas, self.objective_coeffs)
                self.curr_predictions = self.agent.curr_predictions
            return curr_act
        
        def random_actions(self, num_samples):
            return self.agent.random_actions(num_samples)
    ###
    
    def get_actor(self, objective_coeffs=None, random_prob=1., random_objective_coeffs=False):
        return Agent.Actor(self, objective_coeffs, random_prob, random_objective_coeffs)
    
    
    def train_one_batch(self, experience):
        state_imgs, state_meas, rwrds, terms, acts, targs, objs = experience.get_random_batch(self.batch_size)
        acts = self.preprocess_actions(acts)
        res = self.sess.run([self.tf_minim, self.short_summary, self.detailed_summary] + self.errs_to_print,
                        feed_dict={ self.input_images: state_imgs, \
                                    self.input_measurements: state_meas, \
                                    self.input_targets: targs, \
                                    self.input_actions: acts, \
                                    self.input_objective_coeffs: objs})
                                                                
        curr_short_summary = res[1]
        curr_detailed_summary = res[2]
        curr_errs = res[3:]
        
        if np.mod(self.curr_step, self.print_err_every) == 0:
            print(time.strftime("[%Y/%m/%d %H:%M:%S] ") + "[Batch %4d] time: %4.4f, losses: " \
                % (self.curr_step,  time.time() - self.prev_time), curr_errs)
            self.prev_time = time.time()
            self.writer.add_summary(curr_short_summary, self.curr_step)
            
        if np.mod(self.curr_step, self.detailed_summary_every) == 0:
            self.writer.add_summary(curr_detailed_summary, self.curr_step)
            
        if self.save_param_histograms_every and np.mod(self.curr_step, self.save_param_histograms_every) == 0:
            summary_string = self.sess.run(self.param_summary, feed_dict={ self.input_images: state_imgs, \
                                                                        self.input_measurements: state_meas, \
                                                                        self.input_targets: targs, \
                                                                        self.input_actions: acts, \
                                                                        self.input_objective_coeffs: objs})
            self.writer.add_summary(summary_string, self.curr_step)     
            
        self.curr_step += 1
        
        
    def train(self, simulator, experience, num_steps, test_policy_experience=None):
        # load the model if available and initialize variables
        if self.init_model:
            if self.load(self.init_model, init=True):
                print('Loaded a model from', self.init_model)
            else:
                print('Could not load a model from', self.init_model)
        else:
            print('No model to load')
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(os.path.join(self.log_dir, self.model_dir)):
            os.makedirs(os.path.join(self.log_dir, self.model_dir))
            
        self.writer = tf.summary.FileWriter(os.path.join(self.log_dir, self.model_dir), self.sess.graph)
        self.prev_time = time.time()
            
        self.train_actor = self.get_actor(objective_coeffs=self.objective_coeffs, 
                                          random_prob=self.random_exploration_schedule(self.curr_step),random_objective_coeffs=self.random_objective_coeffs)
        
        print('Filling the training memory')
        experience.add_n_steps_with_actor(simulator, 
                                          experience.capacity / simulator.num_simulators, 
                                          self.train_actor, verbose=True)               
        
        for _ in range(num_steps):
            if np.mod(self.curr_step, self.checkpoint_every) == 0:
                self.save(self.checkpoint_dir, self.curr_step)
                
            if self.test_policy_every and np.mod(self.curr_step, self.test_policy_every) == 0:
                self.test_policy(simulator, test_policy_experience, self.objective_coeffs, self.num_steps_per_policy_test, random_prob=0., write_summary=True)
            
            self.train_one_batch(experience)
            if np.mod(self.curr_step, self.add_experiences_every) == 0:
                self.train_actor.random_prob = self.random_exploration_schedule(self.curr_step)
                experience.add_n_steps_with_actor(simulator, 
                                                self.new_memories_per_batch, 
                                                self.train_actor)
    
    def test_policy(self, simulator, experience, objective_coeffs, num_steps, random_prob=0., write_summary=False, write_predictions=False):
        print('== Testing the policy ==')
        old_head_offset = experience.head_offset
        experience.head_offset = num_steps + 1
        experience.reset()
        actor = self.get_actor(objective_coeffs=objective_coeffs, random_prob=random_prob, random_objective_coeffs=False)
        experience.add_n_steps_with_actor(simulator, num_steps, actor, verbose=True, write_predictions=write_predictions, write_logs=True, global_step=self.curr_step*self.batch_size)
        total_avg_meas, total_avg_rwrd = experience.compute_avg_meas_and_rwrd(0, num_steps*simulator.num_simulators)
        print('Average mean measurements  per episode:', total_avg_meas, '\nAverage reward per episode:', total_avg_rwrd)
            
        if write_summary:
            pass
        #TODO
            #tf_avg_meas = tf.placeholder(tf.float32, total_avg_meas.shape)
            #tf_avg_meas_sum = tf.summary.merge([tf.summary.scalar("test_avg_" + tag, tf_avg_meas) for tag,meas in simulator.simulators[0].meas_tags
            #tf_avg_meas_summary = self.sess.run([tf_avg_meas_sum], feed_dict = {tf_avg_meas: total_avg_meas})
            #self.writer.add_summary(tf_avg_meas_summary[0], self.curr_step)
            
            #tf_avg_rwrd = tf.placeholder(tf.float32, total_avg_rwrd.shape)
            #tf_avg_rwrd_sum = tf.summary.scalar("test_avg_rwrd", tf_avg_rwrd)
            #tf_avg_rwrd_summary = self.sess.run([tf_avg_rwrd_sum], feed_dict = {tf_avg_rwrd: total_avg_rwrd})
            #self.writer.add_summary(tf_avg_rwrd_summary[0], self.curr_step)
            
        experience.head_offset = old_head_offset
        return None

    
    def save(self, checkpoint_dir, step):       
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, self.model_name),
                        global_step=step)

    def load(self, checkpoint_dir, init=False):
        if init:
            curr_saver = self.saver_init
        else:
            curr_saver = self.saver
            
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            curr_saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
        
    def set_init_step(self, checkpoint_dir):
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.curr_step = int(re.search('-(\d*)?', ckpt_name).group(1))
            print('Continuing from step ', self.curr_step)
            return True
        else:
            return False
        
        
