'''
Implementation of a simplified agent - with expectation/action split, but without normalization in the action stream
'''

from __future__ import print_function
import numpy as np
import time
import tensorflow as tf
from . import tf_ops as my_ops
import os
import re
from .agent import Agent

class FuturePredictorAgentAdvantageNoNorm(Agent):
    
    def make_net(self, input_images, input_measurements, input_actions, input_objectives, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        
        self.fc_val_params = np.copy(self.fc_joint_params)
        self.fc_val_params['out_dims'][-1] = self.target_dim
        self.fc_adv_params = np.copy(self.fc_joint_params)
        self.fc_adv_params['out_dims'][-1] = len(self.net_discrete_actions) * self.target_dim
        p_img_conv = my_ops.conv_encoder(input_images, self.conv_params, 'p_img_conv', msra_coeff=0.9)
        p_img_fc = my_ops.fc_net(my_ops.flatten(p_img_conv), self.fc_img_params, 'p_img_fc', msra_coeff=0.9)
        p_meas_fc = my_ops.fc_net(input_measurements, self.fc_meas_params, 'p_meas_fc', msra_coeff=0.9)
        if isinstance(self.fc_obj_params, np.ndarray):
            p_obj_fc = my_ops.fc_net(input_objectives, self.fc_obj_params, 'p_obj_fc', msra_coeff=0.9)
            p_concat_fc = tf.concat([p_img_fc,p_meas_fc,p_obj_fc], 1)
        else:
            p_concat_fc = tf.concat([p_img_fc,p_meas_fc], 1)
            if self.random_objective_coeffs:
                raise Exception('Need fc_obj_params with randomized objectives')
            
        p_val_fc = my_ops.fc_net(p_concat_fc, self.fc_val_params, 'p_val_fc', last_linear=True, msra_coeff=0.9)
        p_adv_fc = my_ops.fc_net(p_concat_fc, self.fc_adv_params, 'p_adv_fc', last_linear=True, msra_coeff=0.9)
        
        adv_reshape = tf.reshape(p_adv_fc, [-1, len(self.net_discrete_actions), self.target_dim])
        
        #pred_all_nomean = adv_reshape - tf.reduce_mean(adv_reshape, reduction_indices=1, keep_dims=True)
        pred_all = adv_reshape + tf.reshape(p_val_fc, [-1, 1, self.target_dim])
        pred_relevant = tf.boolean_mask(pred_all, tf.cast(input_actions, tf.bool))
        
        return pred_all, pred_relevant
    
    def make_losses(self, pred_relevant, targets_preprocessed, objective_indices, objective_coeffs):
        # make a loss function and compute some summary numbers
        
        per_target_loss = my_ops.mse_ignore_nans(pred_relevant, targets_preprocessed, reduction_indices=0)
        loss = tf.reduce_sum(per_target_loss)
        
        # compute objective value, just for logging purposes
        # TODO add multiplication by the objective_coeffs (somehow not trivial)
        obj = tf.reduce_sum(self.postprocess_predictions(targets_preprocessed), 1)
        #obj = tf.sum(self.postprocess_predictions(targets_preprocessed[:,objective_indices]) * objective_coeffs[None,:], axis=1)
        obj_nonan = tf.where(tf.is_nan(obj), tf.zeros_like(obj), obj)
        num_valid_targets = tf.reduce_sum(1-tf.cast(tf.is_nan(obj), tf.float32))
        mean_obj = tf.reduce_sum(obj_nonan) / num_valid_targets
        
        # summaries
        obj_sum = tf.summary.scalar("objective_todo", mean_obj)
        #TODO
        per_target_loss_sums = []
        #per_target_loss_sums = [tf.summary.scalar(name, loss) for name,loss in zip(self.target_names,per_target_loss)]
        loss_sum = tf.summary.scalar("full loss", loss)
        
        #self.per_target_loss = tf.get_variable('avg_targets', [self.target_dim], initializer=tf.constant_initializer(value=0.))
        
        full_loss = loss
        errs_to_print = [loss]
        short_summary = [loss_sum]
        detailed_summary = per_target_loss_sums + [obj_sum]
        
        return full_loss, errs_to_print, short_summary, detailed_summary
        
    def act_net(self, state_imgs, state_meas, objective_coeffs):
        #Act given a state and objective_coeffs
        if objective_coeffs.ndim == 1:
            curr_objective_coeffs = np.tile(objective_coeffs[None,:],(state_imgs.shape[0],1))
        else:
            curr_objective_coeffs = objective_coeffs
        
        predictions = self.sess.run(self.pred_all, feed_dict={self.input_images: state_imgs, 
                                                              self.input_measurements: state_meas,
                                                              self.input_objective_coeffs: curr_objective_coeffs})
        
        self.curr_predictions = predictions[:,:,self.objective_indices]*curr_objective_coeffs[:,None,:]
        self.curr_objectives = np.sum(self.curr_predictions, axis=2)
        
        curr_action = np.argmax(self.curr_objectives, axis=1)
        return curr_action
        
