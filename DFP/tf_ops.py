import math
import numpy as np 
import tensorflow as tf

def msra_stddev(x, k_h, k_w): 
    return 1/math.sqrt(0.5*k_w*k_h*x.get_shape().as_list()[-1])

def mse_ignore_nans(preds, targets, **kwargs):
    #Computes mse, ignores targets which are NANs
    
    # replace nans in the target with corresponding preds, so that there is no gradient for those
    targets_nonan = tf.where(tf.is_nan(targets), preds, targets)
    return tf.reduce_mean(tf.square(targets_nonan - preds), **kwargs)

def conv2d(input_, output_dim, 
        k_h=3, k_w=3, d_h=2, d_w=2, msra_coeff=1,
        name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=msra_coeff * msra_stddev(input_, k_h, k_w)))
        b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))

        return tf.nn.bias_add(tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME'), b)

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def linear(input_, output_size, name='linear', msra_coeff=1):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name):
        w = tf.get_variable("w", [shape[1], output_size], tf.float32,
                                tf.random_normal_initializer(stddev=msra_coeff * msra_stddev(input_, 1, 1)))
        b = tf.get_variable("b", [output_size], initializer=tf.constant_initializer(0.0))
        return tf.matmul(input_, w) + b
    
def conv_encoder(data, params, name, msra_coeff=1):
    layers = []
    for nl, param in enumerate(params):
        if len(layers) == 0:
            curr_inp = data
        else:
            curr_inp = layers[-1]
            
        layers.append(lrelu(conv2d(curr_inp, param['out_channels'], k_h=param['kernel'], k_w=param['kernel'], d_h=param['stride'], d_w=param['stride'], name=name + str(nl), msra_coeff=msra_coeff)))
        
    return layers[-1]
        
def fc_net(data, params, name, last_linear = False, return_layers = [-1], msra_coeff=1):
    layers = []
    for nl, param in enumerate(params):
        if len(layers) == 0:
            curr_inp = data
        else:
            curr_inp = layers[-1]
        
        if nl == len(params) - 1 and last_linear:
            layers.append(linear(curr_inp, param['out_dims'], name=name + str(nl), msra_coeff=msra_coeff))
        else:
            layers.append(lrelu(linear(curr_inp, param['out_dims'], name=name + str(nl), msra_coeff=msra_coeff)))
            
    if len(return_layers) == 1:
        return layers[return_layers[0]]
    else:
        return [layers[nl] for nl in return_layers]

def flatten(data):
    return tf.reshape(data, [-1, np.prod(data.get_shape().as_list()[1:])])
