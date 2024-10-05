import tensorflow as tf
import numpy as np
from math import ceil
 
 
def Conv(name, x, filter_size, in_filters, out_filters, strides, padding):
    with tf.variable_scope(name):
        kernel = tf.get_variable('filter', [filter_size, filter_size, in_filters, out_filters], tf.float32,
                                 initializer=tf.random_normal_initializer(stddev=0.01))
        bias = tf.get_variable('bias', [out_filters], tf.float32, initializer=tf.zeros_initializer())
        
        return tf.nn.conv2d(x, kernel, [1, strides, strides, 1], padding=padding) + bias

#双线插值法求kernel
def upsampling_bilinear(filter_size, out_filters, in_filters):
    width = filter_size

    height = filter_size
    f = ceil(width/2)
    c = (2*f-1-f%2)/(2.0*f)

    bilinear = np.zeros([filter_size, filter_size])

    for x in range(width):
        for y in range(height):
            value = (1 - abs(x/f-c))*(1-abs(y/f-c))
            bilinear[x,y] = value

    weights = np.zeros([filter_size, filter_size, out_filters, in_filters])

    for i in range(out_filters):
        for j in range(in_filters):

            weights[:,:,i,j] = bilinear
    init = tf.constant_initializer(value = weights,dtype=tf.float32)
    return tf.get_variable(name="filter", initializer=init,shape=weights.shape)

    
 
def Conv_transpose(name, x, filter_size, in_filters, out_filters, fraction=2, padding="SAME"):
    with tf.variable_scope(name):
        n = filter_size * filter_size * out_filters
        #kernel = tf.get_variable('filter', [filter_size, filter_size, out_filters, in_filters], tf.float32,
        #                         initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)))
        kernel = upsampling_bilinear(filter_size,out_filters,in_filters)
        size = tf.shape(x)
        output_shape = tf.stack([size[0], size[1] * fraction, size[2] * fraction, out_filters])

        x = tf.nn.conv2d_transpose(x, kernel, output_shape, [1, fraction, fraction, 1], padding)
        
        return x
 
 
def instance_norm(x, BN_epsilon=1e-3):
    mean, variance = tf.nn.moments(x, axes=[1, 2])
    x = (x - mean) / ((variance + BN_epsilon) ** 0.5)
    return x
