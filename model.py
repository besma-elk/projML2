from constants import *
from parameters import *

import tensorflow as tf

# FUNCTIONS COPIED FROM CODE IN PAPER !!!! Only a few changes


def model(data):

    conv_1_1 = conv_layer(data, conv_1_1_weights, conv_1_1_bias)
    conv_1_2 = conv_layer(conv_1_1, conv_1_2_weights, conv_1_2_bias)

    pool_1, pool_1_argmax = pool_layer(conv_1_2)

    conv_2_1 = conv_layer(pool_1, conv_2_1_weights, conv_2_1_bias)
    conv_2_2 = conv_layer(conv_2_1, conv_2_2_weights, conv_2_2_bias)

    pool_2, pool_2_argmax = pool_layer(conv_2_2)

    conv_3_1 = conv_layer(pool_2, conv_3_1_weights, conv_3_1_bias)
    conv_3_2 = conv_layer(conv_3_1, conv_3_2_weights, conv_3_2_bias)
    conv_3_3 = conv_layer(conv_3_2, conv_3_3_weights, conv_3_3_bias)

    pool_3, pool_3_argmax = pool_layer(conv_3_3)

    conv_4_1 = conv_layer(pool_3, conv_4_1_weights, conv_4_1_bias)
    conv_4_2 = conv_layer(conv_4_1, conv_4_2_weights, conv_4_2_bias)
    conv_4_3 = conv_layer(conv_4_2, conv_4_3_weights, conv_4_3_bias)

    pool_4, pool_4_argmax = pool_layer(conv_4_3)

    conv_5_1 = conv_layer(pool_4, conv_5_1_weights, conv_5_1_bias)
    conv_5_2 = conv_layer(conv_5_1, conv_5_2_weights, conv_5_2_bias)
    conv_5_3 = conv_layer(conv_5_2, conv_5_3_weights, conv_5_3_bias)

    pool_5, pool_5_argmax = pool_layer(conv_5_3)

    fc_6 = conv_layer(pool_5, fc_6_weights, fc_6_bias)
    fc_7 = conv_layer(fc_6, fc_7_weights, fc_7_bias)

    deconv_fc_6 = deconv_layer(fc_7, deconv_fc_6_weights, deconv_fc_6_bias)

    unpool_5 = unpool_layer2x2(deconv_fc_6, pool_5_argmax, tf.shape(conv_5_3))

    deconv_5_3 = deconv_layer(unpool_5, deconv_5_3_weights, deconv_5_3_bias)
    deconv_5_2 = deconv_layer(deconv_5_3, deconv_5_2_weights, deconv_5_2_bias)
    deconv_5_1 = deconv_layer(deconv_5_2, deconv_5_1_weights, deconv_5_1_bias)

    unpool_4 = unpool_layer2x2(deconv_5_1, pool_4_argmax, tf.shape(conv_4_3))

    deconv_4_3 = deconv_layer(unpool_4, deconv_4_3_weights, deconv_4_3_bias)
    deconv_4_2 = deconv_layer(deconv_4_3, deconv_4_2_weights, deconv_4_2_bias)
    deconv_4_1 = deconv_layer(deconv_4_2, deconv_4_1_weights, deconv_4_1_bias)

    unpool_3 = unpool_layer2x2(deconv_4_1, pool_3_argmax, tf.shape(conv_3_3))

    deconv_3_3 = deconv_layer(unpool_3, deconv_3_3_weights, deconv_3_3_bias)
    deconv_3_2 = deconv_layer(deconv_3_3, deconv_3_2_weights, deconv_3_2_bias)
    deconv_3_1 = deconv_layer(deconv_3_2, deconv_3_1_weights, deconv_3_1_bias)

    unpool_2 = unpool_layer2x2(deconv_3_1, pool_2_argmax, tf.shape(conv_2_2))

    deconv_2_2 = deconv_layer(unpool_2, deconv_2_2_weights, deconv_2_2_bias)
    deconv_2_1 = deconv_layer(deconv_2_2, deconv_2_1_weights, deconv_2_1_bias)

    unpool_1 = unpool_layer2x2(deconv_2_1, pool_1_argmax, tf.shape(conv_1_2))

    deconv_1_2 = deconv_layer(unpool_1, deconv_1_2_weights, deconv_1_2_bias)
    deconv_1_1 = deconv_layer(deconv_1_2, deconv_1_1_weights, deconv_1_1_bias)

    score = deconv_layer(deconv_1_1, score_weights, score_bias)

    return score


def conv_layer(x, weights, bias, padding='SAME'):
    return tf.nn.relu(tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding=padding) + bias)

def pool_layer(x):
    return tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def deconv_layer(x, weights, bias, padding='SAME'):

    x_shape = tf.shape(x)
    out_shape = tf.stack([x_shape[0], x_shape[1], x_shape[2], W_shape[2]])

    return tf.nn.conv2d_transpose(x, weights, out_shape, [1, 1, 1, 1], padding=padding) + bias

def unravel_argmax(argmax, shape):
    output_list = []
    output_list.append(argmax // (shape[2] * shape[3]))
    output_list.append(argmax % (shape[2] * shape[3]) // shape[3])
    return tf.stack(output_list)

def unpool_layer2x2(x, raveled_argmax, out_shape):
    argmax = unravel_argmax(raveled_argmax, tf.to_int64(out_shape))
    output = tf.zeros([out_shape[1], out_shape[2], out_shape[3]])

    height = tf.shape(output)[0]
    width = tf.shape(output)[1]
    channels = tf.shape(output)[2]

    t1 = tf.to_int64(tf.range(channels))
    t1 = tf.tile(t1, [((width + 1) // 2) * ((height + 1) // 2)])
    t1 = tf.reshape(t1, [-1, channels])
    t1 = tf.transpose(t1, perm=[1, 0])
    t1 = tf.reshape(t1, [channels, (height + 1) // 2, (width + 1) // 2, 1])

    t2 = tf.squeeze(argmax)
    t2 = tf.stack((t2[0], t2[1]), axis=0)
    t2 = tf.transpose(t2, perm=[3, 1, 2, 0])

    t = tf.concat([t2, t1], 3)
    indices = tf.reshape(t, [((height + 1) // 2) * ((width + 1) // 2) * channels, 3])

    x1 = tf.squeeze(x)
    x1 = tf.reshape(x1, [-1, channels])
    x1 = tf.transpose(x1, perm=[1, 0])
    values = tf.reshape(x1, [-1])

    delta = tf.SparseTensor(indices, values, tf.to_int64(tf.shape(output)))
    return tf.expand_dims(tf.sparse_tensor_to_dense(tf.sparse_reorder(delta)), 0)

def unpool_layer2x2_batch(x, argmax):
    '''
    Args:
        x: 4D tensor of shape [batch_size x height x width x channels]
        argmax: A Tensor of type Targmax. 4-D. The flattened indices of the max
        values chosen for each output.
    Return:
        4D output tensor of shape [batch_size x 2*height x 2*width x channels]
    '''
    x_shape = tf.shape(x)
    out_shape = [x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]]

    batch_size = out_shape[0]
    height = out_shape[1]
    width = out_shape[2]
    channels = out_shape[3]

    argmax_shape = tf.to_int64([batch_size, height, width, channels])
    argmax = unravel_argmax(argmax, argmax_shape)

    t1 = tf.to_int64(tf.range(channels))
    t1 = tf.tile(t1, [batch_size*(width//2)*(height//2)])
    t1 = tf.reshape(t1, [-1, channels])
    t1 = tf.transpose(t1, perm=[1, 0])
    t1 = tf.reshape(t1, [channels, batch_size, height//2, width//2, 1])
    t1 = tf.transpose(t1, perm=[1, 0, 2, 3, 4])

    t2 = tf.to_int64(tf.range(batch_size))
    t2 = tf.tile(t2, [channels*(width//2)*(height//2)])
    t2 = tf.reshape(t2, [-1, batch_size])
    t2 = tf.transpose(t2, perm=[1, 0])
    t2 = tf.reshape(t2, [batch_size, channels, height//2, width//2, 1])

    t3 = tf.transpose(argmax, perm=[1, 4, 2, 3, 0])

    t = tf.concat([t2, t3, t1], 4)
    indices = tf.reshape(t, [(height//2)*(width//2)*channels*batch_size, 4])

    x1 = tf.transpose(x, perm=[0, 3, 1, 2])
    values = tf.reshape(x1, [-1])

    delta = tf.SparseTensor(indices, values, tf.to_int64(out_shape))

    return tf.sparse_tensor_to_dense(tf.sparse_reorder(delta))