import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

conv_1_1_weights = weight_variable([3, 3, 3, 64])
conv_1_1_bias = bias_variable([64])

conv_1_2_weights = weight_variable([3, 3, 64, 64])
conv_1_2_bias = bias_variable([64])

conv_2_1_weights = weight_variable([3, 3, 64, 128])
conv_2_1_bias = bias_variable([128])

conv_2_2_weights = weight_variable([3, 3, 128, 128])
conv_2_2_bias = bias_variable([128])

conv_3_1_weights = weight_variable([3, 3, 128, 256])
conv_3_1_bias = bias_variable([256])

conv_3_2_weights = weight_variable([3, 3, 256, 256])
conv_3_2_bias = bias_variable([256])

conv_3_3_weights = weight_variable([3, 3, 256, 256])
conv_3_3_bias = bias_variable([256])

conv_4_1_weights = weight_variable([3, 3, 256, 512])
conv_4_1_bias = bias_variable([512])

conv_4_2_weights = weight_variable([3, 3, 512, 512])
conv_4_2_bias = bias_variable([512])

conv_4_3_weights = weight_variable([3, 3, 512, 512])
conv_4_3_bias = bias_variable([512])

conv_5_1_weights = weight_variable([3, 3, 512, 512])
conv_5_1_bias = bias_variable([512])

conv_5_2_weights = weight_variable([3, 3, 512, 512])
conv_5_2_bias = bias_variable([512])

conv_5_3_weights = weight_variable([3, 3, 512, 512])
conv_5_3_bias = bias_variable([512])

fc_6_weights = weight_variable([7, 7, 512, 4096])
fc_6_bias = bias_variable([4096])

fc_6_weights = weight_variable([1, 1, 4096, 4096])
fc_6_bias = bias_variable([4096])

deconv_fc_6_weights = weight_variable([7, 7, 512, 4096])
deconv_fc_6_bias = bias_variable([512])

deconv_5_3_weights = weight_variable([3, 3, 512, 512])
deconv_5_3_bias = bias_variable([512])

deconv_5_2_weights = weight_variable([3, 3, 512, 512])
deconv_5_2_bias = bias_variable([512])

deconv_5_1_weights = weight_variable([3, 3, 512, 512])
deconv_5_1_bias = bias_variable([512])

deconv_4_3_weights = weight_variable([3, 3, 512, 512])
deconv_4_3_bias = bias_variable([512])

deconv_4_2_weights = weight_variable([3, 3, 512, 512])
deconv_4_2_bias = bias_variable([512])

deconv_4_1_weights = weight_variable([3, 3, 256, 512])
deconv_4_1_bias = bias_variable([256])

deconv_3_3_weights = weight_variable([3, 3, 256, 256])
deconv_3_3_bias = bias_variable([256])

deconv_3_2_weights = weight_variable([3, 3, 256, 256])
deconv_3_2_bias = bias_variable([256])

deconv_3_1_weights = weight_variable([3, 3, 128, 256])
deconv_3_1_bias = bias_variable([128])

deconv_2_2_weights = weight_variable([3, 3, 128, 128])
deconv_2_2_bias = bias_variable([128])

deconv_2_1_weights = weight_variable([3, 3, 64, 128])
deconv_2_1_bias = bias_variable([64])

deconv_1_2_weights = weight_variable([3, 3, 64, 64])
deconv_1_2_bias = bias_variable([64])

deconv_1_1_weights = weight_variable([3, 3, 3, 64])
deconv_1_1_bias = bias_variable([32])

score_weights = weight_variable([1, 1, 2, 32])
score_bias = bias_variable([2])
