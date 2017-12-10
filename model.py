def model(train=False):

    train_data_node = tf.placeholder( tf.float32, shape=(BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS)) # --> 400*400*3

    conv1 = conv_layer(train_data_node,conv1_weights,conv1_bias) # --> 400*400*32

    pool1, pool1_argmax = pool_layer(conv1) # --> 200*200*32

    conv2 = conv_layer(pool1, conv2_weights, conv2_bias) # --> 200*200*64

    pool2, pool2_argmax = pool_layer(conv2) # --> 100*100*64

    conv3 = conv_layer(pool2, conv3_weights, conv3_bias) # --> 100*100*128

    pool3, pool3_argmax = pool_layer(conv3) # --> 50*50*128

    conv4 = conv_layer(pool3, conv4_weights, conv4_bias) # --> 50*50*256 

    pool4, pool4_argmax = pool_layer(conv4) # --> 25*25*256

    fc1 = conv_layer(pool4, [, 7, 512, 4096], 4096, 'fc_6')
    fc2 = conv_layer(fc_6, [1, 1, 4096, 4096], 4096, 'fc_7')



def define_weights():

    conv1_weights = tf.Variable(tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                            stddev=0.1,
                            seed=SEED))    

    conv1_biases = tf.Variable(tf.zeros([32]))
    
    conv2_weights = tf.Variable(tf.truncated_normal([5, 5, 32, 64],  # 5x5 filter, depth 64.
                            stddev=0.1,
                            seed=SEED))    

    conv2_biases = tf.Variable(tf.zeros([64]))

    conv3_weights = tf.Variable(tf.truncated_normal([5, 5, 64, 128],  # 5x5 filter, depth 128.
                            stddev=0.1,
                            seed=SEED))    

    conv3_biases = tf.Variable(tf.zeros([128]))
    
    conv4_weights = tf.Variable(tf.truncated_normal([5, 5, 128, 256],  # 5x5 filter, depth 256.
                            stddev=0.1,
                            seed=SEED))    

    conv4_biases = tf.Variable(tf.zeros([256]))

    fc1_weights = tf.Variable(tf.truncated_normal([5, 5, 128, 256],  # 25x25 filter, depth .
                            stddev=0.1,
                            seed=SEED))   



def pool_layer(x):

    return tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def conv_layer(x,weights,bias,padding='SAME'):

    return f.nn.relu(tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding=padding) + bias)
