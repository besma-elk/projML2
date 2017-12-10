# DON'T KNOW WHICH IMPORTS ARE USEFUL

import gzip
import os
import sys
import urllib
import matplotlib.image as mpimg
from PIL import Image

import code
from constants import *
from parameters import *
from model import model

import tensorflow.python.platform

import numpy
import tensorflow as tf


def main(argv=None):

    data_dir = 'training/'
    train_data_filename = data_dir + 'images/'
    train_labels_filename = data_dir + 'groundtruth/' 

    # Extract it into numpy arrays, images of size 224*224 ###### TODO #######
    train_data = None
    train_labels = None

########################################################################

    # Number of data points per class 
    c0 = 0
    c1 = 0
    for i in range(len(train_labels)):
        if train_labels[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print ('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

    # Balance training data
    print ('Balancing training data...')
    min_c = min(c0, c1)
    idx0 = [i for i, j in enumerate(train_labels) if j[0] == 1]
    idx1 = [i for i, j in enumerate(train_labels) if j[1] == 1]
    new_indices = idx0[0:min_c] + idx1[0:min_c]
    print (len(new_indices))
    print (train_data.shape)
    train_data = train_data[new_indices,:,:,:]
    train_labels = train_labels[new_indices]


    train_size = train_labels.shape[0]
    print('train_size is '+ str(train_size))

    c0 = 0
    c1 = 0
    for i in range(len(train_labels)):
        if train_labels[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print ('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

########################################################################

    train_data_node = tf.placeholder(
        tf.float32,
        shape=(1, SCALED_IMG_SIZE, SCALED_IMG_SIZE, NUM_CHANNELS))
    train_labels_node = tf.placeholder(tf.float32,
                                       shape=(1,SCALED_IMG_SIZE,SCALED_IMG_SIZE, NUM_LABELS))

    
    logits = tf.reshape(model(train_data_node), (-1, 2))

    # MAKE SURE SIZES CORRESPOND (NOT TESTED AT ALL)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=train_labels_node, logits=logits))



    # Can change to MomentumOptimizer
    optimzer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)


    # Predictions  ????? TO DO 

    # In paper
            #prediction = tf.argmax(tf.reshape(tf.nn.softmax(logits), tf.shape(score_1)), dimension=3)
            #self.accuracy = tf.reduce_sum(tf.pow(self.prediction - expected, 2))

    # In main_example
            #train_prediction = tf.nn.softmax(logits)


    # .... 




