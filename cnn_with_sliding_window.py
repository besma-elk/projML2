# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Conv2D,Activation, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import Nadam

import numpy as np

from sklearn.model_selection import train_test_split



class Cnn:
    
    def __init__(self):   
        self.window_size = 64
        self.patch_size = 16
        self.build_model()
        
    def build_model(self):
        window_size = self.window_size
        patch_size = self.patch_size
        
        self.model = Sequential()
   
        self.model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(window_size,window_size,3)))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
    
 
        self.model.add(Conv2D(128, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(128, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        
        self.model.add(Conv2D(256, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(256, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
   
   
        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(2))
        self.model.add(Activation('softmax'))
      
    
    def train(self, data, labels):
   
        window_size = self.window_size
        patch_size = self.patch_size
        batch_size = 128
        num_classes = 2
        #data, x_test, labels, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

        # To use the window sliding method we need to use padding to work on the corners
        
        pad_val = int((window_size-patch_size)/2)  
        
        X = np.zeros( (data.shape[0],data.shape[1] + 2*pad_val, data.shape[2] + 2*pad_val, data.shape[3]) )
        Y = np.zeros( (labels.shape[0], labels.shape[1] + 2*pad_val, labels.shape[2] +2*pad_val) )
        
        for i in range(data.shape[0]):
            X[i] = np.lib.pad(data[i], ((pad_val, pad_val), (pad_val, pad_val), (0, 0)), 'symmetric')
            Y[i] = np.lib.pad(labels[i], ((pad_val, pad_val), (pad_val, pad_val)), 'symmetric')
            

        self.model.compile(loss='categorical_crossentropy',
                      optimizer=Nadam(lr=0.001),
                      metrics=['accuracy'])

        np.random.seed(1)
        
        #data augmentation
        def generator(data, labels):
            ### TODO

        
      

        self.model.fit_generator(generator(data,labels),validation_data=generator(data,labels),validation_steps=0.1,
                          samples_per_epoch=10,
                            nb_epoch=20,
                            verbose=2,
                                   )


