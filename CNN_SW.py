from helpers import *

import numpy as np
from keras.models import *
from keras.layers import Input, merge, Conv2D, Conv2DTranspose,MaxPooling2D,Activation, UpSampling2D, Dropout, Cropping2D, concatenate, Dense, Flatten
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.constraints import maxnorm
from keras.utils import to_categorical

# This model consists of a basic CNN with a Sliding Window to generate random patches for the training period.
# The window_size-patch_size should be even
# No bagging is done

class CNN_SW:
    
    def __init__(self):   
        self.window_size = 64
        self.patch_size = 16
        self.training_stride = 8
        self.num_epochs = 20
        self.batch_size = 64
        self.pad_val = int((self.window_size-self.patch_size)/2)  
        self.build_model()
        
    def build_model(self):
        
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), input_shape=(self.window_size, self.window_size, 3), padding='same', activation='relu', 
            kernel_constraint=maxnorm(3)))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))


        self.model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(2, activation='sigmoid'))
        
        print(self.model.summary())
        
    def train(self, data, labels):
        
        num_images = data.shape[0]
        image_width = data.shape[1]
        image_height = data.shape[2]
        
        samples_per_epoch = num_images*image_height*image_width//256  
        
        padded_data = np.zeros( (num_images, image_width + 2*self.pad_val, image_height + 2*self.pad_val, 3) )
        
        padding_dims = (self.pad_val, self.pad_val)
        for i in range(num_images):
            padded_data[i] = np.lib.pad(data[i], (padding_dims, padding_dims, (0, 0)), 'reflect')
            

        self.model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=0.001),
                      metrics=['accuracy'])
        

        np.random.seed(1)
            
        def generator():
            
            while True:
                batch_features = np.zeros((self.batch_size, self.window_size, self.window_size, 3))
                batch_labels = np.zeros((self.batch_size,2))
                
                for i in range(self.batch_size):
                    idx = np.random.choice(num_images)
                    
                     # Compute window center and fetch the labels and pixel-patch values
                    wnd_center = [np.random.randint(int(self.window_size/2), image_width-int(self.window_size/2)),
                                  np.random.randint(int(self.window_size/2), image_height-int(self.window_size/2))]
                    
                    batch_features[i] = data[idx,wnd_center[0] - int(self.window_size/2): 
                                             wnd_center[0] + int(self.window_size/2),
                                             wnd_center[1] - int(self.window_size/2): 
                                             wnd_center[1] + int(self.window_size/2)]
                    
                    gt_patch = labels[idx,wnd_center[0] - int(self.patch_size/2): 
                                      wnd_center[0] + int(self.patch_size/2),
                                      wnd_center[1] - int(self.patch_size/2): 
                                      wnd_center[1] + int(self.patch_size/2)]
                    
                    label = np.asarray(value_to_class(np.mean(gt_patch)))
                    batch_labels[i] =  label.astype(np.float64)
                    
                    
                yield (batch_features, batch_labels)

        self.model.fit_generator(generator(),
                            steps_per_epoch=samples_per_epoch//self.batch_size,
                            nb_epoch=self.num_epochs,
                            verbose=1)
    def save(self):
        self.model.save_weights('cnn_sl_model_.h5')
    
    def load(self):
        self.model.load_weights('cnn_sl_model_.h5')
        
    # If groundtruth images are present, this method prints out the obtained accuracy
    def predict(self, images, gt_images = []):
        
        images = convert_to_hsv(images)
        
        images = pad_images(images,self.pad_val)
            
        image_patches = image_to_patches(images, self.window_size, self.window_size, self.patch_size)
        
        predictions = self.model.predict(image_patches)
        
        hard_predictions = np.rint(predictions).astype(int)
        
        if len(gt_images):
            labels = gt_to_patches(gt_images, self.patch_size, self.patch_size, self.patch_size)
            acc = numpy.count_nonzero(numpy.subtract(hard_predictions,labels))/(2*len(labels))
            print("Prediction accuracy: %.2f%%" % ((1-acc)*100))
        print(predictions,hard_predictions)
        return predictions , hard_predictions
