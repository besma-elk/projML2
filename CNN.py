from sklearn.model_selection import train_test_split
from helpers import *
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.constraints import maxnorm
from keras.optimizers import SGD

# This model consists of a basic CNN without any sliding window
# The data is balanced and bagging is done
class CNN:
    
    def __init__(self):
        self.patch_size = 16
        self.bagging_size = 3
        self.bagging_ratio = 0.5
        self.training_stride = 8
        self.num_epochs = 10
        self.batch_size = 16
        self.models = []
        self.build_models()
        

    def build_models(self):
                
        for i in range(self.bagging_size):
            model = Sequential()
            model.add(Conv2D(32, (5, 5), input_shape=(self.patch_size, self.patch_size,3), padding='same',
                             activation='relu', kernel_constraint=maxnorm(3)))
            model.add(MaxPooling2D(pool_size=(2, 2)))


            model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())
            model.add(Dropout(0.5))
            model.add(Dense(2, activation='sigmoid'))

            sgd = SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=False)
            model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
            
            self.models.append(model)
    
        print(model.summary())
    
    def train(self,data_images,gt_images):
        
        data_images = convert_to_hsv(data_images)
        
        data_patches = image_to_patches(data_images, self.patch_size, self.patch_size, self.training_stride)
        gt_patches = gt_to_patches(gt_images, self.patch_size, self.patch_size, self.training_stride)
        
        def balance_data():
            c0 = 0
            c1 = 0
            for i in range(len(gt_patches)):
                if gt_patches[i][0] == 1:
                    c0 = c0 + 1
                else:
                    c1 = c1 + 1

            min_c = min(c0, c1)
            idx0 = [i for i, j in enumerate(gt_patches) if j[0] == 1]
            idx1 = [i for i, j in enumerate(gt_patches) if j[1] == 1]
            new_indices = idx0[0:min_c] + idx1[0:min_c]
            data = data_patches[new_indices,:,:,:]
            labels = gt_patches[new_indices]

            return data, labels
               
        #data_patches , gt_patches = balance_data()
        
        X_train, X_test, Y_train, Y_test = train_test_split(data_patches, gt_patches, test_size=0.2, random_state=42)
        
        data_sets , label_sets = create_bagging_sets(X_train, Y_train, self.bagging_ratio, self.bagging_size)

        for i in range(self.bagging_size):
                print('Training model n°%d' % (i+1))
                self.models[i].fit(data_sets[i],label_sets[i],validation_data=(X_test, Y_test),
                    epochs=self.num_epochs, batch_size=self.batch_size)
    
    def save(self):
        for i in range(self.bagging_size):
            self.models[i].save_weights('cnn_model_' + str(i) + '.h5')
    
    def load(self):
        for i in range(self.bagging_size):
            self.models[i].load_weights('cnn_model_' + str(i) + '.h5')
            
    # If groundtruth images are present, this method prints out the obtained accuracy
    # Hard_voting is majority vote whereas soft_voting computes the mean of all soft predictions
    def predict(self, images, gt_images = [], soft_voting = False):
        
        images = convert_to_hsv(images)
        
        image_patches = image_to_patches(images, self.patch_size, self.patch_size, self.patch_size)
                
        preds = np.asarray([self.models[i].predict(image_patches) for i in range(self.bagging_size)])
        predictions = np.mean(preds,axis=0)
        
        hard_predictions = np.zeros(predictions.shape)
    
        indices = np.arange(len(predictions)).reshape((len(predictions),1))
        
        if soft_voting:
            values = np.argmax(predictions,1).reshape((len(predictions),1))
        else:
            values = np.rint(np.mean(np.argmax(preds,2),0)).astype(int).reshape((len(predictions),1))

        hard_predictions[indices,values] = 1
    
        if len(gt_images):
            labels = gt_to_patches(gt_images, self.patch_size, self.patch_size, self.patch_size)
            acc = numpy.count_nonzero(numpy.subtract(hard_predictions,labels))/(2*len(labels))
            print("Prediction accuracy: %.2f%%" % ((1-acc)*100))
        
        return predictions , hard_predictions    