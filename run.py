from helpers import *
from CNN import CNN
from CNN_SW import CNN_SW
import keras



RESTORE = 0 # Set to 1 if the weights are to be loaded
WITH_SUBMISSION = 0 # Set to 1 if a Kaggle submission is to be created
WITH_SAVE = 0 # Set to 1 if the weights are to be savec after the training (ie RESTORE has to be 0)
NUM_TRAIN_IMAGES = 50

# Uncomment second line if you wish to use the CNN model (without sliding window)
model = CNN_SW()
#model = CNN()

if RESTORE:
    model.load() 
else:
    train_images = extract_images('training/images/',NUM_TRAIN_IMAGES)
    gt_images = extract_groundtruth('training/groundtruth/',NUM_TRAIN_IMAGES)
    model.train(train_images,gt_images)
    
    if WITH_SAVE:
        model.save()

if WITH_SUBMISSION:
    generate_submission(model)