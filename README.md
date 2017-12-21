# EPFL Machine Learning Project 2 - Road Segmenation on Aerial Images
##### Besma Elketroussi, Ines Bahej, Romain Leteurtre

This project is presented as part of EPFL's Machine Learning Course in 2017. It's aim is to implement a classifier to segment roads on satellite images from Google Maps.
Three different classifiers were implemented: an SVM classifier, a basic Convolutional Neural Network with bagging, and a basic CNN with a sliding window.

### Libraries
The following libraries have to be used in order to run the project:

- Tensorflow 1.4.0 https://www.tensorflow.org/
- Keras 2.1.2 https://keras.io/#installation

Tensorflow is a popular machine learning toolkit developped by Google.
Keras is a high-level neural networks API capable of running on top of Tensorflow which facilitates CNN implementations

Also libraries coming with a basic python instalaltion were used:

- Numpy 1.12.1
- Sklearn 0.18.1
- PIL 4.1.1
- Matplotlib 2.0.2

### How to run

Keras uses the computer's CPU, and isn't configured to use the GPU.

To run the CNN models simply run the script `run.py`. In the script, simply uncomment the model you wish to use and comment out the other one. 

Three flags in the script have a binary value:
- set `RESTORE` to `0` if you wish to train the model or `1` to use the savec weights
It is advised to use the saved weights since the training phase for both CNNs last between 3-4 hours.

- set `WITH_SAVE` to `1` if you wish to save the weights if the model was trained (i.e. if `RESTORE` is set to `1`)
	if set to `0` the weights will be discarded after the training

- set `WITH_SUBMISSION` to `1` if you wish to generate a Kaggle submission for the selected CNN

- set `NUM_TRAIN_IMAGES` to the number of images you wish to de the training on. The amount of images can go from 0 to 100

Finally, use the jupyter notebook script `svm.ipynb` to use the Support Vector Machine classifier

The `training/images/`, `training/groundtruth/` folders containing the training data have to be in the same folder as the python script and methods if the `RESTORE` flag is set to `0`
Moreover, the `test_set_images` folder also has to be in the same folder if the `WITH_SUBMISSION` flag is set to `1`


### Description of the files

- `run.py` is the main script that should be used every time. You can modify the flags in the script and the model used for testing purposes.
- `CNN.py` is the python class implementing a basic CNN with bagging
- `CNN_SW.py` is the python class implementing a basic CNN with a sliding window
- `helpers.py`regroups all the utility functions used in the training and testing phases
- `svm.ipynb`is the jupyter notebook scipt that uses Support Vector Machine as a classifier


