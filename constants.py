# Constants

NUM_CHANNELS = 3 # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
TRAINING_SIZE = 20 # Number of images to train
VALIDATION_SIZE = 5  # Size of the validation set.
SEED = None  # Set to None for random seed.
NUM_EPOCHS = 5
RESTORE_MODEL = False # If True, restore existing model instead of training a new one
RECORDING_STEP = 1000
IMG_SIZE = 400

LEARNING_RATE = 1e-6

# Square size of scaled image
SCALED_IMAGE_SIZE = 224

# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!
# Used to compute patch label for submission
IMG_PATCH_SIZE = 16