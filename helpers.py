import numpy as np
import matplotlib.colors as colors
import matplotlib.image as mpimg
from PIL import Image
import re
import os

# Helper functions


# Extracts images according to the filename
# Notice that the image IDs have to be of the form satImage_001 (for instance)
def extract_images(filename, num_images):
    
    imgs = []
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    num_images = len(imgs)    
    return np.asarray(imgs)

# Extracts groundtruth images according to the filename
# Notice that the image IDs have to be of the form satImage_001 (for instance)
def extract_groundtruth(filename, num_images):
    gt_imgs = []
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    return np.asarray(gt_imgs).astype(np.float32)

# Converts a list of images from RGB to HSV color representation
def convert_to_hsv(images):
    
    if len(images.shape) < 4:
        images = colors.rgb_to_hsv(images)
    else :
        for i in range(len(images)):
            images[i] = colors.rgb_to_hsv(images[i])
    
    return images

# Padds the list of images with the 'reflect' option
def pad_images(images,pad_val):
    
    if len(images.shape) < 4:
        images = np.expand_dims(images, axis=0)
        
    padded_images = []
    for i in range(len(images)):
        im = np.lib.pad(images[i], ((pad_val, pad_val), (pad_val, pad_val), (0, 0)), 'reflect')
        padded_images.append(im)
    
    return np.asarray(padded_images)

# Returns a list of patches from image im of height h, width w, and with a certain stride between patches
def img_crop(im, w, h, stride):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight-h+1,stride):
        for j in range(0,imgwidth-w+1,stride):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches

# Returns a list of patches comming sequentially from the images in the given list
def image_to_patches(images, w, h, stride):
    
    if len(images.shape) < 4:
        images = np.expand_dims(images, axis=0)
        
    num_images = images.shape[0]
    patches_aux = [img_crop(images[i], w, h, stride) for i in range(num_images)]
    patches = [patches_aux[i][j] for i in range(len(patches_aux)) for j in range(len(patches_aux[i]))]
    
    return np.asarray(patches)

# Returns a list of groundtruth values of the corresponding patches 
# comming sequentially from the groudtruth images in the given list
def gt_to_patches(gt_images, w, h, stride):
    
    gt_patches_aux = [img_crop(gt_images[i], w, h, stride) for i in range(len(gt_images))]
    gt_patches = np.asarray([gt_patches_aux[i][j] for i in range(len(gt_patches_aux)) for j in range(len(gt_patches_aux[i]))])
    
    labels = np.asarray([value_to_class(np.mean(gt_patches[i])) for i in range(len(gt_patches))])

    return labels.astype(np.float32)

# Returns the categorical 2-label array corresponding to a soft label
def value_to_class(v):
    foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
    if v > foreground_threshold:
        return [0, 1]
    else:
        return [1, 0]

# Creates n bagging sets of size ratio*length_of_data (random with replacement)
def create_bagging_sets(train_data,train_labels,ratio,n):

    np.random.seed()
    
    l = np.size(train_data,0)
    batch_size = int(np.floor(ratio*l))
    indices = np.random.randint(l,size=(n,batch_size))
    
    data_sets = [train_data[indices[i]] for i in range(n)]
    label_sets = [train_labels[indices[i]] for i in range(n)]
    
    
    return np.asarray(data_sets), np.asarray(label_sets)

# Creates the image from the labels
def label_to_img(imgwidth, imgheight, w, h, labels):
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            im[j:j+w, i:i+h] = labels[idx,1]
            idx = idx + 1
    return im

# Returns the hard-binary value of a patch 
def patch_to_label(patch):
    foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0

# Reads a single image and outputs the strings that should go into the submission file
def mask_to_submission_strings(image_filename):
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))

# Creates the submission from a given image mask
def masks_to_submission(submission_filename, *image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn))

# Generates the submission for test_images in 'test_set_images' given a model
def generate_submission(model):
    
    image_filenames = []

    for i in range(1, 51):
        img = mpimg.imread('test_set_images/test_' + str(i) + '.png')
        p, hard_p = model.predict(img)
        
        image_filename = 'test_set_images/pred_' + str(i) + '.png'
        predicted_img = label_to_img(img.shape[0], img.shape[1], 16, 16, hard_p)
        Image.fromarray(predicted_img*255).convert('RGB').save(image_filename)

        if (i%5) == 0:
            print('%d images predicted' % i)
        
        image_filenames.append(image_filename)
        
        
    masks_to_submission('submission.csv', *image_filenames)