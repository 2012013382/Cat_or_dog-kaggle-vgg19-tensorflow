from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.misc import imread, imresize
from os import  walk
from os.path import join
import tensorflow as tf
def read_images(path, classes, img_height = 224, img_width = 224, img_channels = 3):

    filenames = (walk(path)).next()[2]
    num_files = len(filenames)

    images = np.zeros((num_files, img_height, img_width, img_channels), dtype=np.float32)
    labels = np.zeros((num_files, ), dtype=np.int32)
    for i, filename in enumerate(filenames):
        img = imread(join(path, filename)) #read train data
        img = imresize(img, (img_height, img_width)) #resize image to 224x224
        img.astype(np.float32)
        images[i, :, :, :] = img
        labels[i] = classes.index(filename[0:3])    # Luckily both 'cat' and 'dog' have 3 characters
        # Convert from [0, 255] -> [-0.5, 0.5] floats.
        images[i, :, :, :] = images[i, :, :, :] * (1. / 255) - 0.5
        one_hot_labels = np.zeros((num_files, len(classes)))
        if i % 1000 == 0:
           print('Load the %d image of 25000.' % (i))
    for i in range(num_files):
        one_hot_labels[i, labels[i]] = 1
    return images, one_hot_labels

#Read image function for kaggle test data
def read_images_kaggle_result(path, img_height = 224, img_width = 224, img_channels = 3):

    filenames = (walk(path)).next()[2]
    num_files = len(filenames)

    images = np.zeros((num_files, img_height, img_width, img_channels), dtype=np.float32)
    for i, filename in enumerate(filenames):
        img = imread(join(path, filename)) #read train data
        img = imresize(img, (img_height, img_width)) #resize image to 224x224
        img.astype(np.float32)
        images[int(filename[:-4]) - 1, :, :, :] = img
        images[int(filename[:-4]) - 1, :, :, :] = images[int(filename[:-4]) - 1, :, :, :] * (1. / 255) - 0.5
        if i % 1000 == 0:
           print ('Load the %d image of 12500.'%(i))
    return images
