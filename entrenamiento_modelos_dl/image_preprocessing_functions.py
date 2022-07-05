# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 12:26:46 2022

@author: mario
"""


import matplotlib.pyplot as plt
import matplotlib.patches as pat
from matplotlib.lines import Line2D
import numpy as np
import os
import random
import pandas as pd
import math
import pickle
from operator import itemgetter
import datetime

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras.models import load_model



from macros import *
from distance_functions import *




"""
###############################################################################
READ IMAGES IN DATASETS

The following functions are used to read an image from a path within TensorFlow
datasets. They are going to be mapped over Tf datasets containing image
paths to get the image datasets.
"""

"""
Load the image file specified with filename to a Tf tensor.
Firstly it is loaded into a Tf string tensor without any parsing,
then it is processed in order to convert it into a float image tensor
with its proper shape.
"""
def get_image(filename):
    image_string = tf.io.read_file(filename) # Load into string tensor

    image = tf.image.decode_jpeg(image_string, channels=3) # Convert to int tensor
    image = tf.image.convert_image_dtype(image, tf.float32) # Convert to float tensor
    
    image = tf.image.resize(image, input_shape) # Adapt shape

    return image


"""
Get each one of the images of a single triplet given their 
filenames.
"""
def get_triplet(anchor, positive, negative):
    anchor_im   = get_image(anchor)
    positive_im = get_image(positive)
    negative_im = get_image(negative)

    return anchor_im, positive_im, negative_im

"""
Get each one of the images of a single pair given their 
filenames.
"""
def get_pair(skull_path, face_name):
    skull_im = get_image(skull_path)

    return skull_im, face_name


"""
Get each one of the images of a single triplet given their 
filenames.
"""
def get_triplet_with_label(triplet_im, triplet_labels):
    (anchor, positive, negative) = triplet_im

    anchor, positive, negative = get_triplet(anchor, positive, negative)

    return (anchor, positive, negative), triplet_labels

"""
Get both images in a pair given their filenames.
"""
def get_pair_im(images, label):
    im1, im2 = images
    
    im1 = get_image(im1)
    im2 = get_image(im2)

    return (im1, im2), label



"""
###############################################################################
FACE IMAGE DATA AUGMENTATION

Random horizontal flips, random contrast adjustment and random brightness
adjustments will be made to face images in the training dataset.

https://arxiv.org/abs/1904.11685
"""

def data_augmentation_faces(image):
    image = tf.image.random_flip_left_right (image, seed=SEED)
    image = tf.image.random_contrast        (image, 1.0-ctrst_factor, 1.0+ctrst_factor, seed=SEED)
    image = tf.image.random_brightness      (image, brt_factor, seed=SEED)
    # image = tf.image.random_hue             (image, hue_factor, seed=SEED)  # In theory not adecuate because changes skin hue
    image = tf.image.random_saturation      (image, sat_upper_factor, sat_lower_factor, seed=SEED)

    image = tf.clip_by_value(image, 0, 1) # Clip values lower than 0 and higher than 1, just in case

    return image

def data_aug_faces_online(triplet_im, triplet_labels):
    (anchor, positive, negative) = triplet_im

    anchor   = data_augmentation_faces(anchor)
    positive = data_augmentation_faces(positive)
    negative = data_augmentation_faces(negative)

    return (anchor, positive, negative), triplet_labels

def data_aug_pairs(images, label):
    im1, im2 = images
    im1 = data_augmentation_faces(im1)

    return (im1, im2), label



"""
###############################################################################
IMAGE RESIZING TO FIT FACENET INPUT SHAPE
"""
def resize_im(image, new_size):
    return tf.image.resize(image, new_size)

def resize_im_train_online(triplet_im, triplet_labels):
    (anchor, positive, negative) = triplet_im

    anchor   = resize_im(anchor, faceNet_shape)
    positive = resize_im(positive, faceNet_shape)
    negative = resize_im(negative, faceNet_shape)

    return (anchor, positive, negative), triplet_labels


def resize_pair(images, label):
    im1, im2 = images
    
    im1 = resize_im(im1, faceNet_shape)
    im2 = resize_im(im2, faceNet_shape)

    return (im1, im2), label