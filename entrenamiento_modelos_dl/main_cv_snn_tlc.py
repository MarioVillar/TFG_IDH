# -*- coding: utf-8 -*-
"""

Triplet Loss SNN Cross Validation


Script for Cross Validating different models of SNN with Triplet Loss.
Evaluation (for each fold) is performed over a positive individuals test set
and over a negative individuals test set.
Individuals are separated for training and test.
SNN is based in FaceNet.
CMC curve and distance histogram are implemented.
Decision threshold detection is implemented and threshold is applied to
predictions in test dataset.
"""


###############################################################################
###############################################################################
###############################################################################
# SET UP


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
from threshold_detection_functions import *
from utilities import *
from test_functions import *
from snn_tl_offline import SNNTLOffline
from image_preprocessing_functions import *




###############################################################################
###############################################################################
###############################################################################
# Prepare Data


"""
###############################################################################
LOAD DATA

Dataset is stored in ./dataset/

###############################################################################
IMPORT DATA FROM DIRECTORIES

A Tensorflow pipeline is going to be used in order to import data from
directories. In this sense, a list with the anchors, positive and negative
examples filenames is going to be created to then map on them the funtion
which reads images.

Just as remainder, anchor images are face images, positive images are skull
images corresponding to the same person as the anchor image and negative images
are skull images corresponding to a different person in relation to the anchor image.

Skull and face images in dataset are `224x224` shaped,
whereas FaceNet has input shape of `160x160`.

###############################################################################
GET INFORMATION OF DATASET

The information about the face images related to each skull is in the second worksheet
"""

xls = pd.ExcelFile(info_path) # Excel file with the info
df_info = pd.read_excel(xls, 1) # Only read the second worksheet, which contains skull-face image relationships

df_info.rename(columns = {'Photo Key':'K'}, inplace = True) # Rename Photo Key column for easier access

df_info.drop(['Num Imagenes'], inplace=True, axis=1) # Second and Third column are not going to be used

# The rows corresponding to the invalid models are dropped.
df_info.drop(df_info.index[df_info['K'] == 'Mal modelo'], inplace=True)


"""
Get a dictionary with the image name and image path of all the
face images available. This face images are in correspondence
with some of the skull images.
"""

# Create empty dictionary
face_im_dict = {} # The Keys are going to be the image name and the values the image path

# For each file in image directory
for filename in os.listdir(face_im_path):
    # Append to dictionary
    face_im_dict[ int(filename.split(".")[0].split("_")[0]) ] = face_im_path + "/" + filename


num_faces = len(face_im_dict)


"""
###############################################################################
DIVIDE INTO POSITIVE AND NEGATIVE INDIVIDUALS

Positive individuals are those for which skull and face images are both
available, whereas negative individuals are those for which skull image
is available but face image is not.
"""

pos_individuals = df_info.loc[df_info['K'] != 'No']['Individuo'].to_numpy()

neg_individuals = df_info.loc[df_info['K'] == 'No']['Individuo'].to_numpy()

num_individuals = len(df_info)
num_pos_ind     = len(pos_individuals)
num_neg_ind     = len(neg_individuals)


"""
###############################################################################
SPLIT INDIVIDUALS IN FOLDS

In order to randomly split individuals (positive and negative) in equal folds,
the arrays `pos_individuals` and `neg_individuals` are randomly shuffled and
divided into same-length parts.
"""

split_pct = 1.0 / n_folds

# Random shuffle of positive and negative individuals
np.random.shuffle(pos_individuals)
np.random.shuffle(neg_individuals)

# Partition pos_individuals array into n_folds parts
pos_ind_splits = np.array_split(pos_individuals, n_folds)
# Partition neg_individuals array into n_folds parts
neg_ind_splits = np.array_split(neg_individuals, n_folds)


"""
############################
SPLIT POSITIVE INDIVIDUALS INTO TRAINING AND VALIDATION

When evaluating in fold i, all individuals of that fold serve as test, 
while training and validation individuals of the other fols are used
as training set and validation set.
"""

pos_val_ind_splits   = []
pos_train_ind_splits = []

for i in range(0, n_folds):
    # 10% of positive train individuals are chosen randomly for validation
    pos_val_ind_fold_i = pos_ind_splits[i][random.sample(range(0, pos_ind_splits[i].size-1), # Both bounds are included in random sample
                                                         math.ceil(pos_ind_splits[i].size*val_split))]

    # The remaining positive train individuals are utilisied for training
    pos_train_ind_fold_i = pos_ind_splits[i][~np.in1d(pos_ind_splits[i], pos_val_ind_fold_i)]
    
    pos_val_ind_splits.append(pos_val_ind_fold_i)
    pos_train_ind_splits.append(pos_train_ind_fold_i)



"""
The splits of individuals for each fold are saved in a dictionary
(`fold_individuals`) where the keys are the fold indexes (0..n_folds)
and the values are tuples of splits of positive training, positive validation
and negative individuals for each fold.
"""

# Create fold invdividuals dictionary
fold_individuals = {}

for i in range(0, n_folds):
    fold_individuals[i] = (pos_train_ind_splits[i], pos_val_ind_splits[i], neg_ind_splits[i])

"""
As each fold will be sometimes used for training and sometimes used for 
test, training and test datasets will be created for each one of the folds.
"""

"""
###############################################################################
TRAINING DATASETS

Each fold has got one training dataset associated, containing triplets 
generated with face images and skull images of the positive (test and 
validation) and negative individuals associated to that fold.

Training triplets are formed by an anchor image (a face image), a 
positive example (a skull image of the same person) and a negative 
example (a face image of other person). Only the training individuals 
are used for generating these triplets.

Train dataset for fold i is composed by training triplets formed with
all the individuals (test and validation) of that fold. This dataset
is used for training when evaluating in the other folds.
"""

# Number of available skull images for positive individuals
num_pos_images = len(os.listdir(skull_im_path + "/" + pos_individuals[0]))

# Number of skull images to be used for negative individuals in order to have the
# the same total number of skull images of positive and negative individuals and
# and generate the training triplets
num_neg_images = math.ceil(num_pos_images * len(pos_individuals) /
                           len(neg_individuals))

"""
The training datasets (for each one of the folds) are generated 
using a TensorFlow pipeline:

1.   Generate a list with Anchor image paths, a list with Positive image 
        pahts and a list with Negative image paths.
2.   Randomly shuffle the lists. The anchor list and the positive list 
        are shuffle in the same exact order.
3.   Create one dataset from slices from each one of the previous lists. 
        These three datasets contain the paths of the respective
        images.
4.   Create the training dataset comprising in triplets the paths of the
        anchor, positive and negative images.
5.   Map over the training dataset the function to read and return image. 
        This function transforms the dataset from one containing 
        paths to one containing images.

For each fold, a training dataset is generated and then saved to a 
dictionary (`fold_train_datasets`) where the keys are the folds and 
the values are the training datasets.
"""

fold_train_datasets = {}

# For each Fold
for i in range(0, n_folds):
    ##############
    # Path of Anchor images using test individuals
    # Get the face image path from the face_im_dict of the individual skull_ind:
    #   For all the skull_ind that are positive train individuals
    #       For each skull image of skull_ind
    anchor_path_list = [ face_im_dict[df_info.loc[df_info['Individuo'] == skull_ind].iloc[0,1]] # Image path gotten from dictionary with the image name (obtained from info df)
                              for skull_ind in fold_individuals[i][0]                             # For each positive train individual 
                                  for skull_im in os.listdir(skull_im_path + "/" + skull_ind)]    # For each skull image of that individual
    ##############
    # Path of Positive images using test individuals
    # Get the path of a skull image from skull_ind:
    #   For all the skull_ind that are positive train individuals
    #       For each skull image of skull_ind
    pos_path_list = [ skull_im_path + "/" + skull_ind + "/" + skull_im                   # Skull image path
                          for skull_ind in fold_individuals[i][0]                          # For each positive train individual 
                              for skull_im in os.listdir(skull_im_path + "/" + skull_ind)] # For each skull image of that individual
    
    
    ##############
    # Get total anchor and positive images
    anchor_path_list = anchor_path_list_1 + anchor_path_list_2
    pos_path_list = pos_path_list_1 + pos_path_list_2
    
    ##############
    # Path of Negative images
    # Get the path of a skull image from skull_ind:
    #   For all the skull_ind that are negative train individuals
    #       For each skull image of skull_ind
    #           If index is lower than the number of images needed
    neg_path_list = [ skull_im_path + "/" + skull_ind + "/" + skull_im                  # Skull image path
                        for skull_ind in fold_individuals[i][2]                         # For each individual in the skull dataset
                            for skull_im in os.listdir(skull_im_path + "/" + skull_ind) # For each skull image of that individual
                                if int(skull_im.split("_")[3].split(".")[0]) < num_neg_images ]
    ##############
    # Shuffle randomly anchor and positive images (together) and negative images
    # Zip together anchor and positive images
    anchor_positive_zip = list( zip(anchor_path_list, pos_path_list) )

    # Randomly shuffle the zip altogether
    random.shuffle(anchor_positive_zip)

    # Unzip
    anchor_path_list, pos_path_list = zip(*anchor_positive_zip)

    # Convert to list as the zip output comes as tuples
    anchor_path_list, pos_path_list = list(anchor_path_list), list(pos_path_list)

    # Randomly shuffle negative images
    random.shuffle(neg_path_list)

    ##############
    # Create a Tf dataset from each of the previous path lists
    anchor_dataset   = tf.data.Dataset.from_tensor_slices(anchor_path_list) # Anchors path dataset
    positive_dataset = tf.data.Dataset.from_tensor_slices(pos_path_list)    # Positives path dataset
    negative_dataset = tf.data.Dataset.from_tensor_slices(neg_path_list)    # Negative path dataset

    ##############
    # Create Tf dataset comprising anchor-positive-negative image paths (with them grouped in triplets)
    fold_i_triplets_dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset)) # Triplets path dataset
    fold_i_triplets_dataset = fold_i_triplets_dataset.shuffle(buffer_size=1024, seed=SEED)

    ##############
    # Transform dataset from one with paths to the real one with images
    
    # Map the funtions to read the images over the dataset to get the image dataset (previously the dataset contained paths)
    fold_i_train_dataset = fold_i_triplets_dataset.map(get_triplet)

    # Batch the dataset
    fold_i_train_dataset = fold_i_train_dataset.batch(train_batch_size)

    # Set prefecth buffer
    fold_i_train_dataset = fold_i_train_dataset.prefetch(8) # Setting prefetch buffer size to 8

    ##############
    # Save the dataset to Folds dataset dictionary
    fold_train_datasets[i] = fold_i_train_dataset



"""
###############################################################################
VALIDATION DATASET

Contains one skull image of every positive validation individual, 
together with its label. The label of a test skull image is the 
name of its corresponding face image.

The positive validation dataset is composed of pairs of skull 
test images and their labels. The label of a skull image is the 
name of the corresponding face image.
"""

fold_val_datasets = {}

# For each Fold
for i in range(0, n_folds):
    # Get the paths of the skull positive validation images. 
    # For each positive validation individual, one of its skull 
    # images is randomly taken. The name of the face image of each 
    # individual (of the face test images) is also saved as the label.
    
    val_pos_skull_path_list  = [] # Path of each skull test image
    val_pos_skull_label_list = [] # Name of each face test image
    
    for skull_ind in fold_individuals[i][1]: # For each individual in the validation skull dataset
        for image_index in range(0, 100):
            # Append skull image path to list
            val_pos_skull_path_list.append(skull_im_path + "/" + skull_ind + "/" + os.listdir(skull_im_path + "/" + skull_ind)[image_index])
    
            # Append corresponding face image name to list
            val_pos_skull_label_list.append(df_info.loc[df_info['Individuo'] == skull_ind].iloc[0,1])
    
    
    
    val_pos_skull_path_dataset   = tf.data.Dataset.from_tensor_slices(val_pos_skull_path_list)   # Skull validation images path dataset
    val_pos_skull_label_dataset  = tf.data.Dataset.from_tensor_slices(val_pos_skull_label_list)  # Face validation images names dataset
    
    fold_i_pos_val_pairs_dataset = tf.data.Dataset.zip((val_pos_skull_path_dataset, val_pos_skull_label_dataset)) # Test pairs path dataset
    pos_val_pairs_dataset = fold_i_pos_val_pairs_dataset.shuffle(buffer_size=1024, seed=SEED)
    
    
    fold_i_pos_val_dataset = fold_i_pos_val_pairs_dataset.map(get_pair)
    
    size_pos_val = int(fold_i_pos_val_dataset.__len__().numpy())
    
    test_batch_size = size_pos_val
    fold_i_pos_val_dataset = fold_i_pos_val_dataset.batch(test_batch_size)
    fold_i_pos_val_dataset = fold_i_pos_val_dataset.prefetch(8) # Setting prefetch buffer size to 8
    
    ##############
    # Save the dataset to Folds dataset dictionary
    fold_val_datasets[i] = fold_i_pos_val_dataset



"""
###############################################################################
TEST DATASETS

The model is evaluated over the skull images of the test individuals.

In order to do this, three datasets are defined:


*   Positive Test dataset (**one for each fold**): contains one skull image 
        of every positive test individual, together with its label. 
        The label of a test skull image is the name of its 
        corresponding face image.
*   Negative Test dataset (**one for each fold**): contains one skull image 
        of every negative test individual.
*   Face dataset (**same for all folds**): contains every face image in the 
        Data Base together with its label (the face image name). 
        This is used to generate the embeddings of all the face 
        images, used to evaluate the model.
"""

"""
###############################################################################
POSITIVE TEST DATASET

The positive test dataset is composed of pairs of skull test images 
and their labels. The label of a skull image is the name of the 
corresponding face image.

Get the paths of the skull positive test images. For each positive test 
individual, one of its skull images is randomly taken. The name of the 
face image of each individual (of the face test images) is also saved 
as the label.
"""
fold_pos_test_datasets = {}

# For each Fold
for i in range(0, n_folds):
    ##############
    # Get skull image paths and their labels
    pos_skull_path_list  = [] # Path of each skull test image
    pos_skull_label_list = [] # Name of each face test image

    # For each one of the positive individuals of that fold 
    for skull_ind in fold_individuals[i][0] + fold_individuals[i][1]:
        image_index = random.randint(0,num_pos_images-1) # Random image

        # Append skull image path to list
        pos_skull_path_list.append(skull_im_path + "/" + skull_ind + "/" + \
                                   os.listdir(skull_im_path + "/" + skull_ind)[image_index])

        # Append corresponding face image name to list
        pos_skull_label_list.append(df_info.loc[df_info['Individuo'] == skull_ind].iloc[0,1])


    ##############
    # Create a Tf dataset from each of the previous path lists
    pos_skull_path_dataset   = tf.data.Dataset.from_tensor_slices(pos_skull_path_list)   # Skull test images path dataset
    pos_skull_label_dataset  = tf.data.Dataset.from_tensor_slices(pos_skull_label_list)  # Face test images names dataset

    ##############
    # Create Tf dataset comprising skull image paths and their labels (with them grouped in pairs)
    fold_i_pos_test_pairs_dataset = tf.data.Dataset.zip((pos_skull_path_dataset, pos_skull_label_dataset)) # Test pairs path dataset
    fold_i_pos_test_pairs_dataset = fold_i_pos_test_pairs_dataset.shuffle(buffer_size=1024, seed=SEED)

    ##############
    # Transform dataset from one with paths to the real one with images
    
    # Map the funtions to read the images over the dataset to get the image dataset (previously the dataset contained paths)
    fold_i_pos_test_dataset = fold_i_pos_test_pairs_dataset.map(get_pair)
    
    size_pos_test = int(fold_i_pos_test_dataset.__len__().numpy())

    test_batch_size = size_pos_test

    # Batch the dataset
    fold_i_pos_test_dataset = fold_i_pos_test_dataset.batch(test_batch_size)

    # Set prefecth buffer
    fold_i_pos_test_dataset = fold_i_pos_test_dataset.prefetch(8) # Setting prefetch buffer size to 8

    ##############
    # Save the dataset to Folds dataset dictionary
    fold_pos_test_datasets[i] = fold_i_pos_test_dataset


num_pos_ind_test = 0

for i in range(0, n_folds):
    num_pos_ind_test += len(fold_individuals[i][0])


"""
###############################################################################
NEGATIVE TEST DATASET

The negative test dataset is composed of skull test images. For each 
negative test individual, one of its skull images is randomly taken.
"""

fold_neg_test_datasets = {}

# For each Fold
for i in range(0, n_folds):
    ##############
    # Get skull image paths
    neg_skull_path_list  = [] # List with teh path of each skull test image

    # For each one of the negative individuals of that fold
    for skull_ind in fold_individuals[i][2]:
        image_index = random.randint(0,num_neg_images-1) # Random image

        # Append skull image path to list
        neg_skull_path_list.append(skull_im_path + "/" + skull_ind + "/" +
                                   os.listdir(skull_im_path + "/" + skull_ind)[image_index])
        
    ##############
    # Create a Tf dataset from the previous path list
    neg_skull_path_dataset = tf.data.Dataset.from_tensor_slices(neg_skull_path_list)   # Skull test images path dataset
    neg_skull_path_dataset = neg_skull_path_dataset.shuffle(buffer_size=1024, seed=SEED)

    ##############
    # Transform dataset from one with paths to the real one with images
    
    # Map the funtions to read the images over the dataset to get the image dataset (previously the dataset contained paths)
    fold_i_neg_test_dataset = neg_skull_path_dataset.map(get_image)
    
    size_neg_test = int(fold_i_neg_test_dataset.__len__().numpy())

    test_batch_size = size_neg_test

    # Batch the dataset
    fold_i_neg_test_dataset = fold_i_neg_test_dataset.batch(test_batch_size)

    # Set prefecth buffer
    fold_i_neg_test_dataset = fold_i_neg_test_dataset.prefetch(8) # Setting prefetch buffer size to 8

    ##############
    # Save the dataset to Folds dataset dictionary
    fold_neg_test_datasets[i] = fold_i_neg_test_dataset


num_neg_ind_test = 0

for i in range(0, n_folds):
    num_neg_ind_test += len(fold_individuals[i][2])
    
"""
###############################################################################
FACE DATASET

Face dataset is composed of pairs of face images and their names. 
This datasets is common to all the folds.
"""

################################
#  UTKFACE DATASET IMAGE PATHS
all_utk_im_names = os.listdir(UTKFace_path)

all_utk_im_paths = [ UTKFace_path + "/" + face_im for face_im in all_utk_im_names ]

random.shuffle(all_utk_im_paths)


################################
# CREATE FACE DATASETS

face_db_size = 100
fold_face_datasets = {}

# For each Fold
for i in range(0, n_folds):
    pos_face_path_list  = [] # Path of each skull test image
    pos_face_label_list = [] # Name of each face test image
    
    # Append faces from positive individuals (train split) of this fold
    for skull_ind in fold_individuals[i][0]: 
        pos_face_path_list.append(face_im_dict[df_info.loc[df_info['Individuo'] == skull_ind].iloc[0,1]])
    
        # Append corresponding face image name to list
        pos_face_label_list.append(df_info.loc[df_info['Individuo'] == skull_ind].iloc[0,1])
        
    # Append faces from validation individuals of all folds
    for j in range(0, n_folds):
        for skull_ind in fold_individuals[j][1]: 
            pos_face_path_list.append(face_im_dict[df_info.loc[df_info['Individuo'] == skull_ind].iloc[0,1]])
        
            # Append corresponding face image name to list
            pos_face_label_list.append(df_info.loc[df_info['Individuo'] == skull_ind].iloc[0,1])
    
    # Take UTK faces to complete face dataset
    im_utk_paths = all_utk_im_paths[-(face_db_size-len(pos_face_path_list)):]
    
    # Delete selected images, so they are not repeated in more than one fold
    del all_utk_im_paths[-(face_db_size-len(pos_face_path_list)):]
    
    im_utk_labels = [-1 for i in range(0, len(im_utk_paths))]
    
    fold_i_face_im_paths = pos_face_path_list + im_utk_paths
    fold_i_face_im_labels = pos_face_label_list + im_utk_labels
    
    fold_i_face_path_dataset  = tf.data.Dataset.from_tensor_slices(fold_i_face_im_paths)   # Face images path dataset
    fold_i_face_label_dataset = tf.data.Dataset.from_tensor_slices(fold_i_face_im_labels)  # Face images names dataset
    
    fold_i_face_pairs_dataset = tf.data.Dataset.zip((fold_i_face_path_dataset, fold_i_face_label_dataset)) # Face pairs path dataset
    
    fold_i_face_dataset = fold_i_face_pairs_dataset.map(get_pair)
    
    fold_i_face_dataset = fold_i_face_dataset.batch(face_db_size)
    fold_i_face_dataset = fold_i_face_dataset.prefetch(8) # Setting prefetch buffer size to 8
    
    fold_face_datasets[i] = fold_i_face_dataset

# # The dictionary face_im_dict contains the name of every face image, associated to its path
# face_im_paths  = list(face_im_dict.values())
# face_im_labels = list(face_im_dict.keys())

# face_db_size = len(face_im_dict)

# face_path_dataset   = tf.data.Dataset.from_tensor_slices(face_im_paths)   # Face images path dataset
# face_label_dataset  = tf.data.Dataset.from_tensor_slices(face_im_labels)  # Face images names dataset

# face_pairs_dataset = tf.data.Dataset.zip((face_path_dataset, face_label_dataset)) # Face pairs path dataset
# face_pairs_dataset = face_pairs_dataset.shuffle(buffer_size=1024, seed=SEED)


# face_dataset = face_pairs_dataset.map(get_pair)


# n_pairs = int(face_dataset.__len__().numpy())

# face_dataset = face_dataset.batch(n_pairs)
# face_dataset = face_dataset.prefetch(8) # Setting prefetch buffer size to 8


###############################################################################
###############################################################################
###############################################################################
# RESIZE IMAGES TO FIT FACENET INPUT
"""
Images in data set are `224x224` shaped, and FaceNet
accepts `160x160` shaped images.
"""

# For each Fold
for i in range(0, n_folds):
    # Map resizing in training dataset for that fold
    fold_train_datasets[i] = fold_train_datasets[i].map(
    lambda anchor, positive, negative:
        (resize_im(anchor, faceNet_shape),
         resize_im(positive, faceNet_shape),
         resize_im(negative, faceNet_shape))
    )
    
    # Map resizing in validation dataset for that fold
    fold_val_datasets[i] = fold_val_datasets[i].map(
        lambda image, label:
            (resize_im(image, faceNet_shape), label)
    )
    
    # Map resizing in positive test dataset for that fold
    fold_pos_test_datasets[i] = fold_pos_test_datasets[i].map(
        lambda image, label:
            (resize_im(image, faceNet_shape), label)
    )
    
    # Map resizing in negative test dataset for that fold
    fold_neg_test_datasets[i] = fold_neg_test_datasets[i].map(
        lambda image: resize_im(image, faceNet_shape)
    )
    
    # Map resizing in face dataset for that fold
    fold_face_datasets[i] = fold_face_datasets[i].map(
        lambda image, label:
            (resize_im(image, faceNet_shape), label)
    )

# face_dataset = face_dataset.map(
#     lambda image, label:
#         (resize_im(image, faceNet_shape), label)
# )


###############################################################################
###############################################################################
###############################################################################
# APPLY DATA AUGMENTATION TO FACE IMAGES
"""
Random horizontal flips, random contrast adjustment and random brightness 
adjustments will be made to face images in the training dataset.

https://arxiv.org/abs/1904.11685
"""

# For each Fold
for i in range(0, n_folds):
    # Map resizing in training dataset for that fold
    fold_train_datasets[i] = fold_train_datasets[i].map(
        lambda anchor, positive, negative:
            (data_augmentation_faces(anchor), positive, negative)
    )


###############################################################################
###############################################################################
###############################################################################
# SET UP THE SIAMESE NETWORK
"""
###############################################################################
CREATE EMBEDDING GENERATOR MODEL
"""
# Create embedding generator model

embedding = load_model(faceNet_model_path)

emb_initial_weights = embedding.get_weights()

# retrainables_layers = ["Bottleneck", "Block8", "Block17", "Mixed_7a"]
retrainables_layers = ["Bottleneck", "Block8"]

for layer in embedding.layers:
    if (    any(retrain_layer in layer.name for retrain_layer in retrainables_layers) and 
            (not isinstance(layer, layers.BatchNormalization))
       ):
        layer.trainable = True
    else:
        layer.trainable = False

"""
###############################################################################
CREATE SNN WITH THE PREVIOUS EMBEDDING GENERATOR
"""
# Create snn with the previous embedding generator

anchor_input   = layers.Input(name="anchor",   shape=faceNet_shape + (3,))
positive_input = layers.Input(name="positive", shape=faceNet_shape + (3,))
negative_input = layers.Input(name="negative", shape=faceNet_shape + (3,))

# SNN inputs the anchor, positive and negative images and outputs the embedding of each image
siamese_network = Model(
    inputs  = [anchor_input, positive_input, negative_input],
    outputs = [ embedding(anchor_input),
                embedding(positive_input),
                embedding(negative_input)  ]
)

snn_initial_weights = siamese_network.get_weights()


"""
###############################################################################
CROSS VALIDATION OF THE MODEL
"""


train_epochs       = 1000             # Training epochs (not relevant, using Early Stopping)

# Early stopping to halt training when validation does not improve
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor="val_mean_pct_ranking",
        # "no longer improving" being defined as "no better than 0.005, which is 0.5%"
        min_delta=0.005,
        # "no longer improving" being further defined as "for at least 3 epochs"
        patience=3,
        mode='min',
        verbose=0,
    )
]



# Create parameter space with the next format:
#       [alpha, l2_pen, alpha_pen, epsi, l_r]

parameter_space = [              
                    [  5,  0.2, 0.2, 0.9, 1e-4], # From best top-k
                    [  5,  0.2, 0.5, 0.5, 1e-4], # From best top-k
                    [  5,  0.2, 0.2, 0.1, 1e-4], # From best top-k
                    [0.2, 0.01, 0.1, 0.1, 1e-6], # From best acc
                    [ 10,  0.2, 0.5, 0.1, 1e-6], # From best acc
                    [  5,  0.2, 0.1, 0.1, 1e-6], # From best acc
                  ]



##############
# Obtain object to save the results from file
exp_results = expResults.get_results_from_disk(results_cv_without_on_file_path,
                                               num_pos_ind_test,
                                               num_neg_ind_test,
                                               face_db_size)

for alpha_margin, l2_penalizer, alpha_penalty, \
    epsilon, learn_rate in parameter_space:
                      
    print(flush=True)
    ##############
    # Initialize results lists to be stored in disk
    model_cmc_values_ta_tr = [] # All CMC values obtained with current model (using TA-TR)
    model_nfa_values_ta_tr = [] # All NFA values obtained with current model (using TA-TR)

    model_cmc_values_raap_raan = [] # All CMC values obtained with current model (using RAAP-RAAN)
    model_nfa_values_raap_raan = [] # All NFA values obtained with current model (using RAAP-RAAN)

    model_cmc_values_tree = [] # All CMC values obtained with current model (using Tree)
    model_nfa_values_tree = [] # All NFA values obtained with current model (using Tree)

    # Create accuracies array for all folds (positive accuracy, negative accuracy and overall accuracy)
    accuracies_ta_tr     = np.empty((n_folds,3)) # (using TA-TR)
    accuracies_raap_raan = np.empty((n_folds,3)) # (using RAAP-RAAN)
    accuracies_tree      = np.empty((n_folds,3)) # (using Tree)
    
    model_val_ranking_hist = [] 
    
    model_top_k_acc      = np.empty((n_folds,3))


    # For each Test Fold. In iteration i, Fold i is used for Tets and teh rest are used for training
    for test_fold in range(0, n_folds):        
        ##############
        # Get Positive and Negative Test datasets
        pos_test_dataset = fold_pos_test_datasets[test_fold]
        neg_test_dataset = fold_neg_test_datasets[test_fold]
        
        size_pos_test = len(fold_individuals[test_fold][0])
        size_neg_test = len(fold_individuals[test_fold][2])
        
        ##############
        # Get Positive validation Test datasets
        pos_val_dataset = fold_val_datasets[test_fold]
        
        ##############
        # Get face dataset
        face_dataset = fold_face_datasets[test_fold]
        

        ##############
        # Get Training dataset from merging the other folds' training datasets
        train_folds = [nf for nf in range(0, n_folds) if nf != test_fold]

        # Merge all training datatasets corresponding to the train_folds
        train_dataset = fold_train_datasets[train_folds[0]] # Initialize train dataset to the training dataset of the first train fold

        # Concatente the training datasets of the remaining train folds
        for i in range(1, len(train_folds)):
            train_dataset = train_dataset.concatenate( fold_train_datasets[ train_folds[i] ] )
        
        
        ###############################################################################
        # CREATE MODEL AND TRAIN IT

        ##############
        # CREATE MODEL
        siamese_model = SNNTLOffline(siamese_network,             # Underneath network model
                                 embedding,                   # Underneath embedding generator model
                                 face_dataset,                # Face database for validation
                                 alpha_margin,                # Triplet Loss Margin
                                 l2_penalizer,                # L2 penalization stregth
                                 alpha_penalty, epsilon       # Conditinal Triplet Loss parameters
                                 ) 

        
        # siamese_model.compile(optimizer=optimizers.Adagrad(learn_rate))
        siamese_model.compile(optimizer=optimizers.Adam(learn_rate))
    
        
        ##############
        # Restore initial weights in embedding generator and siamese_network
        embedding.set_weights(emb_initial_weights)
        siamese_network.set_weights(snn_initial_weights)
        
        
        ##############
        # TRAIN MODEL

        print("\nModelo", alpha_margin, l2_penalizer, alpha_penalty,
              epsilon, learn_rate, flush=True)
        print("Comenzando entrenamiento del fold ", test_fold, "...", sep='', flush=True)
        
        history = siamese_model.fit(train_dataset, epochs=train_epochs,
                                    validation_data=pos_val_dataset,
                                    callbacks=callbacks,
                                    verbose=0)
        
        
        # Get hisory of validation mean ranking percentage metric
        val_ranking_hist = np.array(history.history['val_mean_pct_ranking'])
        
        model_val_ranking_hist.append(val_ranking_hist)


        ###############################################################################
        # CHECK AP AND AN DISTANCES AND GET THRESHOLDS

        ##############
        # Get A-P and A-N distances in training triplets
        ap_distances, an_distances = get_ap_an_distances(embedding, train_dataset)

        ##############
        # Obtain decision thresholds
        beta_array = get_decision_thresholds(ap_distances, an_distances)

        ##############
        # Obtain face embeddings
        face_emb_dict = obtain_face_DB_emb_dict(embedding, face_dataset)

        ##############
        # Predicions and CMC values in POSITIVE TEST DATASET
        pos_pred_ta_tr, cmc_values_ta_tr, \
        pos_pred_raap_raan, cmc_values_raap_raan, \
        pos_pred_tree, cmc_values_tree = get_pos_predictions_cmc_values_all_thresholds(embedding, pos_test_dataset,
                                                                                       size_pos_test, face_emb_dict,
                                                                                       beta_array)
        # Save CMC values to model CMC values array
        model_cmc_values_ta_tr.append(cmc_values_ta_tr)
        model_cmc_values_raap_raan.append(cmc_values_raap_raan)
        model_cmc_values_tree.append(cmc_values_tree)

        ##############
        # Predicions and CMC values in NEGATIVE TEST DATASET
        neg_pred_ta_tr, nfa_values_ta_tr, \
        neg_pred_raap_raan, nfa_values_raap_raan, \
        neg_pred_tree, nfa_values_tree = get_neg_predictions_nfa_values_all_thresholds(embedding, neg_test_dataset,
                                                                                       size_neg_test, face_emb_dict,
                                                                                       beta_array)
        # Save NFA values to model NFA values array
        model_nfa_values_ta_tr.append(nfa_values_ta_tr)
        model_nfa_values_raap_raan.append(nfa_values_raap_raan)
        model_nfa_values_tree.append(nfa_values_tree)

        ##############
        # Get positive accuracy, negative accuracy and overall accuracy with each threshold
        # Save them to accuracies array for all folds
        # Using TA-TR method
        accuracies_ta_tr[test_fold] = get_test_accuracies(cmc_values_ta_tr, nfa_values_ta_tr)

        # Using RAAP-RAAN method
        accuracies_raap_raan[test_fold] = get_test_accuracies(cmc_values_raap_raan, nfa_values_raap_raan)

        # Using Tree method
        accuracies_tree[test_fold] = get_test_accuracies(cmc_values_tree, nfa_values_tree)
        
        ##############
        # Get first top-k acc = 1.0 with each threshold.
        model_top_k_acc[test_fold] = first_max_topk_all_thresholds(cmc_values_ta_tr,
                                                                   cmc_values_raap_raan,
                                                                   cmc_values_tree,
                                                                   face_db_size+1)

    ##############
    # Format results data in order to save it to disk
    # Model parameters
    parameters_used = "SNNTLC." + \
                      "\nAlpha = " + str(alpha_margin) + \
                      "\nL2 penalizer = " + str(l2_penalizer) + \
                      "\nAlpha penality = " + str(alpha_penalty) + "(Conditional Triplet Loss)" + \
                      "\nEpsilon = " + str(epsilon) + "(Conditional Triplet Loss)" + \
                      "\nTrain epochs = " + str(train_epochs) + \
                      "\nLearn rate = " + str(learn_rate) + \
                      "\nBatch size = " +  str(train_batch_size)

    # TA-TR CMC values and NFA values
    model_cmc_values_ta_tr = np.concatenate(model_cmc_values_ta_tr, axis=0)
    model_nfa_values_ta_tr = np.concatenate(model_nfa_values_ta_tr, axis=0)

    # RAAP-RAAN CMC values and NFA values
    model_cmc_values_raap_raan = np.concatenate(model_cmc_values_raap_raan, axis=0)
    model_nfa_values_raap_raan = np.concatenate(model_nfa_values_raap_raan, axis=0)

    # Tree CMC values and NFA values
    model_cmc_values_tree = np.concatenate(model_cmc_values_tree, axis=0)
    model_nfa_values_tree = np.concatenate(model_nfa_values_tree, axis=0)

    # Accuracies
    acc_ta_tr_mean     = accuracies_ta_tr.mean(axis=0)
    acc_raap_raan_mean = accuracies_raap_raan.mean(axis=0)
    acc_tree_mean      = accuracies_tree.mean(axis=0)
    
    model_top_k_acc = model_top_k_acc.mean(axis=0)


    ##############
    # Add model results to object
    exp_results.append_model(parameters_used, beta_array,
                             model_cmc_values_ta_tr, model_nfa_values_ta_tr,
                             model_cmc_values_raap_raan, model_nfa_values_raap_raan,
                             model_cmc_values_tree, model_nfa_values_tree,
                             acc_ta_tr_mean, acc_raap_raan_mean, acc_tree_mean,
                             model_top_k_acc,
                             model_val_ranking_hist)

    ##############
    # Save results object to disk
    expResults.save_results_to_disk(exp_results, results_cv_without_on_file_path)