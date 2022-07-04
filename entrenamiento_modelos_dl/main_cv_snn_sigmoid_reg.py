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
from tensorflow.keras import regularizers
from tensorflow.keras import activations
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras.models import load_model



from global_variables import *
from distance_functions import *
from threshold_detection_functions import *
from utilities import *
from test_functions import *
from neural_network_model import ModelCustomVal
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
SPLIT POSITIVE INDIVIDUALS INTO TEST AND VALIDATION

When evaluating in fold i, the test individuals of that fold serve as test
and the validation individuals as validation. The rest of individuals (from
test and validation of other folds) are used as training.
"""

pos_val_ind_splits   = []
pos_test_ind_splits = []

for i in range(0, n_folds):
    # 10% of positive train individuals are chosen randomly for validation
    pos_val_ind_fold_i = pos_ind_splits[i][random.sample(range(0, pos_ind_splits[i].size-1), # Both bounds are included in random sample
                                                         math.ceil(pos_ind_splits[i].size*val_split))]

    # The remaining positive train individuals are utilisied for training
    pos_train_ind_fold_i = pos_ind_splits[i][~np.in1d(pos_ind_splits[i], pos_val_ind_fold_i)]
    
    pos_val_ind_splits.append(pos_val_ind_fold_i)
    pos_test_ind_splits.append(pos_train_ind_fold_i)



"""
The splits of individuals for each fold are saved in a dictionary
(`fold_individuals`) where the keys are the fold indexes (0..n_folds)
and the values are tuples of splits of positive test, positive validation
and negative individuals for each fold.
"""

# Create fold invdividuals dictionary
fold_individuals = {}

for i in range(0, n_folds):
    fold_individuals[i] = (pos_test_ind_splits[i], pos_val_ind_splits[i], neg_ind_splits[i])



"""
###############################################################################
UTKFACE DATASET IMAGE PATHS
"""
all_utk_im_names = os.listdir(UTKFace_path)

all_utk_im_paths = [ UTKFace_path + "/" + face_im for face_im in all_utk_im_names ]

random.shuffle(all_utk_im_paths)




"""
As each fold will be sometimes used for training and sometimes used for 
test, training and test datasets will be created for each one of the folds.
"""

"""
###############################################################################
TRAINING DATASETS

Each fold has got one positive training dataset and one negative training
dataset associated, containing pairs of face images and skull images
of the positive and negative individuals associated to that fold.

Training pairs are formed by a face image and a positive example
(a skull image of the same person) or a negative example
(a face image of other person).

Train dataset for fold i is composed by training triplets formed with
all the individuals (test and validation) of that fold. This dataset
is used for training when evaluating in the other folds.
"""

# Number of available skull images for positive individuals
num_pos_images = 100

# Number of skull images to be used for negative individuals 
num_neg_images = 100



# # UTKFace dataset divided into folds
# num_im_utkface_total = num_neg_ind * num_neg_images

# utk_im_names = random.sample(os.listdir(UTKFace_path), num_im_utkface_total)
# utk_im_paths = [ UTKFace_path + "/" + face_im for face_im in utk_im_names ]

# im_per_fold = num_im_utkface_total // n_folds



fold_pos_train_datasets = {}
fold_neg_train_datasets = {}

# For each Fold
for i in range(0, n_folds):
    ##########################################
    # Get all skull image paths of positive train individuals

    ##############
    # Path of positive skulls using test individuals
    # Get the path of a skull image from skull_ind:
    #   For all the skull_ind that are positive train individuals
    #       For each skull image of skull_ind
    pos_skulls_path_list_1 = [ skull_im_path + "/" + skull_ind + "/" + skull_im                   # Skull image path
                                    for skull_ind in fold_individuals[i][0]                           # For each positive train individual
                                        for skull_im in os.listdir(skull_im_path + "/" + skull_ind)] # For each skull image of that individual
    
    ##############
    # Path of faces associated to positive skulls using test individuals    
    # Get the face image path from the face_im_dict of the individual skull_ind:
    #   For all the skull_ind that are positive train individuals
    #       For each skull image of skull_ind
    pos_faces_path_list_1 = [ face_im_dict[df_info.loc[df_info['Individuo'] == skull_ind].iloc[0,1]] # Image path gotten from dictionary with the image name (obtained from info df)
                                    for skull_ind in fold_individuals[i][0]                              # For each positive train individual 
                                        for skull_im in os.listdir(skull_im_path + "/" + skull_ind)]    # For each skull image of that individual
    
    
    ##############
    # Path of positive skulls using validation individuals
    # Get the path of a skull image from skull_ind:
    #   For all the skull_ind that are positive train individuals
    #       For each skull image of skull_ind
    pos_skulls_path_list_2 = [ skull_im_path + "/" + skull_ind + "/" + skull_im                   # Skull image path
                                    for skull_ind in fold_individuals[i][1]                           # For each positive train individual
                                        for skull_im in os.listdir(skull_im_path + "/" + skull_ind)] # For each skull image of that individual
    
    
    ##############
    # Path of faces associated to positive skulls using validation individuals       
    # Get the face image path from the face_im_dict of the individual skull_ind:
    #   For all the skull_ind that are positive train individuals
    #       For each skull image of skull_ind
    pos_faces_path_list_2 = [ face_im_dict[df_info.loc[df_info['Individuo'] == skull_ind].iloc[0,1]] # Image path gotten from dictionary with the image name (obtained from info df)
                                    for skull_ind in fold_individuals[i][1]                              # For each positive train individual 
                                        for skull_im in os.listdir(skull_im_path + "/" + skull_ind)]    # For each skull image of that individual
    
    
    pos_skulls_path_list = pos_skulls_path_list_1 + pos_skulls_path_list_2
    pos_faces_path_list = pos_faces_path_list_1 + pos_faces_path_list_2
    
    
    
    ##############
    # Path of negative skulls using validation individuals
    # Get the path of a skull image from skull_ind:
    #   For all the skull_ind that are negative train individuals
    #       For each skull image of skull_ind
    #           If index is lower than the number of images needed
    neg_skulls_path_list = [ skull_im_path + "/" + skull_ind + "/" + skull_im                  # Skull image path
                                for skull_ind in fold_individuals[i][2]                          # For each individual in the skull dataset
                                    for skull_im in os.listdir(skull_im_path + "/" + skull_ind) # For each skull image of that individual
                                        if int(skull_im.split("_")[3].split(".")[0]) < num_neg_images ]
    
    
    ##############
    # Path of faces not associated to skulls using validation individuals
    # Get one face image from UTKFace dataset for each negative skull image 
    # neg_faces_path_list = utk_im_paths[im_per_fold*i : im_per_fold*(i+1)]
    
    # Get one face image from UTKFace dataset for each negative skull image 
    num_im_utkface = len(neg_skulls_path_list)
    
    neg_faces_path_list = all_utk_im_paths[-num_im_utkface:]
    del all_utk_im_paths[-num_im_utkface:]


    # Generate labels for positive pairs (label 1) and negative pairs (label 0)
    pos_labels = [1 for i in range(0, len(pos_faces_path_list))]
    neg_labels = [0 for i in range(0, len(neg_faces_path_list))]
    
    # No need to shuffle images in order to combine them (as with triplets in Triplet Loss SNN)
    
    pos_faces_dataset  = tf.data.Dataset.from_tensor_slices(pos_faces_path_list)  # Positive face path dataset
    pos_skulls_dataset = tf.data.Dataset.from_tensor_slices(pos_skulls_path_list) # Positive skull path dataset
    
    pos_labels_dataset = tf.data.Dataset.from_tensor_slices(pos_labels) # Positive labels dataset
    
    neg_faces_dataset  = tf.data.Dataset.from_tensor_slices(neg_faces_path_list)  # Negative face path dataset
    neg_skulls_dataset = tf.data.Dataset.from_tensor_slices(neg_skulls_path_list) # Negative skull path dataset
    
    neg_labels_dataset = tf.data.Dataset.from_tensor_slices(neg_labels) # Negative labels dataset
    
    
    ##############
    # Create Tf dataset comprising pairs of image paths (with them grouped in triplets)
    fold_i_pos_pairs_dataset = tf.data.Dataset.zip(((pos_faces_dataset, pos_skulls_dataset), 
                                                    pos_labels_dataset)) # Positive pairs path dataset
    
    fold_i_neg_pairs_dataset = tf.data.Dataset.zip(((neg_faces_dataset, neg_skulls_dataset), 
                                                    neg_labels_dataset)) # Negative pairs path dataset
    
    ##############
    # Transform dataset from one with paths to the real one with images
    fold_i_train_pos_dataset = fold_i_pos_pairs_dataset.map(get_pair_im)
    fold_i_train_neg_dataset = fold_i_neg_pairs_dataset.map(get_pair_im)
    
    fold_i_train_pos_dataset = fold_i_train_pos_dataset.prefetch(8) # Setting prefetch buffer size to 8
    fold_i_train_neg_dataset = fold_i_train_neg_dataset.prefetch(8) # Setting prefetch buffer size to 8
    

    ##############
    # Save the datasets to Folds dataset dictionaries
    fold_pos_train_datasets[i] = fold_i_train_pos_dataset
    fold_neg_train_datasets[i] = fold_i_train_neg_dataset



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
    
    fold_i_pos_val_dataset = fold_i_pos_val_dataset.batch(validation_batch_size)
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

    # For each one of the positive test individuals of that fold 
    for skull_ind in fold_individuals[i][0]:
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

    ##############
    # Transform dataset from one with paths to the real one with images
    
    # Map the funtions to read the images over the dataset to get the image dataset (previously the dataset contained paths)
    fold_i_pos_test_dataset = fold_i_pos_test_pairs_dataset.map(get_pair)

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
    
    ##############
    # Transform dataset from one with paths to the real one with images
    
    # Map the funtions to read the images over the dataset to get the image dataset (previously the dataset contained paths)
    fold_i_neg_test_dataset = neg_skull_path_dataset.map(get_image)

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

face_db_size = 100
fold_face_datasets = {}

# For each Fold
for i in range(0, n_folds):
    pos_face_path_list  = [] # Path of each skull test image
    pos_face_label_list = [] # Name of each face test image
    
    # Append faces from test individuals
    for skull_ind in fold_individuals[i][0]: 
        pos_face_path_list.append(face_im_dict[df_info.loc[df_info['Individuo'] == skull_ind].iloc[0,1]])
    
        # Append corresponding face image name to list
        pos_face_label_list.append(df_info.loc[df_info['Individuo'] == skull_ind].iloc[0,1])
        
    # Append faces from validation individuals
    for skull_ind in fold_individuals[i][1]: 
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
    # Map resizing in positive training dataset for that fold
    fold_pos_train_datasets[i] = fold_pos_train_datasets[i].map(resize_pair)
    
    # Map resizing in negative training dataset for that fold
    fold_neg_train_datasets[i] = fold_neg_train_datasets[i].map(resize_pair)
    
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
    fold_pos_train_datasets[i] = fold_pos_train_datasets[i].map(data_aug_pairs)
    


###############################################################################
###############################################################################
###############################################################################
# CREATE FINAL TRAINING DATASET
"""
Combine positive and negative train datasets
"""

fold_train_datasets = {}


# For each Fold
for i in range(0, n_folds):
    fold_i_train_dataset = fold_pos_train_datasets[i].concatenate(fold_neg_train_datasets[i])
    
    # Random shuffle all pairs
    fold_i_train_dataset = fold_i_train_dataset.shuffle(buffer_size=1024, seed=SEED)
    
    # Batch dataset
    fold_i_train_dataset = fold_i_train_dataset.batch(train_batch_size)
    fold_i_train_dataset = fold_i_train_dataset.prefetch(8) # Setting prefetch buffer size to 8
    
    fold_train_datasets[i] = fold_i_train_dataset


###############################################################################
###############################################################################
###############################################################################
# SET UP THE NEURAL NETWORK WITHOUT REGULARIZATION
"""
The network has got two inpur channels (face and skull images).
It generates the embedding of both images, then combines them by
concatenating them together and generates a sigmoid output.

An output of 1 means images are totally alike, whereas an output of 0
means they are totally different.
"""

# Create embedding generator model

embedding_con_reg = load_model(faceNet_model_path)

# Save weights
emb_initial_weights_con_reg = embedding_con_reg.get_weights()


# CREATE NEURAL NETWORK MODEL

# Two input channels
input1_con_reg = layers.Input(name="input1", shape=faceNet_shape + (3,))
input2_con_reg = layers.Input(name="input2", shape=faceNet_shape + (3,))

# Generate both embeddings
emb1_con_reg = embedding_con_reg(input1_con_reg)
emb2_con_reg = embedding_con_reg(input2_con_reg)

# Concantenate embeddings
x_con_reg = layers.concatenate([emb1_con_reg, emb2_con_reg], axis=1)

# Add first Dense layer
x_con_reg = layers.Dropout(0.7)(x_con_reg)
x_con_reg = layers.Dense(128, activity_regularizer=regularizers.L2(0.1))(x_con_reg)
x_con_reg = layers.BatchNormalization()(x_con_reg)
x_con_reg = layers.Activation(activations.relu)(x_con_reg)

# Add second Dense layer
x_con_reg = layers.Dropout(0.7)(x_con_reg)
x_con_reg = layers.Dense(32, activity_regularizer=regularizers.L2(0.1))(x_con_reg)
x_con_reg = layers.BatchNormalization()(x_con_reg)
x_con_reg = layers.Activation(activations.relu)(x_con_reg)

# Add output neuron
x_con_reg = layers.Dense(1)(x_con_reg)

# Sigmoid output
output_con_reg = layers.Activation(activations.sigmoid)(x_con_reg)

# Create whole model
network_con_reg = Model([input1_con_reg, input2_con_reg], output_con_reg)
# model = ModelCustomVal(_inputs=[input1, input2], _outputs=output, face_dataset=face_dataset)
# model_con_reg = ModelCustomVal(network_con_reg, face_dataset)

# Save weights
# model_initial_weights_con_reg = model_con_reg.get_weights()
network_initial_weights_con_reg = network_con_reg.get_weights()


"""
###############################################################################
OBTAIN OBJECT TO SAVE THE RESULTS FROM FILE
"""
exp_results = expResultsTrad.get_results_from_disk(results_cv_traditional_nn_reg_file_path,
                                               num_pos_ind_test,
                                               num_neg_ind_test,
                                               face_db_size)


"""
###############################################################################
CROSS VALIDATION OF THE MODEL WITHOUT REGULARIZATION
"""

#########################
# TRAINING PARAMETERS

train_epochs       = 1000             # Training epochs (not relevant, using Early Stopping)

# Early stopping to halt training when validation does not improve
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        # monitor="val_mean_pct_ranking",
        monitor="val_topk_acc",
        # "no longer improving" being defined as "no better than 0.005, which is 0.5%"
        min_delta=0.005,
        # "no longer improving" being further defined as "for at least 3 epochs"
        mode='min',
        patience=3,
        restore_best_weights=True,
        verbose=0,
    )
]


# Create hyperparameter space

# Learn rates
learn_rates = [1e-3, 1e-4, 1e-5, 1e-6] # Learning rate of the Optimizer in training

# Retrainable layers combinations
retrain_layers_comb = [ ["Bottleneck"],
                        ["Bottleneck", "Block8"],
                        ["Bottleneck", "Block8", "Mixed_7a"],
                        ["Bottleneck", "Block8", "Mixed_7a", "Block17"] ]
                        
parameter_space_con_reg = [ [1e-5, ["Bottleneck", "Block8", "Mixed_7a"]],
                            [1e-4, ["Bottleneck"]],
                            [1e-5, ["Bottleneck", "Block8"]],
                            [1e-5, ["Bottleneck", "Block8", "Mixed_7a", "Block17"]],
                            [1e-4, ["Bottleneck", "Block8"]]
                          ]

for learn_rate, retrainables_layers in parameter_space_con_reg:
                  
    print(flush=True)
    ##############
    # Initialize results lists to be stored in disk
    model_cmc_values = [] # All CMC values obtained with current model 
    model_nfa_values = [] # All NFA values obtained with current model 

    # Create accuracies array for all folds (positive accuracy, negative accuracy and overall accuracy)
    accuracies = np.empty((n_folds,3))
    
    model_val_ranking_hist = [] 
    
    model_top_k_acc = np.empty((n_folds))


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
        
        
        ##############
        # SET RETRAINABLE LAYERS IN EMBEDDING GENERATOR
        for layer in embedding_con_reg.layers:
            # Reset trainable to false
            layer.trainable = False 
            
            # Only trainable if is in list (and it is not BatchNormalization)
            if (    any(retrain_layer in layer.name for retrain_layer in retrainables_layers) and 
                    (not isinstance(layer, layers.BatchNormalization))
               ):
                layer.trainable = True
    
        
        ##############
        # COMPILE MODEL 
        
        model_con_reg = ModelCustomVal(network_con_reg, face_dataset)
        
        # Reset weights to initial ones
        embedding_con_reg.set_weights(emb_initial_weights_con_reg)
        # model_con_reg.set_weights(model_initial_weights_con_reg)
        network_con_reg.set_weights(network_initial_weights_con_reg)
        
        model_con_reg.compile(optimizer=optimizers.Adam(learn_rate),
                              loss=losses.MeanSquaredError())
        
        
        ##############
        # TRAIN MODEL

        print("\nModelo", retrainables_layers, learn_rate, flush=True)
        print("Comenzando entrenamiento del fold ", test_fold, "...", sep='', flush=True)
        
        history = model_con_reg.fit(train_dataset, epochs=train_epochs,
                            validation_data=pos_val_dataset,
                            callbacks=callbacks,
                            verbose=0)
        
        
        # Get hisory of validation mean ranking percentage metric
        val_ranking_hist = np.array(history.history['val_topk_acc'])
        
        model_val_ranking_hist.append(val_ranking_hist)


        ###############################################################################
        # CHECK DISTANCES AND GET THRESHOLDS
        
        ##############
        # Get Positive Training dataset from merging the other folds' positive training datasets
        # Merge all training datatasets corresponding to the train_folds
        pos_train_dataset = fold_pos_train_datasets[train_folds[0]] # Initialize train dataset to the training dataset of the first train fold

        # Concatente the training datasets of the remaining train folds
        for i in range(1, len(train_folds)):
            pos_train_dataset = pos_train_dataset.concatenate( fold_pos_train_datasets[ train_folds[i] ] )
        
        # Get predictions for positive train pairs
        pos_train_pred = model_con_reg.predict(pos_train_dataset.batch(train_batch_size)).flatten()
        
        ##############
        # Get Negative Training dataset from merging the other folds' negative training datasets
        # Merge all training datatasets corresponding to the train_folds
        neg_train_dataset = fold_neg_train_datasets[train_folds[0]] # Initialize train dataset to the training dataset of the first train fold

        # Concatente the training datasets of the remaining train folds
        for i in range(1, len(train_folds)):
            neg_train_dataset = neg_train_dataset.concatenate( fold_neg_train_datasets[ train_folds[i] ] )
            
        # Get predictions for negative train pairs
        neg_train_pred = model_con_reg.predict(neg_train_dataset.batch(train_batch_size)).flatten()
        
        ##############
        # Obtain decision threshold
        beta_threshold, metrics = get_decision_threshold_trad_nn(pos_train_pred, neg_train_pred)  
        
        
        ###############################################################################
        # EVALUATE MODEL

        ##############
        # CMC values in POSITIVE TEST DATASET
        cmc_values = model_con_reg.get_cmc_values(pos_test_dataset, beta_threshold)
        
        # Save CMC values to model CMC values array
        model_cmc_values.append(cmc_values)
        
        ##############
        # NFA values in NEGATIVE TEST DATASET
        nfa_values = model_con_reg.get_nfa_values(neg_test_dataset, beta_threshold)
        
        # Save NFA values to model NFA values array
        model_nfa_values.append(nfa_values)
        
        ##############
        # Get positive accuracy, negative accuracy and overall accuracy with each threshold
        # Save them to accuracies array for all folds
        accuracies[test_fold] = get_test_accuracies(cmc_values, nfa_values)
        
        ##############
        # Get first top-k acc = 1.0 with each threshold.
        model_top_k_acc[test_fold] = first_max_topk(cmc_values)
        

    ##############
    # Format results data in order to save it to disk
    # Model parameters
    parameters_used = "FaceNet (normal ConvNet) WITH REG." + \
                      "\nTrain epochs = " + str(train_epochs) + \
                      "\nLearn rate = " + str(learn_rate) + \
                      "\nBatch size = " +  str(train_batch_size) + \
                      "\nRetrainable layers: " + ' '.join(retrainables_layers)

    # CMC values and NFA values
    model_cmc_values = np.concatenate(model_cmc_values, axis=0)
    model_nfa_values = np.concatenate(model_nfa_values, axis=0)

    # Accuracies
    accuracies = accuracies.mean(axis=0)
    
    model_top_k_acc = model_top_k_acc.mean()


    ##############
    # Add model results to object
    exp_results.append_model(parameters_used, beta_threshold, 
                             model_cmc_values, model_nfa_values,
                             accuracies, model_top_k_acc, 
                             val_ranking_hist)

    ##############
    # Save results object to disk
    expResults.save_results_to_disk(exp_results, results_cv_traditional_nn_reg_file_path)