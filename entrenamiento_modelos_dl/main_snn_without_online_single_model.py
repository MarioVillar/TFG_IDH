# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 12:57:35 2022

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



from global_variables import *
from general_functions import *
from threshold_detection_functions import *
from utilities import *
from test_functions import *
from siamese_keras_model import *
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
DIVIDE INDIVIDUALS INTO TRAINING AND TEST

`25%` of positive individuals and `25%` of negative individuals are 
going to be separated as the Positive Test set and the Negaitve Test set. 
The reamining `75%` of the individuals (positive and negative together) 
are going to constitute the training set.
"""
# 25% of positive individuals are chosen randomly for test
pos_test_individuals  = pos_individuals[random.sample(range(0, pos_individuals.size-1), # Both bounds are included in random sample
                                                      math.ceil(pos_individuals.size*test_split))]

# The remaining positive individuals are chosen for training
pos_train_individuals = pos_individuals[~np.in1d(pos_individuals, pos_test_individuals)]


# 25% of negative individuals are chosen randomly for test
neg_test_individuals  = neg_individuals[random.sample(range(0, neg_individuals.size-1), # Both bounds are included in random sample
                                                      math.ceil(neg_individuals.size*test_split))]

# The remaining negative individuals are chosen for training
neg_train_individuals = neg_individuals[~np.in1d(neg_individuals, neg_test_individuals)]



num_pos_ind_test = len(pos_test_individuals)
num_neg_ind_test = len(neg_test_individuals)


"""
###############################################################################
DIVIDE INDIVIDUALS INTO VALIDATION AND TRAINING

`10%` of positive train individuals are going to be separated as the 
validation set. The reamining `90%` of the positive train individuals 
(together with the negative train individuals) are going to constitute 
the training set.
"""
# 10% of positive train individuals are chosen randomly for validation
pos_val_individuals  = pos_train_individuals[random.sample(range(0, pos_train_individuals.size-1), # Both bounds are included in random sample
                                                           math.ceil(pos_train_individuals.size*val_split))]

# The remaining positive train individuals are utilisied for training
pos_train_individuals = pos_train_individuals[~np.in1d(pos_train_individuals, pos_val_individuals)]



"""
###############################################################################
TRAIN DATASET

DEFINING TRAINING TRIPLETS

Training triplets are formed by an anchor image (a face image), 
a positive example (a skull image of the same person) and a negative 
example (a face image of other person). Only the training individuals 
are used for generating these triplets.

To use the Tf pipeline we need to get a list with the image 
paths for the anchors, the positive and the negative images.
"""

# Number of available skull images for positive individuals
num_pos_images = len(os.listdir(skull_im_path + "/" + pos_train_individuals[0]))

# Number of skull images to be used for negative individuals in order to have the
# the same total number of skull images of positive and negative individuals and
# and generate the training triplets
num_neg_images = math.ceil(num_pos_images * len(pos_train_individuals) /
                           len(neg_train_individuals))

# Get face image paths associated to positive individuals' skull images

# Get the face image path from the face_im_dict of the individual skull_ind:
#   For all the skull_ind that are positive train individuals
#       For each skull image of skull_ind
anchor_path_list = [ face_im_dict[df_info.loc[df_info['Individuo'] == skull_ind].iloc[0,1]] # Image path gotten from dictionary with the image name (obtained from info df)
                        for skull_ind in pos_train_individuals                              # For each positive train individual 
                            for skull_im in os.listdir(skull_im_path + "/" + skull_ind)]    # For each skull image of that individual

# Get all skull image paths of positive train individuals

# Get the path of a skull image from skull_ind:
#   For all the skull_ind that are positive train individuals
#       For each skull image of skull_ind
pos_path_list = [ skull_im_path + "/" + skull_ind + "/" + skull_im                   # Skull image path
                    for skull_ind in pos_train_individuals                           # For each positive train individual
                        for skull_im in os.listdir(skull_im_path + "/" + skull_ind)] # For each skull image of that individual

# Get 124 skull image paths of negative individuals in order to balance the dataset

# Get the path of a skull image from skull_ind:
#   For all the skull_ind that are negative train individuals
#       For each skull image of skull_ind
#           If index is lower than the number of images needed
neg_path_list = [ skull_im_path + "/" + skull_ind + "/" + skull_im                  # Skull image path
                    for skull_ind in neg_train_individuals                          # For each individual in the skull dataset
                        for skull_im in os.listdir(skull_im_path + "/" + skull_ind) # For each skull image of that individual
                            if int(skull_im.split("_")[3].split(".")[0]) < num_neg_images ]



"""
Anchor-positive and negative images are randomly shuffled.
The element `i` of `anchor_path_list` is associated with the 
element `i` of `positive_path_list`, so both of them should be 
shuffled in the same exact order.
"""

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


anchor_dataset   = tf.data.Dataset.from_tensor_slices(anchor_path_list) # Anchors path dataset
positive_dataset = tf.data.Dataset.from_tensor_slices(pos_path_list)    # Positives path dataset
negative_dataset = tf.data.Dataset.from_tensor_slices(neg_path_list)    # Negative path dataset


triplets_dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset)) # Triplets path dataset
triplets_dataset = triplets_dataset.shuffle(buffer_size=1024, seed=SEED)


train_dataset = triplets_dataset.map(get_triplet)
n_triplets = int(train_dataset.__len__().numpy())


batch_size  = 32
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(8) # Setting prefetch buffer size to 8


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

# Get the paths of the skull positive validation images. 
# For each positive validation individual, one of its skull 
# images is randomly taken. The name of the face image of each 
# individual (of the face test images) is also saved as the label.

val_pos_skull_path_list  = [] # Path of each skull test image
val_pos_skull_label_list = [] # Name of each face test image

for skull_ind in pos_val_individuals: # For each individual in the validation skull dataset
    for image_index in range(0, 100):
        # Append skull image path to list
        val_pos_skull_path_list.append(skull_im_path + "/" + skull_ind + "/" + os.listdir(skull_im_path + "/" + skull_ind)[image_index])

        # Append corresponding face image name to list
        val_pos_skull_label_list.append(df_info.loc[df_info['Individuo'] == skull_ind].iloc[0,1])



val_pos_skull_path_dataset   = tf.data.Dataset.from_tensor_slices(val_pos_skull_path_list)   # Skull validation images path dataset
val_pos_skull_label_dataset  = tf.data.Dataset.from_tensor_slices(val_pos_skull_label_list)  # Face validation images names dataset

pos_val_pairs_dataset = tf.data.Dataset.zip((val_pos_skull_path_dataset, val_pos_skull_label_dataset)) # Test pairs path dataset
pos_val_pairs_dataset = pos_val_pairs_dataset.shuffle(buffer_size=1024, seed=SEED)


pos_val_dataset = pos_val_pairs_dataset.map(get_pair)

size_pos_val = int(pos_val_dataset.__len__().numpy())

test_batch_size = size_pos_val
pos_val_dataset = pos_val_dataset.batch(test_batch_size)
pos_val_dataset = pos_val_dataset.prefetch(8) # Setting prefetch buffer size to 8





"""
###############################################################################
TEST DATASETS

The model is evaluated over the skull images of the test individuals.

In order to do this, three datasets are defined:


*   Positive Test dataset: contains one skull image of every positive test individual, 
        together with its label. The label of a test skull image is the
        name of its corresponding face image.
*   Negative Test dataset: contains one skull image of every negativetest individual.
*   Face dataset: contains every face image in the Data Base together with its label 
        (the face image name). This is used to generate the embeddings
        of all the face images, used to evaluate the model.
"""


"""
#### DEFINING POSITIVE TEST DATASET


The positive test dataset is composed of pairs of skull test images and 
their labels. The label of a skull image is the name of the corresponding face image.

"""

pos_skull_path_list  = [] # Path of each skull test image
pos_skull_label_list = [] # Name of each face test image

for skull_ind in pos_test_individuals: # For each individual in the skull dataset
    for image_index in range(0, 100):
        # Append skull image path to list
        pos_skull_path_list.append(skull_im_path + "/" + skull_ind + "/" + os.listdir(skull_im_path + "/" + skull_ind)[image_index])

        # Append corresponding face image name to list
        pos_skull_label_list.append(df_info.loc[df_info['Individuo'] == skull_ind].iloc[0,1])


pos_skull_path_dataset   = tf.data.Dataset.from_tensor_slices(pos_skull_path_list)   # Skull test images path dataset
pos_skull_label_dataset  = tf.data.Dataset.from_tensor_slices(pos_skull_label_list)  # Face test images names dataset

pos_test_pairs_dataset = tf.data.Dataset.zip((pos_skull_path_dataset, pos_skull_label_dataset)) # Test pairs path dataset
pos_test_pairs_dataset = pos_test_pairs_dataset.shuffle(buffer_size=1024, seed=SEED)


pos_test_dataset = pos_test_pairs_dataset.map(get_pair)

size_pos_test = int(pos_test_dataset.__len__().numpy())

test_batch_size = size_pos_test
pos_test_dataset = pos_test_dataset.batch(test_batch_size)
pos_test_dataset = pos_test_dataset.prefetch(8) # Setting prefetch buffer size to 8



"""
#### DEFINING NEGATIVE TEST DATASET

The negative test dataset is composed of skull test images.

"""


neg_skull_path_list  = [] # Path of each skull test image

for skull_ind in neg_test_individuals: # For each individual in the skull dataset
    for image_index in range(0, 100):
        # Append skull image path to list
        neg_skull_path_list.append(skull_im_path + "/" + skull_ind + "/" + os.listdir(skull_im_path + "/" + skull_ind)[image_index])


neg_skull_path_dataset = tf.data.Dataset.from_tensor_slices(neg_skull_path_list)   # Skull test images path dataset
neg_skull_path_dataset = neg_skull_path_dataset.shuffle(buffer_size=1024, seed=SEED)


neg_test_dataset = neg_skull_path_dataset.map(get_image)

size_neg_test = int(neg_test_dataset.__len__().numpy())

test_batch_size = size_neg_test
neg_test_dataset = neg_test_dataset.batch(test_batch_size)
neg_test_dataset = neg_test_dataset.prefetch(8) # Setting prefetch buffer size to 8


"""
#### DEFINING FACE DATASET

Face dataset is composed of pairs of face images and their names.
"""

# The dictionary face_im_dict contains the name of every face image, associated to its path
face_im_paths  = list(face_im_dict.values())
face_im_labels = list(face_im_dict.keys())

face_path_dataset   = tf.data.Dataset.from_tensor_slices(face_im_paths)   # Face images path dataset
face_label_dataset  = tf.data.Dataset.from_tensor_slices(face_im_labels)  # Face images names dataset

face_pairs_dataset = tf.data.Dataset.zip((face_path_dataset, face_label_dataset)) # Face pairs path dataset
face_pairs_dataset = face_pairs_dataset.shuffle(buffer_size=1024, seed=SEED)


face_dataset = face_pairs_dataset.map(get_pair)


n_pairs = int(face_dataset.__len__().numpy())

face_dataset = face_dataset.batch(n_pairs)
face_dataset = face_dataset.prefetch(8) # Setting prefetch buffer size to 8


"""
###############################################################################
RESIZE IMAGES TO FIT FACENET INPUT

Images in data set are `224x224` shaped, and FaceNet accepts `160x160` shaped images.
"""

train_dataset = train_dataset.map(
    lambda anchor, positive, negative:
        (resize_im(anchor, faceNet_shape),
         resize_im(positive, faceNet_shape),
         resize_im(negative, faceNet_shape))
)


pos_val_dataset = pos_val_dataset.map(
    lambda image, label:
        (resize_im(image, faceNet_shape), label)
)


pos_test_dataset = pos_test_dataset.map(
    lambda image, label:
        (resize_im(image, faceNet_shape), label)
)


neg_test_dataset = neg_test_dataset.map(
    lambda image: resize_im(image, faceNet_shape)
)


face_dataset = face_dataset.map(
    lambda image, label:
        (resize_im(image, faceNet_shape), label)
)

    
    
"""
###############################################################################
APPLY DATA AUGMENTATION TO FACE IMAGES

Random horizontal flips, random contrast adjustment and random 
brightness adjustments will be made to face images in the 
training dataset.

https://arxiv.org/abs/1904.11685
"""
    
train_dataset = train_dataset.map(
    lambda anchor, positive, negative:
        (data_augmentation_faces(anchor), positive, negative)
)   


    
    
"""
###############################################################################
###############################################################################
###############################################################################
# SET UP THE SIAMESE NETWORK
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
TRAINING PARAMETERS
"""
alpha_margin  = 10     # Distance margin in Triplet Loss
l1_penalizer  = 0      # L1 penalization strength parameter in Triplet Loss
l2_penalizer  = 0.1      # L2 penalization strength parameter in Triplet Loss
alpha_penalty = 0    # Distance margin penalty in Conditional Triplet Loss
epsilon       = 0    # epsilon for Conditional Triplet Loss
triplet_selection = False # If True, Select hard and semi hard triplets

train_epochs = 10       # Training epochs
learn_rate   = 1e-4     # Learning rate of the Optimizer in training


# Model parameters
parameters_used = "Triplet Loss sin online generation." + \
                  "\nAlpha = " + str(alpha_margin) + \
                  "\nL1 penalizer = " + str(l1_penalizer) + \
                  "\nL2 penalizer = " + str(l2_penalizer) + \
                  "\nAlpha penality = " + str(alpha_penalty) + "(Conditional Triplet Loss)" + \
                  "\nEpsilon = " + str(epsilon) + "(Conditional Triplet Loss)" + \
                  "\nTrain epochs = " + str(train_epochs) + \
                  "\nLearn rate = " + str(learn_rate) + \
                  "\nSelect hard and semi-hard = " + str(triplet_selection)



"""
###############################################################################
CREATE MODEL AND TRAIN IT
"""

siamese_model = SiameseModel(siamese_network,             # Underneath network model
                             embedding,                   # Underneath embedding generator model
                             face_dataset,                # Face database for validation
                             alpha_margin,                # Triplet Loss Margin
                             l1_penalizer, l2_penalizer,  # L1 and L2 penalization stregth
                             alpha_penalty, epsilon,      # Conditinal Triplet Loss parameters
                             triplet_selection            # If True, Select hard and semi hard triplets
                             ) 

# siamese_model.compile(optimizer=optimizers.Adam(learn_rate))
siamese_model.compile(optimizer=optimizers.Adagrad(learn_rate))
# siamese_model.compile(optimizer=optimizers.SGD(learn_rate))


# Early stopping to halt training when validation does not improve
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor="val_mean_pct_ranking",
        # "no longer improving" being defined as "no better than 0.001, which is 0.1%"
        min_delta=0.001,
        # "no longer improving" being further defined as "for at least 3 epochs"
        patience=3,
        mode='min',
        verbose=1,
    )
]


# Reset weights to initial ones
embedding.set_weights(emb_initial_weights)
siamese_network.set_weights(snn_initial_weights)

print("Comenzando entrenameinto...")
history = siamese_model.fit(train_dataset, epochs=train_epochs,
                            validation_data=pos_val_dataset, callbacks=callbacks)


# Get hisory of validation mean ranking percentage metric
val_ranking_hist = np.array(history.history['val_mean_pct_ranking'])




"""
###############################################################################
CHECK AP AND AN DISTANCES AND GET THRESHOLDS
"""
##############
# Get A-P and A-N distances in training triplets
ap_distances, an_distances = get_ap_an_distances(embedding, train_dataset)

##############
# Obtain decision thresholds
beta_array = get_decision_thresholds(ap_distances, an_distances)

# Save plot of distances and thresholds
now = datetime.datetime.now()
time_stamp = str(now.hour+2) + "%" + str(now.minute).zfill(2) + "_" + str(now.day) + "%" + str(now.month) + "%" + str(now.year)

plot_dist_thresholds_hist(ap_distances, an_distances,
                          beta_array[0], beta_array[1], beta_array[2],
                          n_bars=50, path_savefig=path_fig_hist_thresholds+time_stamp,
                          title=parameters_used)



"""
###############################################################################
EVALUATE MODEL
"""

##############
# Obtain face embeddings
face_emb_dict = obtain_face_DB_emb_dict(embedding, face_dataset)

face_db_size = len(face_emb_dict)


##############
# Predicions and CMC values in POSITIVE TEST DATASET
pos_pred_ta_tr, cmc_values_ta_tr, \
pos_pred_raap_raan, cmc_values_raap_raan, \
pos_pred_tree, cmc_values_tree = get_pos_predictions_cmc_values_all_thresholds(embedding, pos_test_dataset,
                                                                               size_pos_test, face_emb_dict,
                                                                               beta_array)

##############
# Predicions and CMC values in NEGATIVE TEST DATASET
neg_pred_ta_tr, nfa_values_ta_tr, \
neg_pred_raap_raan, nfa_values_raap_raan, \
neg_pred_tree, nfa_values_tree = get_neg_predictions_nfa_values_all_thresholds(embedding, neg_test_dataset,
                                                                               size_neg_test, face_emb_dict,
                                                                               beta_array)

##############
# Get positive accuracy, negative accuracy and overall accuracy with each threshold
# Save them to accuracies array for all folds
# Using TA-TR method
accuracies_ta_tr = get_test_accuracies(cmc_values_ta_tr, nfa_values_ta_tr)

# Using RAAP-RAAN method
accuracies_raap_raan = get_test_accuracies(cmc_values_raap_raan, nfa_values_raap_raan)

# Using Tree method
accuracies_tree = get_test_accuracies(cmc_values_tree, nfa_values_tree)


##############
# Get first top-k acc = 1.0 with each threshold.
top_k_accuracies = first_max_topk_all_thresholds(cmc_values_ta_tr,
                                                 cmc_values_raap_raan,
                                                 cmc_values_tree,
                                                 face_db_size+1)



"""
###############################################################################
SAVE RESULTS TO DISK
"""

##############
# Obtain object to save the results from file
exp_results = expResults.get_results_from_disk(results_file_path,
                                               size_pos_test,
                                               size_neg_test,
                                               face_db_size)

##############
# Add model results to object
exp_results.append_model(parameters_used, beta_array,
                         cmc_values_ta_tr, nfa_values_ta_tr,
                         cmc_values_raap_raan, nfa_values_raap_raan,
                         cmc_values_tree, nfa_values_tree,
                         accuracies_ta_tr, accuracies_raap_raan, accuracies_tree,
                         top_k_accuracies,
                         val_ranking_hist)

##############
# Save results object to disk
expResults.save_results_to_disk(exp_results, results_file_path)