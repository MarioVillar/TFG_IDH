# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 12:57:35 2022

@author: mario
"""




import numpy as np
import os
import random
import pandas as pd
import math
import pickle
from operator import itemgetter
import datetime



import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import regularizers
from tensorflow.keras import losses
from tensorflow.keras import Model
from tensorflow.keras.models import load_model



from macros import *
from distance_functions import *
from threshold_detection_functions import *
from utilities import *
from test_functions import *
from snn_sigmoid import SNNSIGMOID
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

Train dataset is composed of pairs of face image and skull image. 
The label of a pair is 1 if both images belong to the same individual, 
and 0 otherwise.

To use the Tf pipeline we need to get a list with the image 
paths for the anchors, the positive and the negative images.
"""

# Number of available skull images for positive individuals
num_pos_images = 100

# Number of skull images to be used for negative individuals 
num_neg_images = 100



# Get all skull image paths of positive train individuals

# Get the path of a skull image from skull_ind:
#   For all the skull_ind that are positive train individuals
#       For each skull image of skull_ind
pos_skulls_path_list = [ skull_im_path + "/" + skull_ind + "/" + skull_im                   # Skull image path
                            for skull_ind in pos_train_individuals                           # For each positive train individual
                                for skull_im in os.listdir(skull_im_path + "/" + skull_ind)] # For each skull image of that individual


# Get face image paths associated to positive individuals' skull images

# Get the face image path from the face_im_dict of the individual skull_ind:
#   For all the skull_ind that are positive train individuals
#       For each skull image of skull_ind
pos_faces_path_list = [ face_im_dict[df_info.loc[df_info['Individuo'] == skull_ind].iloc[0,1]] # Image path gotten from dictionary with the image name (obtained from info df)
                            for skull_ind in pos_train_individuals                              # For each positive train individual 
                                for skull_im in os.listdir(skull_im_path + "/" + skull_ind)]    # For each skull image of that individual



# Get 100 skull image paths of negative individuals in order to balance the dataset

# Get the path of a skull image from skull_ind:
#   For all the skull_ind that are negative train individuals
#       For each skull image of skull_ind
#           If index is lower than the number of images needed
neg_skulls_path_list = [ skull_im_path + "/" + skull_ind + "/" + skull_im                  # Skull image path
                            for skull_ind in neg_train_individuals                          # For each individual in the skull dataset
                                for skull_im in os.listdir(skull_im_path + "/" + skull_ind) # For each skull image of that individual
                                    if int(skull_im.split("_")[3].split(".")[0]) < num_neg_images ]


# Get one face image from UTKFace dataset for each negative skull image 
num_im_utkface = len(neg_skulls_path_list)

utk_im_paths = random.sample(os.listdir(UTKFace_path), num_im_utkface)

neg_faces_path_list = [ UTKFace_path + "/" + face_im for face_im in utk_im_paths ]




# Generate labels
pos_labels = [1 for i in range(0, len(pos_faces_path_list))]
neg_labels = [0 for i in range(0, len(neg_faces_path_list))]


# No need to shuffle images in order to combine them (as with triplets in Triplet Loss SNN)


pos_faces_dataset  = tf.data.Dataset.from_tensor_slices(pos_faces_path_list)  # Positive face path dataset
pos_skulls_dataset = tf.data.Dataset.from_tensor_slices(pos_skulls_path_list) # Positive skull path dataset

pos_labels_dataset = tf.data.Dataset.from_tensor_slices(pos_labels) # Positive labels dataset

neg_faces_dataset  = tf.data.Dataset.from_tensor_slices(neg_faces_path_list)  # Negative face path dataset
neg_skulls_dataset = tf.data.Dataset.from_tensor_slices(neg_skulls_path_list) # Negative skull path dataset

neg_labels_dataset = tf.data.Dataset.from_tensor_slices(neg_labels) # Negative labels dataset


pos_pairs_dataset = tf.data.Dataset.zip(((pos_faces_dataset, pos_skulls_dataset), pos_labels_dataset)) # Positive pairs path dataset
num_pos_pairs = int(pos_pairs_dataset.__len__().numpy())

neg_pairs_dataset = tf.data.Dataset.zip(((neg_faces_dataset, neg_skulls_dataset), neg_labels_dataset)) # Negative pairs path dataset
num_pos_pairs = int(pos_pairs_dataset.__len__().numpy())

train_pos_dataset = pos_pairs_dataset.map(get_pair_im)
train_neg_dataset = neg_pairs_dataset.map(get_pair_im)

train_pos_dataset = train_pos_dataset.prefetch(8) # Setting prefetch buffer size to 8
train_neg_dataset = train_neg_dataset.prefetch(8) # Setting prefetch buffer size to 8



# pairs_dataset = pos_pairs_dataset.concatenate(neg_pairs_dataset)
# # Import images from paths
# train_dataset = pairs_dataset.map(get_pair_im)
# size_train = int(train_dataset.__len__().numpy())


# # Random shuffle all pairs
# train_dataset = train_dataset.shuffle(buffer_size=size_train, seed=SEED)

# # Batch dataset
# train_dataset = train_dataset.batch(train_batch_size)
# train_dataset = train_dataset.prefetch(8) # Setting prefetch buffer size to 8


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


pos_val_dataset = pos_val_pairs_dataset.map(get_pair)

pos_val_dataset = pos_val_dataset.shuffle(buffer_size=1024, seed=SEED)

pos_val_dataset = pos_val_dataset.batch(validation_batch_size)
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


pos_test_dataset = pos_test_pairs_dataset.map(get_pair)

size_pos_test = int(pos_test_dataset.__len__().numpy())

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


neg_test_dataset = neg_skull_path_dataset.map(get_image)

size_neg_test = int(neg_test_dataset.__len__().numpy())

neg_test_dataset = neg_test_dataset.batch(test_batch_size)
neg_test_dataset = neg_test_dataset.prefetch(8) # Setting prefetch buffer size to 8


"""
#### DEFINING FACE DATASET

Face dataset is composed of pairs of face images and their names.
"""

# The dictionary face_im_dict contains the name of every face image, associated to its path
face_im_paths  = list(face_im_dict.values())
face_im_labels = list(face_im_dict.keys())

face_db_size = len(face_im_dict)

face_path_dataset   = tf.data.Dataset.from_tensor_slices(face_im_paths)   # Face images path dataset
face_label_dataset  = tf.data.Dataset.from_tensor_slices(face_im_labels)  # Face images names dataset

face_pairs_dataset = tf.data.Dataset.zip((face_path_dataset, face_label_dataset)) # Face pairs path dataset


face_dataset = face_pairs_dataset.map(get_pair)

n_pairs = int(face_dataset.__len__().numpy())

face_dataset = face_dataset.shuffle(buffer_size=1024, seed=SEED)


face_dataset = face_dataset.batch(n_pairs)
face_dataset = face_dataset.prefetch(8) # Setting prefetch buffer size to 8


"""
###############################################################################
RESIZE IMAGES TO FIT FACENET INPUT

Images in data set are `224x224` shaped, and FaceNet accepts `160x160` shaped images.
"""

train_pos_dataset = train_pos_dataset.map(resize_pair)
    
train_neg_dataset = train_neg_dataset.map(resize_pair)


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
training dataset. Only applied to positive train dataset

https://arxiv.org/abs/1904.11685
"""
    
train_pos_dataset = train_pos_dataset.map(data_aug_pairs)   

    
    
"""
###############################################################################
CREATE FINAL TRAINING DATASET

Combine positive and negative train datasets
"""

train_dataset = train_pos_dataset.concatenate(train_neg_dataset)


# Random shuffle all pairs
train_dataset = train_dataset.shuffle(buffer_size=1024, seed=SEED)

# Batch dataset
train_dataset = train_dataset.batch(train_batch_size)
train_dataset = train_dataset.prefetch(8) # Setting prefetch buffer size to 8

    
    
"""
###############################################################################
###############################################################################
###############################################################################
# SET UP THE NEURAL NETWORK

The network has got two inpur channels (face and skull images).
It generates the embedding of both images, then combines them by
concatenating them together and generates a sigmoid output.

An output of 1 means images are totally alike, whereas an output of 0
means they are totally different.
"""


# Create embedding generator model

embedding = load_model(faceNet_model_path)

# Save weights
emb_initial_weights = embedding.get_weights()


# CREATE NEURAL NETWORK MODEL

# Two input channels
input1 = layers.Input(name="input1", shape=faceNet_shape + (3,))
input2 = layers.Input(name="input2", shape=faceNet_shape + (3,))

# Generate both embeddings
emb1 = embedding(input1)
emb2 = embedding(input2)

# model1 = Model(inputs=input1, outputs=emb1)
# model2 = Model(inputs=input2, outputs=emb2)

# Concantenate embeddings
x = layers.concatenate([emb1, emb2], axis=1)

# Add first Dense layer
x = layers.Dropout(0.7)(x)
x = layers.Dense(128, activity_regularizer=regularizers.L2(0.1))(x)
x = layers.BatchNormalization()(x)
x = layers.Activation(activations.relu)(x)

# Add second Dense layer
x = layers.Dropout(0.7)(x)
x = layers.Dense(32, activity_regularizer=regularizers.L2(0.1))(x)
x = layers.BatchNormalization()(x)
x = layers.Activation(activations.relu)(x)

# Add output neuron
x = layers.Dense(1)(x)

# Sigmoid output
output = layers.Activation(activations.sigmoid)(x)

# Create whole model
network = Model([input1, input2], output)
# model = SNNSIGMOID(_inputs=[input1, input2], _outputs=output, face_dataset=face_dataset)
model = SNNSIGMOID(network, face_dataset)

# Save weights
model_initial_weights = model.get_weights()


"""
###############################################################################
TRAINING PARAMETERS
"""
train_epochs       = 1000             # Training epochs (not relevant, using Early Stopping)

val_metric = "val_topk_acc" # "val_mean_pct_ranking"

# Early stopping to halt training when validation does not improve
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor=val_metric,
        # "no longer improving" being defined as "no better than 0.005, which is 0.5%"
        min_delta=1,
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
                        
parameter_space = [[learn_rate, retrainables_layers]
                   for learn_rate in learn_rates
                       for retrainables_layers in retrain_layers_comb
                  ]


##############
# Obtain object to save the results from file
exp_results = expResultsTrad.get_results_from_disk(results_snnsigmoid_reg_hold_out_file_path,
                                                   size_pos_test,
                                                   size_neg_test,
                                                   face_db_size)


print("\n\nSNNSIGMOID STRONGER REG\n\n")

for learn_rate, retrainables_layers in parameter_space:

    # Model parameters
    parameters_used = "SNNSIGMOID STRONGER REG." + \
                      "\nTrain epochs = " + str(train_epochs) + \
                      "\nLearn rate = " + str(learn_rate) + \
                      "\nBatch size = " +  str(train_batch_size) + \
                      "\nRetrainable layers: " + ' '.join(retrainables_layers)

                    
    """
    ###############################################################################
    SET RETRAINABLE LAYERS IN EMBEDDING GENERATOR
    """

    for layer in embedding.layers:
        # Reset trainable to false
        layer.trainable = False 
        
        # Only trainable if is in list (and it is not BatchNormalization)
        if (    any(retrain_layer in layer.name for retrain_layer in retrainables_layers) and 
                (not isinstance(layer, layers.BatchNormalization))
           ):
            layer.trainable = True

    """
    ###############################################################################
    COMPILE MODEL AND TRAIN IT
    """
    
    # Reset weights to initial ones
    embedding.set_weights(emb_initial_weights)
    model.set_weights(model_initial_weights)
    
    model.compile(optimizer=optimizers.Adam(learn_rate),
                  loss=losses.MeanSquaredError())
    # metrics.RootMeanSquaredError()
    
    print("\nSNNSIGMOID STRONGER REG", retrainables_layers, learn_rate, flush=True)
    print("Comenzando entrenamiento...", flush=True)
    history = model.fit(train_dataset, epochs=train_epochs,
                                validation_data=pos_val_dataset,
                                callbacks=callbacks,
                                verbose=1)
    
    
    # Get hisory of validation mean ranking percentage metric
    val_ranking_hist = np.array(history.history[val_metric])
    
    
    
    
    """
    ###############################################################################
    CHECK DISTANCES AND GET THRESHOLDS
    """
    ##############
    # Get predictions for positive train pairs
    pos_train_pred = model.predict(train_pos_dataset.batch(train_batch_size)).flatten()
    
    # Get predictions for negative train pairs
    neg_train_pred = model.predict(train_neg_dataset.batch(train_batch_size)).flatten()
    
    ##############
    # Obtain decision threshold
    beta_threshold, metrics = get_decision_threshold_trad_nn(pos_train_pred, neg_train_pred)    
    # beta_threshold = 0.5
    
    
    """
    ###############################################################################
    EVALUATE MODEL
    """
    
    ##############
    # CMC values in POSITIVE TEST DATASET
    cmc_values = model.get_cmc_values(pos_test_dataset, beta_threshold)
    
    ##############
    # NFA values in NEGATIVE TEST DATASET
    nfa_values = model.get_nfa_values(neg_test_dataset, beta_threshold)
    
    
    ##############
    # Get positive accuracy, negative accuracy and overall accuracy with each threshold
    # Save them to accuracies array for all folds
    accuracies = get_test_accuracies(cmc_values, nfa_values)
    
    
    ##############
    # Get first top-k acc = 1.0 with each threshold.
    top_k_accuracies = first_max_topk(cmc_values)
    
    print("Top-k acc = 1.0 en Positive Test: ", "{:.1f}".format(top_k_accuracies), flush=True)
    
    
    
    """
    ###############################################################################
    SAVE RESULTS TO DISK
    """
    
    ##############
    # Add model results to object
    exp_results.append_model(parameters_used, beta_threshold, cmc_values, nfa_values,
                              accuracies, top_k_accuracies, val_ranking_hist)
    
    ##############
    # Save results object to disk
    expResultsTrad.save_results_to_disk(exp_results, results_snnsigmoid_reg_hold_out_file_path)