# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 12:51:43 2022

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
from distance_functions import *



"""
###############################################################################
OBTAIN FACE EMBEDDINGS (IN TEST)

Face embeddings are returned in a dictionary where the keys are the face
labels and the values are the face embeddings.
"""

"""
Obtain the embedding of every face image in the Data Base.
Parameters:
    emb_generator, the embedding generator model
    face_dataset,  the face dataset, which includes each face image of the DB together
        with its label (its name)
Returns (packed in a list):
    face_emb_dict, dictionary where keys are face labels and values are face embeddings
"""
def obtain_face_DB_emb_dict(emb_generator, face_dataset):
    # Get number of batches in face dataset
    num_batches = int(face_dataset.__len__().numpy())

    # Initialize iterator
    iterator = iter(face_dataset)

    # Initialize dictionary
    face_emb_dict = {}

    # For each batch in face dataset
    for n in range(0, num_batches):
        # Get next batch of face images and their labels
        face_image, face_label = iterator.get_next()

        # Obtain batch of embeddings of the face images in the batch
        face_emb = emb_generator(face_image)

        # Extend dictionary with each one of the face labels in the bacth and its embedding
        for i in range(0, len(face_label)):
            face_emb_dict[int(face_label[i].numpy())] = face_emb[i]
    
    return face_emb_dict



"""
WARNING: FUNCTION MEANT TO USE ONLY WITH POSITIVE TEST DATASETS

For every test skull image, get the face predicted and the ranking at which the
real face image is predicted. If an individual is not matched to a image, the
prediction will be None. This means that if the distance of the associated
image to the skull is the k best, then:
    i. If no prediction was made (no face image distance was lower than beta)
        then that skull is ranked k+1.
    ii. If a prediction was made (at least one face image distance was lower
        than beta) then that skull is ranked k.
For each test skull image, its embedding is obatined and compared to every face
embedding in the Data base. The distance ranking is calculated.
Parameters:
    emb_generator, the embedding generator model
    test_data, the test dataset, which includes every test skull image together with
        its label (the corresponding face image name)
    size_test, number of individuals in test dataset. This parameters is necesary as
        test dataset is batched, so the total number of elements cannot be obtained
    face_embeddings, dictionary where keys are face labels and values are face embeddings
    beta, decision threshold. Any pair of images whose embedding distance is greater than
        beta cannot be matched together
Returns:
    predictions, array with each test individual prediction made
    rankings, array with each test individual ranking
"""
def get_pos_predictions_cmc_values(emb_generator, test_data, size_test, face_embeddings, beta):
    # Get number of batches in skull test dataset
    num_batches = int(test_data.__len__().numpy())

    # Initialize predictions array
    predictions = np.full(size_test, None)

    # Initialize distance of the prediction
    dist_pred = np.full(size_test, np.inf) # Used to obtain minimum distance between embeddings, to generate the prediction

    # Initialize rankings array to first position
    rankings = np.ones(size_test)

    # Initialize iterator
    iterator = iter(test_data)

    # For each batch in test skull dataset
    for n in range(0, num_batches):
        # Get next batch of skull images and their labels
        skull_im, skull_label = iterator.get_next() # These are batch_size (see below) individuals

        # Get bacth size (number of individuals in batch)
        batch_size = len(skull_im)

        # Obtain embeddings of the batch of skull images
        skull_emb = emb_generator(skull_im) # These are batch_size individuals

        # Get true face images embeddings for each individual in the batch
        real_face_emb = itemgetter(*skull_label.numpy())(face_embeddings) # Accesing dictionary with multiple keys simultaneously

        # Get distances between each skull in batch and their associated face embedding (the real face embedding)
        real_dist = emb_distance(skull_emb, real_face_emb).numpy().flatten()
        
        ######
        # Update prediction if real embedding distance is lower than beta:
        # Get those skulls for which the real face embedding has less distance than beta
        index_update_in_batch = np.where(real_dist < beta)[0]

        # Get update indexes in prediction vector (outside batch)
        index_update = index_update_in_batch + n * batch_size

        # Update in predictions vector only those predictions for which distance was lower than beta
        predictions[index_update] = skull_label.numpy()[index_update_in_batch]

        # Update in predictions distance vector only those predictions for which distance was lower than beta
        dist_pred[index_update] = real_dist[index_update_in_batch]

        prueba = 1

        # For each face embedding in face_embeddings dictionary
        for f_label, f_emb in face_embeddings.items():
            num_replicate = tf.constant([batch_size], tf.int32) # Tf tensor with number of skulls in batch (size of batch)
            
            # Get distance between each skull in batch and the current face embedding
            dist = emb_distance( skull_emb,         # Batch of skull embeddings
                                 tf.reshape(tf.tile(f_emb, num_replicate), # Replicate face embedding batch_size times to match skull batch shape
                                            (batch_size, len(f_emb)))
                                ).numpy().flatten()

            ######
            # Update predictions where needed
            # Get the skulls whose distance to current face embedding is lower than beta and less than current prediction
            index_update_in_batch = np.where((dist < beta) &
                                             (dist < dist_pred[n*batch_size:(n+1)*batch_size]) # Prediction distances of the current batch
                                             )[0] 

            # Get update indexes in prediction vector (outside batch)
            index_update = index_update_in_batch + n * batch_size

            # Update in predictions vector only those predictions for which distance was lower than current prediction
            predictions[index_update] = f_label

            # Update in predictions distance vector only those predictions for which distance was lower than current prediction
            dist_pred[index_update] = dist[index_update_in_batch]

            ######
            # Update rankings where needed
            # Get those skulls for which the current face embedding has less distance than the real image
            index_less_dist = np.where(dist < real_dist)

            # Check if current face is the real face of some test individuals
            index_is_real = np.where(skull_label.numpy() == f_label)

            # Only update those individuals whose real face image is not the current one
            index_update = np.setdiff1d(index_less_dist, index_is_real) # This index is within the batch

            # Get update indexes in ranking vector (outside batch)
            index_update = index_update + n * batch_size

            # Add one position in ranking to those individuals
            rankings[index_update] += 1

        ######
        # Update rankings of not-macthed individuals
        # Get individuals who have not been matched to any face image
        index_update = np.where(predictions == None)

        # Increase respective rankings by one
        rankings[index_update] += 1

    return predictions, rankings


"""
WARNING: FUNCTION MEANT TO USE ONLY WITH POSITIVE TEST DATASETS

Get predictions and CMC values using each one of the three thresholds.
Parameters:
    emb_generator, the embedding generator model
    test_data, the test dataset, which includes every test skull image together with
        its label (the corresponding face image name)
    size_test, number of individuals in test dataset. This parameters is necesary as
        test dataset is batched, so the total number of elements cannot be obtained
    face_embeddings, dictionary where keys are face labels and values are face embeddings
    beta_array, array with the three decision thresholds
Returns:
    pos_pred_ta_tr, (using TA-TR method) array with each test individual prediction made
    cmc_values_ta_tr, (using TA-TR method) array with each test individual ranking
    pos_pred_raap_raan, (using RAAP-RAAN method) array with each test individual prediction made
    cmc_values_raap_raan, (using RAAP-RAAN method) array with each test individual ranking
    pos_pred_tree, (using Decision Tree method) array with each test individual prediction made
    cmc_values_tree, (using Decision Tree method) array with each test individual ranking
"""
def get_pos_predictions_cmc_values_all_thresholds(emb_generator, test_data, size_test,
                                                  face_embeddings, beta_array):
    pos_pred_ta_tr, cmc_values_ta_tr = get_pos_predictions_cmc_values(emb_generator, test_data,
                                                                      size_test, face_embeddings,
                                                                      beta_array[0])
    
    pos_pred_raap_raan, cmc_values_raap_raan = get_pos_predictions_cmc_values(emb_generator, test_data,
                                                                              size_test, face_embeddings,
                                                                              beta_array[1])
    
    pos_pred_tree, cmc_values_tree = get_pos_predictions_cmc_values(emb_generator, test_data,
                                                                    size_test, face_embeddings,
                                                                    beta_array[2])
    
    return pos_pred_ta_tr, cmc_values_ta_tr, pos_pred_raap_raan, cmc_values_raap_raan, pos_pred_tree, cmc_values_tree



"""
WARNING: FUNCTION MEANT TO USE ONLY WITH NEGATIVE TEST DATASETS

For every test skull image, get the face predicted. If an individual is not
matched to a image, the prediction will be None.
In order to obtain these predictions, for each test skull image its
embedding is calculated and compared to every face embedding in the Data
base, returning the one with lower distance (if this distance is less than
the decision threshold) or None if no distance is lower than the threshold.
The number of false accepts for each individual is also computed and returned.
Parameters:
    emb_generator, the embedding generator model
    test_data, the test dataset, which includes every test skull image together with
        its label (the corresponding face image name)
    size_test, number of individuals in test dataset. This parameters is necesary as
        test dataset is batched, so the total number of elements cannot be obtained
    face_embeddings, dictionary where keys are face labels and values are face embeddings
    beta, decision threshold. Any pair of images whose embedding distance is greater than
        beta cannot be matched together
Returns:
    predictions, array with each test individual prediction made
    nfas, array with each test individual nfa
"""
def get_neg_predictions_nfa_values(emb_generator, test_data, size_test, face_embeddings, beta):
    # Get number of batches in skull test dataset
    num_batches = int(test_data.__len__().numpy())

    # Initialize predictions array
    predictions = np.full(size_test, None)

    # Initialize distance of the prediction
    dist_pred = np.full(size_test, np.inf) # Used to obtain minimum distance between embeddings, to generate the prediction

    # Initialize rankings array to first position
    nfa_values = np.zeros(size_test, dtype='int')

    # Initialize iterator
    iterator = iter(test_data)

    # For each batch in test skull dataset
    for n in range(0, num_batches):
        # Get next batch of skull images and their labels
        skull_im = iterator.get_next() # These are batch_size (see below) individuals

        # Get bacth size (number of individuals in batch)
        batch_size = len(skull_im)

        # Obtain embeddings of the batch of skull images
        skull_emb = emb_generator(skull_im) # These are batch_size individuals

        # For each face embedding in face_embeddings dictionary
        for f_label, f_emb in face_embeddings.items():
            num_replicate = tf.constant([batch_size], tf.int32) # Tf tensor with number of skulls in batch (size of batch)
            
            # Get distance between each skull in batch and the current face embedding
            dist = emb_distance( skull_emb,         # Batch of skull embeddings
                                 tf.reshape(tf.tile(f_emb, num_replicate), # Replicate face embedding batch_size times to match skull batch shape
                                            (batch_size, len(f_emb)))
                                ).numpy().flatten()

            ######
            # Update predictions where needed
            # Get the skulls whose distance to current face embedding is lower than beta and less than current prediction
            index_update_in_batch = np.where((dist < beta) &
                                             (dist < dist_pred[n*batch_size:(n+1)*batch_size]) # Prediction distances of the current batch
                                             )[0] 

            # Get update indexes in prediction vector (outside batch)
            index_update = index_update_in_batch + n * batch_size

            # Update in predictions vector only those predictions for which distance was lower than current prediction
            predictions[index_update] = f_label

            # Update in predictions distance vector only those predictions for which distance was lower than current prediction
            dist_pred[index_update] = dist[index_update_in_batch]

            ######
            # Update NFAs if real embedding distance is lower than beta
            # Get the skulls whose distance to current face embedding is lower than beta
            index_update_in_batch = np.where(dist < beta)[0] 

            # Get update indexes in prediction vector (outside batch)
            index_update = index_update_in_batch + n * batch_size

            # Update NFAs for those individuals 
            nfa_values[index_update] += 1

    return predictions, nfa_values



"""
WARNING: FUNCTION MEANT TO USE ONLY WITH NEGATIVE TEST DATASETS

Get negative predictions and nfa values for all thresholds.
Parameters:
    emb_generator, the embedding generator model
    test_data, the test dataset, which includes every test skull image together with
        its label (the corresponding face image name)
    size_test, number of individuals in test dataset. This parameters is necesary as
        test dataset is batched, so the total number of elements cannot be obtained
    face_embeddings, dictionary where keys are face labels and values are face embeddings
    beta_array, array with all three decision thresholds
Returns:
    neg_pred_ta_tr, (using TA-TR method) array with each test individual prediction made
    nfa_values_ta_tr, (using TA-TR method) array with each test individual NFA
    neg_pred_raap_raan, (using RAAP-RAAN method) array with each test individual prediction made
    nfa_values_raap_raan, (using RAAP-RAAN method) array with each test individual NFA
    neg_pred_tree, (using Decision Tree method) array with each test individual prediction made
    nfa_values_tree,  (using Decision Tree method) array with each test individual NFA
"""
def get_neg_predictions_nfa_values_all_thresholds(emb_generator, test_data, size_test, face_embeddings,
                                                  beta_array):
    neg_pred_ta_tr, nfa_values_ta_tr = get_neg_predictions_nfa_values(emb_generator, test_data,
                                                                      size_test, face_embeddings,
                                                                      beta_array[0])
    
    neg_pred_raap_raan, nfa_values_raap_raan = get_neg_predictions_nfa_values(emb_generator, test_data,
                                                                              size_test, face_embeddings,
                                                                              beta_array[1])
    
    neg_pred_tree, nfa_values_tree = get_neg_predictions_nfa_values(emb_generator, test_data,
                                                                    size_test, face_embeddings,
                                                                    beta_array[2])
    
    return neg_pred_ta_tr, nfa_values_ta_tr, neg_pred_raap_raan, nfa_values_raap_raan, neg_pred_tree, nfa_values_tree



"""
###############################################################################
TEST ACCURACY (IN TEST)
"""

def get_test_accuracies(cmc_values, nfa_values):
    pos_acc     = np.count_nonzero(cmc_values == 1) / len(cmc_values)
    neg_acc     = np.count_nonzero(nfa_values == 0) / len(nfa_values)
    overall_acc = (pos_acc + neg_acc) / 2

    return np.array([pos_acc, neg_acc, overall_acc])


"""
###############################################################################
FIRST TOP-K ACCURACY EQUAL TO 1.0 (IN TEST)
"""

def first_max_topk(values, sample_size=None):
    topk_max = max(values)
    
    if sample_size == None:
        return topk_max
    else:
        pct_sample = topk_max / sample_size * 100
        return topk_max, int(pct_sample)


def first_max_topk_all_thresholds(cmc_values_ta_tr, cmc_values_raap_raan,
                                  cmc_values_tree, sample_size):
    topk_max_ta_tr, pct_ta_tr         = first_max_topk(cmc_values_ta_tr, sample_size)
    topk_max_raap_raan, pct_raap_raan = first_max_topk(cmc_values_raap_raan, sample_size)
    topk_max_tree, pct_tree           = first_max_topk(cmc_values_tree, sample_size)
    
    return np.array([topk_max_ta_tr, topk_max_raap_raan, topk_max_tree])







