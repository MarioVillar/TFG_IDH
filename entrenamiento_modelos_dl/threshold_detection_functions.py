# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 12:46:22 2022

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
GET DISTANCES BETWEEN ANCHOR-POSITIVE AND ANCHOR-NEGATIVE (AFTER TRAINING)
"""

"""
Get the distances between the embeddings of each pair of anchor-positive images
and between the embeddings of each pair of anchor-negative images.
Parameters:
    emb_generator, the embedding generator model
    dataset, triplets dataset
Returns:
    ap_distances, np array with the distances between each pair anchor-positive
    an_distances, np array with the distances between each pair anchor-negative
"""
def get_ap_an_distances(emb_generator, dataset):
    # Get number of batches in triplets dataset
    num_batches = int(dataset.__len__().numpy())

    # Initialize dataset iterator
    iterator = iter(dataset)

    # Initialize distances arrays
    ap_distances = np.array((), dtype='float32')
    an_distances = np.array((), dtype='float32')

    # For each batch
    for n in range(0,num_batches):
        # Get next batch of triplets
        anchors, positives, negatives = iterator.get_next()

        # Obtain embeddings of all images in batch
        anchor_embs, positive_embs, negative_embs = (
            emb_generator(anchors),
            emb_generator(positives),
            emb_generator(negatives),
        )

        # Get distances between all pairs of anchor-positive and anchor-negative images in the batch
        ap_dist_batch, an_dist_batch = triplets_distance(anchor_embs, positive_embs, negative_embs)
        ap_dist_batch, an_dist_batch = ap_dist_batch.numpy(), an_dist_batch.numpy() # Convert to np array

        # Save them to distances arrays
        ap_distances = np.concatenate((ap_distances, ap_dist_batch.flatten()))
        an_distances = np.concatenate((an_distances, an_dist_batch.flatten()))
            
    return ap_distances, an_distances



"""
Get the distances between the embeddings of each pair of anchor-positive images
and between the embeddings of each pair of anchor-negative images.
Parameters:
    emb_generator, the embedding generator model
    dataset, triplets dataset
Returns:
    ap_distances, np array with the distances between each pair anchor-positive
    an_distances, np array with the distances between each pair anchor-negative
"""
def get_ap_an_distances_online(emb_generator, dataset):
    # Get number of batches in triplets dataset
    num_batches = int(dataset.__len__().numpy())

    # Initialize dataset iterator
    iterator = iter(dataset)

    # Initialize distances arrays
    ap_distances = np.array((), dtype='float32')
    an_distances = np.array((), dtype='float32')

    # For each batch
    for n in range(0,num_batches):
        # Get next batch of triplets
        data, labels = iterator.get_next()
        (anchors, positives, negatives) = data

        # Obtain embeddings of all images in batch
        anchor_embs, positive_embs, negative_embs = (
            emb_generator(anchors),
            emb_generator(positives),
            emb_generator(negatives),
        )

        # Get distances between all pairs of anchor-positive and anchor-negative images in the batch
        ap_dist_batch, an_dist_batch = triplets_distance(anchor_embs, positive_embs, negative_embs)
        ap_dist_batch, an_dist_batch = ap_dist_batch.numpy(), an_dist_batch.numpy() # Convert to np array

        # Save them to distances arrays
        ap_distances = np.concatenate((ap_distances, ap_dist_batch.flatten()))
        an_distances = np.concatenate((an_distances, an_dist_batch.flatten()))
            
    return ap_distances, an_distances



"""
###############################################################################
OBTAIN DECISION THRESHOLDS (AFTER TRAINING)

The decision threshold `β` determines whether a new image can be matched to a
face image or not. If the distance between the new image embedding and the face
image embedding is below the threshold, then both images are considered similar
enough and can be matched; in other case, they are considered sufficiently
differnt to discard the math.
"""

"""
####################################################
### FIRST ALGORITHM (TA-TR ALGORITHM)

This algorithm is called TA-TR (True accepts - True rejects algorithm)
algorithm.

For a specific threshold, the following metrics are defined:

*   Anchor-positive **T**rue **A**ccepts (**TA**): number of anchor-positive
        pairs accepted with the threshold, those anchor-positive pairs
        whose distance is lower than the threshold.
*   Anchor-negative **T**rue **R**ejects (**TR**): number of anchor-negative
        pairs rejected with the threshold, those anchor-negative pairs
        whose distance is greater than the threshold.

The optimal threshold is obtain from training triplets as that threshold
that maximizes the sum of the ratio of TA plus the ratio of TR.
"""

"""
Get the optimal TA-TR decision threshold for training triplets.
The optimal threshold is obtain from training triplets as that
threshold that maximizes the sum of the anchor-positive true
accepts (ap_ta) plus the anchor-negative true rejects (an_tr).
Parameters:
    ap_dists, np array with the distances between each pair anchor-positive
    an_dists, np array with the distances between each pair anchor-negative
Returns:
    optimal_beta, optimal decision threshold
    max_metric, anchor-positive true accepts and anchor-negative true rejects
        sum for optimal beta
    opt_ap_ta, anchor-positive true accepts for optimal beta
    opt_an_tr, anchor-negative true rejects for optimal beta
"""
def get_decision_threshold_ta_tr(ap_dists, an_dists):
    # Get all possible thresholds
    possible_beta = np.unique(np.concatenate((ap_dists, an_dists)))

    opt_beta   = -1 # Optimal threshold 
    opt_ap_ta  = 0 # Anchor-positive True accepts
    opt_an_tr  = 0 # Anchor-negative true rejects
    max_metric = 0 # Sum of both previous metrics

    # For each possible threshold
    for beta in possible_beta:
        ap_ta = np.where(ap_dists < beta)[0].size
        an_tr = np.where(beta < an_dists)[0].size

        if (ap_ta + an_tr) > max_metric:
            opt_beta = beta
            max_metric = ap_ta + an_tr
            opt_ap_ta = ap_ta
            opt_an_tr = an_tr
    
    return opt_beta, max_metric, opt_ap_ta, opt_an_tr

"""
####################################################
SECOND ALGORITHM (RAAP-RAAN ALGORITHM)

This algorithm is called RAAP-RAAN (defined below) algorithm.

For a specific threshold, the histogram of distances is buildt and
the following two metrics are defined:

*   **R**atio of **A**rea under **A**nchor-**P**ositive true accepts (**RAAP**).
        The area under anchor-positive bins with lower distance than
        threshold distance is calculated (via integral) and dividided
        by total anchor-positive bins area to obtain the ratio.
*   **R**atio of **A**rea under **A**nchor-**N**egative true rejects (**RAAN**).
        Simmilarly, the area under anchor-ngeative bins with greater
        distance than threshold distance is calculated (via integral)
        and dividided by total anchor-negative bins area to obtain
        the ratio.

The optimal threshold is obtained from training triplets as the threshold
that maximizes the metric `RAAP*RAAN`.
"""

"""
Compute the area under the histogram of the distances with the bins
and bin widths provided. The area is obtained by multiplying the number of
samples in each bin with the width of the respective bin.
Parameters:
    distances, np array with the distance samples
    bins, limits between histogram bins
    bin_widths, width of each bin
Returns:
    total_area, the area under the histogram
"""
def get_area_under_hist(distances, bins, bin_widths):
    # Append zero as it is the first bin limit
    bins = np.append(bins, 0)

    # Initialize area
    total_area = 0

    # For each bin created
    for i in range(0, len(bins)-1):
        # Number of pairs in that bin
        # Count the number of pairs whose distance is greater than inferiour bound of bin and lower than superiour bound of bin
        n_pairs = np.count_nonzero(np.all([bins[i-1] < distances, distances <= bins[i]], axis=0))

        # Sum area of bin to total accumulated area
        total_area += n_pairs * bin_widths[i]
    
    return total_area


"""
Get the optimal RAAP-RAAN decision threshold for training triplets.
RAAP belongs to Ratio of area under anchor-positive true accepts and
RAAN belongs to Ratio of area under anchor-negative true rejects.
The optimal threshold is obtain from training triplets as that
threshold that maximizes the multiplication RAAP*RAAN.
Parameters:
    ap_dists, np array with the distances between each pair anchor-positive
    an_dists, np array with the distances between each pair anchor-negative
Returns:
    opt_beta, optimal threshold
    max_metrix, optimal metric
    opt_raap, RAAP for optimal threshold
    opt_raan, RAAN for optimal threshold
"""
def get_decision_threshold_raap_raan(ap_dists, an_dists):
    ap_dists = np.sort(ap_dists) # Sort anchor-positive distances array
    an_dists = np.sort(an_dists) # Sort anchor-negative distances array

    # Get all possible distances in data
    possible_dists = np.unique(np.concatenate((ap_dists, an_dists)))

    # Generate threshold space, evenly distributed between zero and maximum distance in data
    possible_beta = np.linspace(0, np.max(possible_dists), len(possible_dists))

    # Initialize RAAP for each possible threshold
    beta_raaps   = np.zeros(len(possible_beta))
    # Initialize RAAN for each possible threshold
    beta_raans   = np.zeros(len(possible_beta))
    # Initialize Metric for each possible threshold
    beta_metrics = np.zeros(len(possible_beta))

    # Bin widths
    bin_widths = possible_beta - np.append(0, possible_beta)[:-1]

    # Total area under AP distances histogram
    total_area_ap = get_area_under_hist(ap_dists, possible_beta, bin_widths)
    # Total area under AN distances histogram
    total_area_an = get_area_under_hist(an_dists, possible_beta, bin_widths)

    # Cummulative area on the left of the threshold
    # Threshold will start in 0, so at the beggining there is no area
    cum_area_ap = 0

    # Cummulative area on the right of the threshold
    # Threshold will start in 0, so at the begging it is the total area
    cum_area_an  = total_area_an

    # Each beta constitutes a bin
    for i in range(0, len(possible_beta)):
        # Current threshold
        beta = possible_beta[i]

        # Previous threshold
        if i == 0:
            beta_previous = 0
        else:
            beta_previous = possible_beta[i-1]

        # Count number of A-P pairs with distance equal to beta threshold
        ap_pair_indexes = np.count_nonzero(np.logical_and(beta_previous < ap_dists,
                                                          ap_dists <= beta))
        
        # Compute area of current bin and add to cummulative area on the left of threshold
        cum_area_ap += ap_pair_indexes * bin_widths[i]


        # Count number of A-N pairs with distance equal to beta threshold
        an_pair_indexes = np.count_nonzero(np.logical_and(beta_previous < an_dists,
                                                          an_dists <= beta))
        
        # Compute area of current bin and subtract to cummulative area on the left of threshold
        cum_area_an -= an_pair_indexes * bin_widths[i]

        # Update RAAP for current threshold
        beta_raaps[i] = cum_area_ap / total_area_ap

        # Update RAAN for current threshold
        beta_raans[i] = cum_area_an / total_area_an

        # Update metric for current threshold
        beta_metrics[i] = beta_raaps[i] * beta_raans[i]
    
    # Get index of optimal beta as the threshold with maximum metric
    index_opt_beta = np.argmax(beta_metrics)

    # Get optimal beta, optimal metric and RAAP and RAAN for optimal threshold
    opt_beta   = possible_beta[index_opt_beta]
    max_metrix = beta_metrics[index_opt_beta]
    opt_raap   = beta_raaps[index_opt_beta]
    opt_raan   = beta_raans[index_opt_beta]

    return opt_beta, max_metrix, opt_raap, opt_raan

"""
####################################################
THIRD ALGORITHM (DECISION TREE)
"""

"""
Get the optimal Decision Tree decision threshold for training triplets.
Parameters:
    ap_dists, np array with the distances between each pair anchor-positive
    an_dists, np array with the distances between each pair anchor-negative
Returns:
    opt_beta, optimal threshold
"""
def get_decision_threshold_tree(ap_dists, an_dists):
    # Train data for the Decision Tree
    x_tree = np.concatenate((ap_dists, an_dists))
    y_tree = np.concatenate((np.ones(len(ap_dists)), np.zeros(len(an_dists))))

    # Create Decision Tree with depth=1 to get just one classification rule (the decision threshold)
    decision_tree = DecisionTreeClassifier(max_depth=1, splitter='best')

    # Train Decision Tree
    decision_tree = decision_tree.fit(x_tree.reshape(-1, 1), y_tree)

    # Get classification rule learnt by Tree
    r = export_text(decision_tree)

    # Get optimal beta
    opt_beta = float(r.splitlines()[0].split("|--- feature_0 <= ")[1])

    return opt_beta

"""
###############################################################################
COMPUTE ALL THRESHOLDS

Compute and return all three thresholds at once.
"""

"""
Compute and return all three types of decision thresholds at once:
    TA-TR threshold
    RAAP-RAAN threshold
    Decision Tree threshold
Parameters:
    ap_dists, np array with the distances between each pair anchor-positive
    an_dists, np array with the distances between each pair anchor-negative
Returns:
    np array containing all three thresholds
"""
def get_decision_thresholds(ap_dists, an_dists):
    # TA-TR threshold
    beta_ta_tr, metric_ta_tr, beta_ap_ta, beta_an_tr = get_decision_threshold_ta_tr(ap_dists, an_dists)

    # RAAP-RAAN threshold
    beta_raap_raan, metric_raap_raan, raap, raan = get_decision_threshold_raap_raan(ap_dists, an_dists)

    # Decision Tree threshold
    beta_tree = get_decision_threshold_tree(ap_dists, an_dists)

    return np.array([beta_ta_tr, beta_raap_raan, beta_tree])



"""
Get the optimal TA-TR decision threshold for training pairs of images
in Traditional Neural Network.
The optimal threshold is obtain from training pairs as that
threshold that maximizes the sum of the positive pairs true
accepts (ap_ta) plus the negative pairs true rejects (an_tr).
Parameters:
    pos_train_pred, np array with the distances between each pair anchor-positive
    neg_train_pred, np array with the distances between each pair anchor-negative
Returns:
    optimal_beta, optimal decision threshold
    in a tuple (the three metrics):
        max_metric, anchor-positive true accepts and anchor-negative true rejects
            sum for optimal beta
        opt_ap_ta, anchor-positive true accepts for optimal beta
        opt_an_tr, anchor-negative true rejects for optimal beta
"""
def get_decision_threshold_trad_nn(pos_train_pred, neg_train_pred):
    # Generate all possible thresholds
    possible_beta = np.linspace(0, 1, 100)

    opt_beta   = -1 # Optimal threshold 
    opt_ap_ta  = 0 # Anchor-positive True accepts
    opt_an_tr  = 0 # Anchor-negative true rejects
    max_metric = 0 # Sum of both previous metrics

    # For each possible threshold
    for beta in possible_beta:
        positive_ta = np.where(beta < pos_train_pred)[0].size
        negative_tr = np.where(neg_train_pred < beta)[0].size

        if (positive_ta + negative_tr) > max_metric:
            opt_beta = beta
            max_metric = positive_ta + negative_tr
            opt_ap_ta = positive_ta
            opt_an_tr = negative_tr
    
    return opt_beta, (max_metric, opt_ap_ta, opt_an_tr)




