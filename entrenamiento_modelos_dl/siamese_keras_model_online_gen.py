# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 14:02:49 2022

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
from threshold_detection_functions import *
from utilities import *
from test_functions import *



class SiameseModel(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
    
       L(A, P, N) = max(‖emb(A) - emb(P)‖² - ‖emb(A) - emb(N)‖² + alpha +
                            lambda_1 * sum_i_d(emb_i(A) + emb_i(P) + emb_i(N)) +             # L1 regularization
                            lambda_2 * sum_i_d(emb_i(A)^2 + emb_i(P)^2 + emb_i(N)^2),        # L2 regularization
                        0)
    
    Where:
        A is the anchor image,
        P is the positive image,
        N is the negative image,
        emb is the embedding of a image,
        alpha is the margin that separates the distance between anchor-positive
            and anchor-negative pairs.
        sum_i_d(emb_i(x)) is the sum of all the coefficients of the embedding
            vector of x
        lambda_1 is the L1 regularization strength parameter,
        lambda_2 is the L2 regularization strength parameter.
    """

    def __init__(self, siamese_network, embedding_gen, face_dataset, 
                 alpha, lambda_2=0, batch_all=False):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network  # Underneath network model
        self.embedding_gen = embedding_gen      # Underneath embedding generator model

        self.face_dataset = face_dataset        # Face database for validation

        self.alpha = alpha             # Distance margin
        self.lambda_2 = lambda_2       # L2 regularization strength parameter
        self.batch_all = batch_all     # Whether to bacth all triplets or not (only hard ones)

        self.loss_tracker = metrics.Mean(name="loss")
        self.mean_ranking_tracker = metrics.Mean(name="mean_ranking")
        self.mean_pct_ranking_tracker = metrics.Mean(name="mean_pct_ranking")


    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Update and return the training loss metric and number of triplets metric.
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        (val_images, val_labels) = data

        faces, face_labels = next(iter(self.face_dataset))

        # Obtain embeddings of validation data and face dataset images
        val_emb  = self.embedding_gen(val_images)
        face_emb = self.embedding_gen(faces)

        # Compute distances between each val image and each face image
        pairwise_dists = self._pairwise_distances(val_emb, face_emb)

        # index_min_dist = tf.math.argmin(d, axis=1)
        # predictions = tf.gather(self.face_labels, index_min_dist)

        # Get indexes of sorted distances. sort_pred[i,j] is the ranking
        # of face image j for val image i
        sort_pred = tf.argsort(pairwise_dists)

        # Get matrix of predictions, sorted. all_predictions[i,:] contains
        # all face image labels sorted by ranking predictions for val image i
        all_predictions = tf.gather(face_labels, sort_pred)

        # Expand val_labels to compare to all_predictions
        val_labels_exp = tf.expand_dims(val_labels, 1)

        # Get boolean mask where mask[i,j]=True si val_label[i]==all_predictions[i,j]
        mask = tf.equal(val_labels_exp, all_predictions)

        # Cast boolean to int (True=1, False=0). There will be only one 1 (one True)
        # in each row
        mask = tf.cast(mask, tf.int32)

        # Get index of 1 in each row. rankings[i] is the ranking obtained for
        # val_images[i]
        rankings = tf.math.argmax(mask, axis=1)

        # Get mean ranking
        mean_ranking = tf.reduce_mean(tf.cast(rankings, tf.float32))

        # Get mean pct ranking
        mean_pct_ranking = mean_ranking / tf.cast(tf.shape(faces)[0], tf.float32)

        # Update and return mean ranking
        self.mean_ranking_tracker.update_state(mean_ranking)
        self.mean_pct_ranking_tracker.update_state(mean_pct_ranking)

        return {"mean_ranking": self.mean_ranking_tracker.result(), "mean_pct_ranking": self.mean_pct_ranking_tracker.result()}
    
    """
    Compute the 2D matrix of distances between two vector of embeddings.
    Parameters:
        emb1, tensor of shape [batch_size, embed_dim]
        emb2, tensor of shape [batch_size, embed_dim]
    Returns:
        pairwise_distances: tensor of shape [batch_size, batch_size]
    """
    def _pairwise_distances(self, emb1, emb2):
        # Get difference between coordinates of all embeddings
        difference = tf.expand_dims(emb1, 1) - tf.expand_dims(emb2, 0)

        # Get square difference between coordinates of all embeddings
        square_diff = tf.math.square(difference)

        # Get sum of square differences between coordinates of all embeddings
        sum_sq_diff = tf.math.reduce_sum(square_diff, axis=2)

        # Get L2 distance as sqrt of sum of square differences between coordinates of all embeddings
        distances = tf.math.sqrt(sum_sq_diff)

        return distances
    
    """
    Return a 3D mask where mask[a, p, n] is True if the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - labels_a[i] == labels_p[j] and labels_a[i] != labels_n[k]
    Parameters:
        labels_a, Tf string Tensor with shape [batch_size]
        labels_p, Tf string Tensor with shape [batch_size]
        labels_n, Tf string Tensor with shape [batch_size]
    Returns:
        mask, Tf boolean Tensor with shape [batch_size, batch_size, batch_size]
    """
    def _get_triplet_mask(self, labels_a, labels_p, labels_n):
        # Check if labels_a[i] == labels_p[j] and labels_a[i] != labels_n[k]
        a_equal_p = tf.equal(tf.expand_dims(labels_a, 0), tf.expand_dims(labels_p, 1)) # Where A==P
        a_equal_p = tf.expand_dims(a_equal_p, 2) # Expand dimensions

        a_equal_n = tf.equal(tf.expand_dims(labels_a, 0), tf.expand_dims(labels_n, 1)) # Where A==N
        a_not_equal_n = tf.logical_not(a_equal_n) # Where A!=N
        a_not_equal_n = tf.expand_dims(a_not_equal_n, 1) # Expand dimensions

        mask = tf.logical_and(a_equal_p, a_not_equal_n) # Where A==P and not(A==N)

        return mask
    
    """
    Return a 2D mask where mask[a, p] is True if a and p have same label.
    Parameters:
        labels_a, Tf string Tensor with shape [batch_size]
        labels_p, Tf string Tensor with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    def _get_anchor_positive_triplet_mask(self, labels_a, labels_p):
        # Check if labels_a[i] == labels_p[j]
        # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
        mask = tf.equal(tf.expand_dims(labels_a, 0), tf.expand_dims(labels_p, 1))

        return mask

    
    """
    Return a 2D mask where mask[a, n] is True if a and n have distinct labels.
    Parameters:
        labels_a, Tf string Tensor with shape [batch_size]
        labels_n, Tf string Tensor with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    def _get_anchor_negative_triplet_mask(self, labels_a, labels_n):
        # Check if labels_a[i] != labels_n[k]
        # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
        labels_equal = tf.equal(tf.expand_dims(labels_a, 0), tf.expand_dims(labels_n, 1))

        mask = tf.logical_not(labels_equal)

        return mask
    

    """
    Compute the loss over semi-hard and hard triplets after online generation.
    All the valid triplets are generated, then the loss is averaged
    over the semi-hard and hard ones.
    Parameters:
        ap_distances, distances between each pair of anchor-positive examples in batch
        an_distances, distances between each pair of anchor-negative examples in batch
        labels_a, Tf string Tensor with shape [batch_size]
        labels_p, Tf string Tensor with shape [batch_size]
        labels_n, Tf string Tensor with shape [batch_size]
    Returns:
        triplet_loss, scalar tensor containing the triplet loss value
    """
    def _batch_all_triplet_loss(self, an_emb, pos_emb, neg_emb,
                                labels_a, labels_p, labels_n):
        an_emb = l2_unit_vector(an_emb)
        pos_emb = l2_unit_vector(pos_emb)
        neg_emb = l2_unit_vector(neg_emb)

        # Get the pairwise distance matrixes
        anchor_positive_dist = self._pairwise_distances(an_emb, pos_emb) # Anchor-positive distances
        anchor_negative_dist = self._pairwise_distances(an_emb, neg_emb) # Anchor-negative distances

        # Expand dimensions in order to get all the combinations of
        # possible triplets and calculate triplet loss for each one
        anchor_positive_dist_exp = tf.expand_dims(anchor_positive_dist, 2)
        anchor_negative_dist_exp = tf.expand_dims(anchor_negative_dist, 1)

        # Compute a Triplet Loss for all possible triplets (including not valid ones)
        # 3D tensor of size (batch_size, batch_size, batch_size)
        # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
        # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
        # and the 2nd (batch_size, 1, batch_size)
        triplet_loss = anchor_positive_dist_exp - anchor_negative_dist_exp + self.alpha

        # Put to zero the loss of the invalid triplets
        mask = self._get_triplet_mask(labels_a, labels_p, labels_n)
        mask = tf.cast(mask, tf.float32)
        triplet_loss = tf.multiply(mask, triplet_loss)

        # Remove negative losses (easy triplets)
        triplet_loss = tf.maximum(triplet_loss, 0.0)

        # Count number of positive (useful) triplets (where triplet_loss > 0)
        valid_triplets = tf.cast(tf.greater(triplet_loss, 1e-16), tf.float32) # Float Mask: 1 for positive triplets and 0 for not positive
        num_positive_triplets = tf.reduce_sum(valid_triplets)
        
        # # Total number of valid triplets
        # num_valid_triplets = tf.reduce_sum(mask)
        # # Ratio of positive triplets in relation to valid triplets
        # fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

        # Get mean Triplet Loss over the positive (and valid) triplets
        triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

        return triplet_loss
    
    
    """
    Compute the loss over the hardest triplets in batch after online generation.
    All the valid triplets are generated, then the loss is averaged
    over the hardest ones.
    Parameters:
        ap_distances, distances between each pair of anchor-positive examples in batch
        an_distances, distances between each pair of anchor-negative examples in batch
        labels_a, Tf string Tensor with shape [batch_size]
        labels_p, Tf string Tensor with shape [batch_size]
        labels_n, Tf string Tensor with shape [batch_size]
    Returns:
        triplet_loss, scalar tensor containing the triplet loss value
    """
    def _batch_hard_triplet_loss(self, an_emb, pos_emb, neg_emb,
                                 labels_a, labels_p, labels_n):
        # Get the pairwise distance matrixes
        anchor_positive_dist = self._pairwise_distances(an_emb, pos_emb) # Anchor-positive distances
        anchor_negative_dist = self._pairwise_distances(an_emb, neg_emb) # Anchor-negative distances

        # For each anchor, get the hardest positive
        # First, we need to get a mask for every valid positive (they should have same label)
        mask_anchor_positive = self._get_anchor_positive_triplet_mask(labels_a, labels_p)
        mask_anchor_positive = tf.cast(mask_anchor_positive, tf.float32)

        # We put to 0 any element where (a, p) is not valid (valid if label(a) == label(p))
        anchor_positive_dist = tf.multiply(mask_anchor_positive, anchor_positive_dist)

        # shape (batch_size, 1)
        hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)


        # For each anchor, get the hardest negative
        # First, we need to get a mask for every valid negative (they should have different labels)
        mask_anchor_negative = self._get_anchor_negative_triplet_mask(labels_a, labels_n)
        mask_anchor_negative = tf.cast(mask_anchor_negative, tf.float32)

        # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
        max_anchor_negative_dist = tf.reduce_max(anchor_negative_dist, axis=1, keepdims=True)
        anchor_negative_dist = anchor_negative_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

        # shape (batch_size,)
        hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)


        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + self.alpha, 0.0)

        # Get final mean triplet loss
        triplet_loss = tf.reduce_mean(triplet_loss)

        return triplet_loss

        
    def _compute_loss(self, data):
        # Unzip triplets and labels 
        (data, labels) = data

        # Unzip anchor, positive and negative labels
        (labels_a, labels_p, labels_n) = labels
        
        # The output of the network is a tuple containing the embeddings
        # of the anchor, the positive and the negative images
        anchor_emb, pos_emb, neg_emb = self.siamese_network(data)
        

        # Batch-all Triplet Loss
        loss = 0
        
        if self.batch_all:
            loss = self._batch_all_triplet_loss(anchor_emb, pos_emb, neg_emb,
                                                labels_a, labels_p, labels_n)
        else:
            loss = self._batch_hard_triplet_loss(anchor_emb, pos_emb, neg_emb,
                                                 labels_a, labels_p, labels_n)
        
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker, self.mean_ranking_tracker,
                self.mean_pct_ranking_tracker]
