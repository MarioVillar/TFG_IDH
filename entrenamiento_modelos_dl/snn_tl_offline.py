# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 13:26:19 2022

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
from threshold_detection_functions import *
from utilities import *
from test_functions import *


# Create Keras Model with the SNN to train it on data

class SNNTLOffline(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
    
       L(A, P, N) = max(‖emb(A) - emb(P)‖² - ‖emb(A) - emb(N)‖² + alpha +
                            # (not used finally) lambda_1 * sum_i_d(emb_i(A) + emb_i(P) + emb_i(N)) +             # L1 regularization
                            lambda_2 * (norm_2(A) + norm_2(P) + norm_2(N)),                  # L2 regularization
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
        (not used finally) lambda_1 is the L1 regularization strength parameter,
        lambda_2 is the L2 regularization strength parameter.
    """

    def __init__(self, siamese_network, embedding_gen, face_dataset,
                 alpha, lambda_2=0, alpha_pen=0, epsilon=0, triplet_selection=False): # lambda_1=0, 
        super(SNNTLOffline, self).__init__()
        self.siamese_network = siamese_network  # Underneath network model
        self.embedding_gen = embedding_gen      # Underneath embedding generator model

        self.face_dataset = face_dataset        # Face database for validation

        self.alpha = alpha                      # Distance margin
        self.alpha_pen = float(alpha_pen)       # Conditional penalty
        self.epsilon = float(epsilon)           # epsilon for Conditional Triplet Loss (between 0 and 1)
        # self.lambda_1 = lambda_1                # L1 regularization strength parameter
        self.lambda_2 = lambda_2                # L2 regularization strength parameter

        self.triplet_selection = triplet_selection

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

        # Let's update and return the training loss metric.
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
        distances: tensor of shape [batch_size, batch_size]
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


    def _compute_loss(self, data):
        # The output of the network is a tuple containing the embeddings
        # of the anchor, the positive and the negative images
        anchor_emb, pos_emb, neg_emb = self.siamese_network(data)

        # Compute the square distance between anchor and positive embeddings
        # and between anchor and negative embeddings
        ap_distance = emb_distance(anchor_emb, pos_emb)
        an_distance = emb_distance(anchor_emb, neg_emb)
        

        # Get L1 penalization as sum of the coefficients of the embeddings
        # of the anchor, positive and negative image
        l1_penalization = tf.norm(anchor_emb, ord=1) + tf.norm(pos_emb, ord=1) + \
                          tf.norm(neg_emb, ord=1)

        # # Get L2 penalization as sum of the square coefficients of the embeddings
        # # of the anchor, positive and negative image
        # l2_penalization = tf.norm(anchor_emb, ord=2) + tf.norm(pos_emb, ord=2) + \
        #                   tf.norm(neg_emb, ord=2)

        # Get L2 penalization as sum of l2 norm of the embeddings
        # of the anchor, positive and negative image
        l2_penalization = tf.reduce_mean(tf.norm(anchor_emb, ord=2) + tf.norm(pos_emb, ord=2) + tf.norm(neg_emb, ord=2))

        ###########
        # Traditional Triplet Loss
        # Get Triplet Loss from AP distance, AN distance, alpha, L1 penalty and
        # L2 penalty
        loss = tf.maximum(  ap_distance - an_distance + self.alpha +
                            # self.lambda_1 * l1_penalization +
                            self.lambda_2 * l2_penalization,
                          0.0)
        
        ###########
        # Selection of hard and semi-hard triplets within batch
        if self.triplet_selection:
          cond_semi_hard_1 = tf.math.greater(an_distance, ap_distance) # returns boolean tensor
          cond_semi_hard_2 = tf.math.greater(ap_distance + self.alpha, an_distance) # returns boolean tensor
          cond_semi_hard   = tf.math.logical_and(cond_semi_hard_1, cond_semi_hard_2)

          cond_hard        = tf.math.greater(ap_distance, an_distance) # returns boolean tensor

          cond_hard_and_semi_hard = tf.math.logical_or(cond_hard, cond_semi_hard)

          loss = tf.boolean_mask(loss, cond_hard_and_semi_hard)

        ###########
        # Conditional Triplet Loss https://www.sciencedirect.com/science/article/pii/S0893608021000022
        if self.alpha_pen != 0:
          loss_worst = loss + self.alpha_pen * (ap_distance + an_distance) / 2
          loss_best  = loss - self.alpha_pen * (an_distance - ap_distance) / 2 

          condition_worst = tf.math.equal(ap_distance, an_distance + self.alpha) 

          cond_best_1    = tf.math.greater(ap_distance - an_distance, self.alpha*(self.epsilon - 1)) # returns boolean tensor
          cond_best_2    = tf.math.greater(self.alpha * (2*self.epsilon - 1), ap_distance - an_distance) # returns boolean tensor
          condition_best = tf.math.logical_and(cond_best_1, cond_best_2)

          loss = tf.where(condition_worst, loss_worst, loss)
          loss = tf.where(condition_best, loss_best, loss)

        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker, self.mean_ranking_tracker,
                self.mean_pct_ranking_tracker]
