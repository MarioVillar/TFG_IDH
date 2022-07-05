# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 19:41:13 2022

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



class SNNSIGMOID(Model):

    def __init__(self, network, face_dataset):
        super(SNNSIGMOID, self).__init__()
        self.network = network
        self.face_dataset = face_dataset        # Face database for validation
        
        self.loss_tracker = metrics.Mean(name="loss")
        self.mean_ranking_tracker = metrics.Mean(name="mean_ranking")
        self.mean_pct_ranking_tracker = metrics.Mean(name="mean_pct_ranking")
        
        self.topk_acc_tracker = MaxTopK(name="topk_acc")
            
    def call(self, inputs):
        return self.network(inputs)
    
    def compile(self, optimizer, loss, sqrt=False):
        super(SNNSIGMOID, self).compile()
        self.nn_optimizer = optimizer
        self.loss_fn = loss
        self.sqrt = sqrt
        
    def train_step(self, data):
        # Unpack the data.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self.network(x, training=True)  # Forward pass
        
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.loss_fn(y, y_pred)#, regularization_losses=self.losses)
            
            if self.sqrt:
                loss = tf.math.sqrt(loss)
            
        # Compute gradients
        gradients = tape.gradient(loss, self.network.trainable_weights)
        
        # Update weights
        # Applying the gradients on the model using the specified optimizer
        self.nn_optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))
        
        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
    
    def test_step(self, data):
        val_images, val_labels = data

        faces, face_labels = next(iter(self.face_dataset))
        
        num_val = tf.shape(val_images)[0]
        num_faces = tf.shape(faces)[0]

        # Obtain all combinations between val images and face images
        # Repeats each val image num_faces times
        val_images_combined = tf.repeat(val_images, num_faces, axis=0)
        # Repeats the whole face set num_val times 
        face_combined = tf.tile(faces, [num_val,1,1,1])

        # Compute predictions
        # y_pred[i] is the prediction for val_images[i % num_val] with
        # faces[i // num_val]
        y_pred = self.network((val_images_combined, face_combined), training=False)
        
        # Reshape predictions to get matrix
        # Now y_pred[i,j] is the prediction of val_images[i] with faces[j]
        y_pred = tf.transpose( tf.reshape(y_pred, (num_faces, num_val)) )

        # # To return the predicted label
        # index_min_dist = tf.math.argmax(y_pred, axis=1)
        # predictions = tf.gather(face_labels, index_min_dist)

        # Get indexes of sorted distances. sort_pred[i,j] is the ranking
        # of face image j for val image i
        sort_pred = tf.argsort(y_pred, direction='DESCENDING', axis=1)

        # Get matrix of predictions, sorted. 
        # all_predictions[i,:] are the predicted labels for val_image[i], 
        #   sorted by the similarity produced by the model
        # all_predictions[i,0] is the predicted label for val_image[i]
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
        
        # Get top-k acc
        top_k_acc = tf.reduce_max(rankings)
        
        # Update and return mean ranking
        self.mean_ranking_tracker.update_state(mean_ranking)
        self.mean_pct_ranking_tracker.update_state(mean_pct_ranking)
        
        self.topk_acc_tracker.update_state(top_k_acc)
        
        
        return {"mean_ranking": self.mean_ranking_tracker.result(),
                "mean_pct_ranking": self.mean_pct_ranking_tracker.result(),
                "topk_acc": self.topk_acc_tracker.result()}

    def get_cmc_values(self, test_dataset, threshold):
        cmc_values = np.empty((0,1), dtype=np.int32)
        
        faces, face_labels = next(iter(self.face_dataset))
        num_faces = tf.shape(faces)[0]
        
        for test_images, test_labels in test_dataset:
            num_test = tf.shape(test_images)[0]
    
            # Obtain all combinations between test images and face images
            # Repeats each test image num_faces times
            test_images_combined = tf.repeat(test_images, num_faces, axis=0)
            # Repeats the whole face set num_test times 
            face_combined = tf.tile(faces, [num_test,1,1,1])
    
            # Compute predictions (values between 0 and 1)
            # y_pred[i] is the prediction for test_images[i % num_test] with
            # faces[i // num_test]
            y_pred = self.network((test_images_combined, face_combined), training=False)
            
            # Reshape predictions to get matrix
            # Now y_pred[i,j] is the prediction of test_images[i] with faces[j]
            y_pred = tf.transpose( tf.reshape(y_pred, (num_faces, num_test)) )
    
            # # To return the predicted label
            # index_min_dist = tf.math.argmax(y_pred, axis=1)
            # predictions = tf.gather(face_labels, index_min_dist)
    
            # Get indexes of sorted distances. sort_pred[i,j] is the ranking
            # of face image j for test image i
            sort_pred = tf.argsort(y_pred, direction='DESCENDING', axis=1)
    
            # Get matrix of predictions, sorted. 
            # all_predictions[i,:] are the predicted labels for test_image[i], 
            #   sorted by the similarity produced by the model
            # all_predictions[i,0] is the predicted label for test_image[i]
            all_predictions = tf.gather(face_labels, sort_pred)
    
            # Expand test_labels to compare to all_predictions
            test_labels_exp = tf.expand_dims(test_labels, 1)
    
            # Get boolean mask where mask[i,j]=True si test_label[i]==all_predictions[i,j]
            mask = tf.equal(test_labels_exp, all_predictions)
    
            # Cast boolean to int (True=1, False=0). There will be only one 1 (one True)
            # in each row
            mask = tf.cast(mask, tf.int32)
    
            # Get index of the 1 in each row. rankings[i] is the ranking 
            # obtained for test_images[i]
            rankings_batch = tf.math.argmax(mask, axis=1)
            
            ################
            # Positive images can only be matched if the predicted value for
            # the real image is greater than threshold. If this is not the
            # case, ranking should be increased by one.
            
            # Get tensor with row indexes in y_pred tensor
            rows = tf.range(0, num_test, dtype=tf.int64)
            
            # Expand test_labels to compare to all_predictions
            test_labels_exp = tf.expand_dims(test_labels, 1)
            
            # Get boolean mask where mask[i,j]=True si test_label[i]==face_label[j]
            mask = tf.equal(face_labels, test_labels_exp)
            
            # Cast boolean to int (True=1, False=0). There will be only one 1 (one True)
            # in each row
            mask = tf.cast(mask, tf.int32)
            
            # Get position of the real face image in y_pred tensor for each test image
            pos_real_face = tf.math.argmax(mask, axis=1)
            
            # Get pairs of indexes [row,col] of the real faces predictions
            index_pred_real = tf.concat([tf.expand_dims(rows,1), tf.expand_dims(pos_real_face,1)], axis=1)
            
            # Get prediction values for each image
            pred_made_on_real = tf.gather_nd(y_pred, index_pred_real)
                        
            # Get boolean mask: True where predicted value is lower than threshold
            pred_less_threshold = tf.math.less(pred_made_on_real, threshold)
            
            # Update rankings to get cmc_values
            cmc_values_batch = rankings_batch.numpy()
            cmc_values_batch[pred_less_threshold.numpy()] += 1            
            
            # Append to cmc_values of all test dataset
            cmc_values = np.append(cmc_values, cmc_values_batch)
        
        return cmc_values
    
    
    def get_nfa_values(self, test_dataset, threshold):
        nfa_values = np.empty((0,1), dtype=np.int32)
        
        faces, face_labels = next(iter(self.face_dataset))
        num_faces = tf.shape(faces)[0]
        
        for test_images in test_dataset:
            num_test = tf.shape(test_images)[0]
    
            # Obtain all combinations between test images and face images
            # Repeats each test image num_faces times
            test_images_combined = tf.repeat(test_images, num_faces, axis=0)
            # Repeats the whole face set num_test times 
            face_combined = tf.tile(faces, [num_test,1,1,1])
    
            # Compute predictions (values between 0 and 1)
            # y_pred[i] is the prediction for test_images[i % num_test] with
            # faces[i // num_test]
            y_pred = self.network((test_images_combined, face_combined), training=False)
            
            # Reshape predictions to get matrix
            # Now y_pred[i,j] is the prediction of test_images[i] with faces[j]
            y_pred = tf.transpose( tf.reshape(y_pred, (num_faces, num_test)) )
    
            # # To return the predicted label
            # index_min_dist = tf.math.argmax(y_pred, axis=1)
            # predictions = tf.gather(face_labels, index_min_dist)
            
            ################
            # Negative images should not be matched to any image. To achieve
            # this, all predicted values should be lower than threshold.
            # NFA is the Number of False Accepts, number of faces with
            # greater predicted value than threshold (for negative images).
            
            # Get boolean mask: True where predicted value is greater than threshold
            pred_greater_threshold = tf.math.greater(y_pred, threshold)
            
            # Conver boolean mask to int. Trues became 1 and Falses became 0
            mask = tf.cast(pred_greater_threshold, tf.int32)
            
            # Count number of 1 in each row (= NFA in each row)
            nfa_values_batch = tf.reduce_sum(mask, axis=1).numpy()
            
            # Append to nfa_values of all test dataset
            nfa_values = np.append(nfa_values, nfa_values_batch)
        
        return nfa_values
    

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        return [self.loss_tracker,
                self.mean_ranking_tracker,
                self.mean_pct_ranking_tracker,
                self.topk_acc_tracker]







class MaxTopK(metrics.Metric):
    
    def __init__(self, name='maxTopK', **kwargs):
        super(MaxTopK, self).__init__(name=name, **kwargs)
        self.topk = self.add_weight(name='topk', initializer='zeros', dtype=tf.int64)
    
    def update_state(self, new_topk):
        combination = tf.stack([self.topk, new_topk])
        
        self.topk.assign( tf.reduce_max(combination))
    
    def result(self):
        return self.topk

    













