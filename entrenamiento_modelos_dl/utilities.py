# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 12:32:25 2022

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


from image_preprocessing_functions import *
from distance_functions import *




###############################################################################
###############################################################################
###############################################################################
# DEFINITION OF CLASS TO SAVE RESULTS IN DISK IN SIAMESE NETWORK

class expResults():
    def __init__(self, pos_ind, neg_ind, face_db_size):
        # Current number of models saved
        self.nmodels = 0

        # Number of faces in DB
        self.face_sample_size = face_db_size + 1 # Add no match prediction possibility

        # Number of individuals with cranium model
        self.pos_individuals = pos_ind

        # Number of individuals without cranium model
        self.neg_individuals = neg_ind

        # Create list with (alpha, l1 penalization, L2 penalization, alpha_penalty,
        #                       epsilon, train_epochs, learn rate, triplet_selection) for each model
        self.parameters = [] # Empty at first
        
        # Create NP array with thresholds: (ta_tr, raap_raan, tree) for each model
        self.thresholds = np.empty((0,3)) # Empty at first

        # List with CMC values obtained with TA-TR method
        self.cmc_values_ta_tr = np.empty((0,pos_ind)) # Empty at first

        # List with NFA values obtained with TA-TR method
        self.nfa_values_ta_tr = np.empty((0,neg_ind)) # Empty at first

        # List with CMC values obtained with RAAP-RAAN method
        self.cmc_values_raap_raan = np.empty((0,pos_ind)) # Empty at first

        # List with NFA values obtained with RAAP-RAAN method
        self.nfa_values_raap_raan = np.empty((0,neg_ind)) # Empty at first

        # List with CMC values obtained with Tree Decision method
        self.cmc_values_tree = np.empty((0,pos_ind)) # Empty at first

        # List with NFA values obtained with Tree Decision method
        self.nfa_values_tree = np.empty((0,neg_ind)) # Empty at first

        # Create NP array with accuracies: (pos_acc, neg_acc, overall_acc) for each model obtained with TA-TR method
        self.accuracies_ta_tr = np.empty((0,3)) # Empty at first

        # Create NP array with accuracies: (pos_acc, neg_acc, overall_acc) for each model obtained with RAAP-RAAN method
        self.accuracies_raap_raan = np.empty((0,3)) # Empty at first

        # Create NP array with accuracies: (pos_acc, neg_acc, overall_acc) for each model obtained with Tree method
        self.accuracies_tree = np.empty((0,3)) # Empty at first
        
        # Create NP array with first top-k accuracy=1.0: (ta_tr, raap_raan, tree) for each model
        self.first_top_k_acc = np.empty((0,3)) # Empty at first
        
        # Create list with np arrays containing validation metric history for each model
        self.val_ranking_pct_history = [] # Empty at first
    
    
    def append_model(self, param, threshold_array,
                     cmc_val_ta_tr, nfa_val_ta_tr,
                     cmc_val_raap_raan, nfa_val_raap_raan,
                     cmc_val_tree, nfa_val_tree,
                     acc_ta_tr, acc_raap_raan, acc_tree,
                     first_top_k,
                     val_ranking_hist):
        # Save model parameters
        self.parameters.append(param) # Param should be a list
        
        # Save model decision thresholds
        self.thresholds = np.append(self.thresholds, 
                                    self.format_array(threshold_array), axis=0)

        # Save cmc values obtained with TA-TR method
        self.cmc_values_ta_tr = np.append(self.cmc_values_ta_tr, 
                                          self.format_array(cmc_val_ta_tr), axis=0)

        # Save nfa values obtained with TA-TR method
        self.nfa_values_ta_tr = np.append(self.nfa_values_ta_tr,
                                          self.format_array(nfa_val_ta_tr), axis=0)

        # Save cmc values obtained with RAAP-RAAN method
        self.cmc_values_raap_raan = np.append(self.cmc_values_raap_raan,
                                              self.format_array(cmc_val_raap_raan), axis=0)

        # Save nfa values obtained with RAAP-RAAN method
        self.nfa_values_raap_raan = np.append(self.nfa_values_raap_raan, 
                                              self.format_array(nfa_val_raap_raan), axis=0)

        # Save cmc values obtained with Tree Decision method
        self.cmc_values_tree = np.append(self.cmc_values_tree,
                                         self.format_array(cmc_val_tree), axis=0)

        # Save nfa values obtained with Tree Decision method
        self.nfa_values_tree = np.append(self.nfa_values_tree,
                                         self.format_array(nfa_val_tree), axis=0)

        # Save model accuracies (obtained with TA-TR method)
        self.accuracies_ta_tr = np.append(self.accuracies_ta_tr, 
                                          self.format_array(acc_ta_tr), axis=0)

        # Save model accuracies (obtained with RAAP-RAAN method)
        self.accuracies_raap_raan = np.append(self.accuracies_raap_raan,
                                              self.format_array(acc_raap_raan), axis=0)

        # Save model accuracies (obtained with Tree method)
        self.accuracies_tree = np.append(self.accuracies_tree, 
                                         self.format_array(acc_tree), axis=0)
        
        self.first_top_k_acc = np.append(self.first_top_k_acc, 
                                         self.format_array(first_top_k), axis=0)
        
        # Save model validaiton metric history
        self.val_ranking_pct_history.append(val_ranking_hist) # Param should be a list
        
        
        # Increase by one the model count
        self.nmodels += 1
    
    def format_array(self, array):
        return np.expand_dims(array, 0)
    
    def get_model(self, index_model):
        data = {}

        if index_model < self.nmodels:
            data['parameters'] = self.parameters[index_model]
            
            data['thresholds'] = self.thresholds[index_model]
            
            data['cmc_ta_tr'] = self.cmc_values_ta_tr[index_model]
            data['nfa_ta_tr'] = self.nfa_values_ta_tr[index_model]

            data['cmc_raap_raan'] = self.cmc_values_raap_raan[index_model]
            data['nfa_raap_raan'] = self.nfa_values_raap_raan[index_model]

            data['cmc_tree'] = self.cmc_values_tree[index_model]
            data['nfa_tree'] = self.nfa_values_tree[index_model]

            data['acc_ta_tr'] = self.accuracies_ta_tr[index_model]
            data['acc_raap_raan'] = self.accuracies_raap_raan[index_model]
            data['acc_tree'] = self.accuracies_tree[index_model]
            
            data['first_top_k'] = self.first_top_k_acc[index_model]
            
            data['val_history'] = self.val_ranking_pct_history[index_model]
        else:
            data = None
        
        return data
    
    def get_n_best_acc_models(self, n_best):
        # Attending overall accuracy
        return np.argpartition(self.accuracies_tree[:,2],
                               len(self.accuracies_tree[:,2]) - n_best)[-n_best:]
    
    def get_n_best_pos_acc_models(self, n_best):
        # Attending overall accuracy
        return np.argsort(self.accuracies_tree[:,0])[-n_best:]
    
    def get_n_best_neg_acc_models(self, n_best):
        # Attending overall accuracy
        return np.argsort(self.accuracies_tree[:,1])[-n_best:]
    
    
    def get_n_best_topk_acc_models(self, n_best):
        # Attending overall accuracy
        return np.argpartition(self.first_top_k_acc[:,2], n_best)[:n_best]

    
    def save_results_to_disk(expResults_object, filename):
        # Open file in write and binary mode
        with open(filename, 'wb') as output:
            # Save object is disk file
            pickle.dump(expResults_object, output, pickle.HIGHEST_PROTOCOL)
    
    def get_results_from_disk(filename, pos_ind=None, neg_ind=None, face_db_size=None):
        object_return = ()
        
        try:
            with open(filename, 'rb') as file:
                try:
                    object_return = pickle.load(file)
                except:
                    print("Could not open file to load expResults object.")
                    if pos_ind is None or neg_ind is None or face_db_size is None:
                        raise Exception("Could not create expResults object") 
                    else:
                        object_return = expResults(pos_ind, neg_ind, face_db_size)
                        print("Returning empty expResults object.")
        except:
            print("Could not open file to load expResults object.")
            if pos_ind is None or neg_ind is None or face_db_size is None:
                raise Exception("Could not create expResults object") 
            else:
                object_return = expResults(pos_ind, neg_ind, face_db_size)
                print("Returning empty expResults object.")
        
        return object_return



###############################################################################
###############################################################################
###############################################################################
# DEFINITION OF CLASS TO SAVE RESULTS IN DISK IN TRADITIONAL NETWORK

class expResultsTrad():
    def __init__(self, pos_ind, neg_ind, face_db_size):
        # Current number of models saved
        self.nmodels = 0

        # Number of faces in DB
        self.face_sample_size = face_db_size + 1 # Add no match prediction possibility

        # Number of individuals with cranium model
        self.pos_individuals = pos_ind

        # Number of individuals without cranium model
        self.neg_individuals = neg_ind

        # Create list with (train_epochs, learn rate, batch_size) for each model
        self.parameters = [] # Empty at first
        
        # Create NP array with the threshold for each model
        self.thresholds = np.empty((0,)) # Empty at first

        # List with CMC values 
        self.cmc_values = np.empty((0,pos_ind)) # Empty at first

        # List with NFA values 
        self.nfa_values = np.empty((0,neg_ind)) # Empty at first

        # Create NP array with accuracies: (pos_acc, neg_acc, overall_acc) for each model
        self.accuracies = np.empty((0,3)) # Empty at first

        # Create NP array with first top-k accuracy=1.0 for each model
        self.first_top_k_acc = np.empty((0,)) # Empty at first
        
        # Create list with np arrays containing validation metric history for each model
        self.val_ranking_pct_history = [] # Empty at first
    
    
    def append_model(self, param, _threshold, cmc_val, nfa_val,
                     acc, first_top_k, val_ranking_hist):
        # Save model parameters
        self.parameters.append(param) # Param should be a list
        
        # Save model decision thresholds
        self.thresholds = np.append(self.thresholds, _threshold)

        # Save cmc values obtained 
        self.cmc_values = np.append(self.cmc_values, self.format_array(cmc_val), axis=0)

        # Save nfa values obtained 
        self.nfa_values = np.append(self.nfa_values, self.format_array(nfa_val), axis=0)

        # Save model accuracies 
        self.accuracies= np.append(self.accuracies, self.format_array(acc), axis=0)
        
        # Save top k acc
        self.first_top_k_acc = np.append(self.first_top_k_acc, first_top_k)
        
        # Save model validaiton metric history
        self.val_ranking_pct_history.append(val_ranking_hist) # Param should be a list
        
        # Increase by one the model count
        self.nmodels += 1
    
    
    def format_array(self, array):
        return np.expand_dims(array, 0)
    
    
    def get_model(self, index_model):
        data = {}

        if index_model < self.nmodels:
            data['parameters'] = self.parameters[index_model]
            
            data['threshold'] = self.thresholds[index_model]
            
            data['cmc_values'] = self.cmc_values[index_model]
            data['nfa_values'] = self.nfa_values[index_model]

            data['accuracies'] = self.accuracies[index_model]
            
            data['first_top_k'] = self.first_top_k_acc[index_model]
            
            data['val_history'] = self.val_ranking_pct_history[index_model]
        else:
            data = None
        
        return data
    
    
    def get_n_best_acc_models(self, n_best):
        # Attending overall accuracy
        return np.argsort(self.accuracies[:,2])[-n_best:]
    
    
    def get_n_best_topk_acc_models(self, n_best):
        # Attending overall accuracy
        return np.argsort(self.first_top_k_acc)[0:n_best]
    
    def get_n_best_pos_acc_models(self, n_best):
        # Attending overall accuracy
        return np.argsort(self.accuracies[:,0])[-n_best:]
    
    def get_n_best_neg_acc_models(self, n_best):
        # Attending overall accuracy
        return np.argsort(self.accuracies[:,1])[-n_best:]
    
    def save_results_to_disk(expResults_object, filename):
        # Open file in write and binary mode
        with open(filename, 'wb') as output:
            # Save object is disk file
            pickle.dump(expResults_object, output, pickle.HIGHEST_PROTOCOL)
    
    def get_results_from_disk(filename, pos_ind=None, neg_ind=None, face_db_size=None):
        object_return = ()
        
        try:
            with open(filename, 'rb') as file:
                try:
                    object_return = pickle.load(file)
                except:
                    print("Could not open file to load expResultsTrad object.")
                    if pos_ind is None or neg_ind is None or face_db_size is None:
                        raise Exception("Could not create expResultsTrad object") 
                    else:
                        object_return = expResultsTrad(pos_ind, neg_ind, face_db_size)
                        print("Returning empty expResultsTrad object.")
        except:
            print("Could not open file to load expResultsTrad object.")
            if pos_ind is None or neg_ind is None or face_db_size is None:
                raise Exception("Could not create expResultsTrad object") 
            else:
                object_return = expResultsTrad(pos_ind, neg_ind, face_db_size)
                print("Returning empty expResultsTrad object.")
        
        return object_return



###############################################################################
# Visualize training images

def show(ax, image):
    ax.imshow(image)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

"""
Visualize three triplets from a batch of triplets.
"""
def visualize_batch(anchor, positive, negative, n_examples=3, 
                    fig_size=(10,10), path_savefig=None):
    fig, axes = plt.subplots(nrows=n_examples, ncols=3, figsize=fig_size)

    fnt_size = 20

    axes[0][0].set_title("Face image", fontsize=fnt_size)
    axes[0][1].set_title("Positive skull image", fontsize=fnt_size)
    axes[0][2].set_title("Negative skull image", fontsize=fnt_size)

    for i in range(n_examples):
        show(axes[i, 0], anchor[i])
        show(axes[i, 1], positive[i])
        show(axes[i, 2], negative[i])
    
    fig.tight_layout()
    
    if path_savefig != None:
        plt.savefig(path_savefig, dpi=300, bbox_inches = "tight")
    

###############################################################################
# Visualize face augmentation images

def visualize_augmentation(faces, n_examples=3, 
                           fig_size=(10,10), path_savefig=None):
    fig, axes = plt.subplots(nrows=n_examples, ncols=3, figsize=fig_size)

    fnt_size = 20

    axes[0][0].set_title("Original face\nimage", fontsize=fnt_size)
    axes[0][1].set_title("Augmented face\nimage (1)", fontsize=fnt_size)
    axes[0][2].set_title("Augmented face\nimage (2)", fontsize=fnt_size)

    for i in range(n_examples):
        show(axes[i, 0], faces[i])
        show(axes[i, 1], data_augmentation_faces(faces[i]))
        show(axes[i, 2], data_augmentation_faces(faces[i]))
    
    fig.tight_layout()
    
    if path_savefig != None:
        plt.savefig(path_savefig, dpi=300, bbox_inches = "tight")


"""
Visualize pairs from a batch of pairs in TradNN.
"""
def visualize_pairs_trad_nn(face_pos, skull_pos, face_neg, skull_neg,
                            n_examples=2, fig_size=(10,10), path_savefig=None):    
    fig, axes = plt.subplots(nrows=n_examples, ncols=2, figsize=fig_size)

    fnt_size = 33

    axes[0][0].set_title("Face image", fontsize=fnt_size)
    axes[0][1].set_title("Skull image", fontsize=fnt_size)

    show(axes[0, 0],  face_pos[0])
    show(axes[0, 1], skull_pos[0])
    
    show(axes[1, 0],  face_neg[0])
    show(axes[1, 1], skull_neg[0])
    
    fig.tight_layout()
    
    if path_savefig != None:
        plt.savefig(path_savefig, dpi=300, bbox_inches = "tight")


"""
###############################################################################
PLOT HISTOGRAM OF DISTANCES (AFTER TRAINING)

The histogram bars identify anchor-positive and anchor-negative distances as
red and black, respectively.
"""

"""
Parameters:
    ap_dists, np array with the distances between each pair anchor-positive
    an_dists, np array with the distances between each pair anchor-negative
    n_bars, number of bars for each distance array
    ap_color, bar color for anchor-positive distances
    an_color, bar color for anchro-negative distances
    fig_size, figure size when showing
    path_savefig, path to save figure to disk. If not specified the image is not saved
"""
def plot_distances_hist(ap_dists, an_dists, n_bars=10, ap_color='red', an_color='black',
                        fig_size=(15,7), path_savefig=None, title=None):
    # Get mean distance between anchor-positive and anchor-negative
    ap_dist_mean = ap_dists.mean() # anchor-positive
    an_dist_mean = an_dists.mean() # anchor-negative

    # Create figure
    fig = plt.figure(figsize=fig_size)

    # Display histograms
    n, bins, patches = plt.hist((ap_dists,an_dists), bins=n_bars, color=(ap_color,an_color))

    # Display vertical lines for mean distances
    plt.axvline(x=ap_dist_mean, color='darkred', linestyle='dashed', linewidth=2)
    plt.axvline(x=an_dist_mean, color='black', linestyle='dashed', linewidth=2)

    # Get x-axis ticks to show labels in x-axis
    ticks = np.linspace(0, math.ceil(bins.max()), n_bars+1)
    
    # Create legend for graph
    legend_elements = [pat.Rectangle((0,0),1,1,fc=ap_color),
                       pat.Rectangle((0,0),1,1,fc=an_color),
                       Line2D([0], [0], color="darkred", ls='dashed', lw=2),
                       Line2D([0], [0], color="black", ls='dashed', lw=2)]
    legend_labels   = ["Anchor-positive pairs",
                       "Anchor-negative pairs",
                       "Anchor-positive mean distance",
                       "Anchor-negative mean distance"]
    plt.legend(legend_elements, legend_labels, loc='upper right', fontsize=15)

    # Set title, X label, Y label and X ticks labels
    plt.title("Anchor-positive and anchor-negative distances", fontsize=20)
    if title != None:
        plt.title(title)
    plt.xlabel("L2 distance between pairs of images", fontsize=15)
    plt.ylabel("Number of pairs", fontsize=15)
    plt.xticks(ticks, rotation=-45)

    if path_savefig != None:
        plt.savefig(path_savefig, dpi=300, bbox_inches = "tight")



"""
###############################################################################
PLOT DECISION THRESHOLDS (AFTER TRAINING)
"""

"""
Parameters:
    ap_dists, np array with the distances between each pair anchor-positive
    an_dists, np array with the distances between each pair anchor-negative
    ta_tr_beta, TA-TR algorithm decision threshold
    raap_raan_beta, RAAP-RAAN algorithm decision threshold
    tree_beta, Decision Tree decision threshold
    n_bars, number of bars for each distance array
    ap_color, bar color for anchor-positive distances
    an_color, bar color for anchro-negative distances
    fig_size, figure size when showing
    path_savefig, path to save figure to disk. If not specified the image is not saved
"""
def plot_dist_thresholds_hist(ap_dists, an_dists, ta_tr_beta, raap_raan_beta, tree_beta,
                              n_bars=10, ap_color='red', an_color='black',
                              fig_size=(15,7), path_savefig=None, title=None):
    # Create figure
    fig = plt.figure(figsize=fig_size)

    # Display histograms
    n, bins, patches = plt.hist((ap_dists,an_dists), bins=n_bars, color=(ap_color,an_color))

    # Display vertical lines for mean distances
    plt.axvline(x=ta_tr_beta, color='blueviolet', linestyle='dashed', linewidth=2)
    plt.axvline(x=raap_raan_beta, color='darkgreen', linestyle='dashed', linewidth=2)
    plt.axvline(x=tree_beta, color='aqua', linestyle='dashed', linewidth=2)

    # Get x-axis ticks to show labels in x-axis
    ticks = np.linspace(0, math.ceil(bins.max()), n_bars+1)
    
    # Create legend for graph
    legend_elements = [pat.Rectangle((0,0),1,1,fc=ap_color),
                       pat.Rectangle((0,0),1,1,fc=an_color),
                       Line2D([0], [0], color="blueviolet", ls='dashed', lw=2),
                       Line2D([0], [0], color="darkgreen", ls='dashed', lw=2),
                       Line2D([0], [0], color="aqua", ls='dashed', lw=2)]
    legend_labels   = ["Anchor-positive pairs",
                       "Anchor-negative pairs",
                       "TA-TR decision threshold",
                       "RAAP-RAAN decision threshold",
                       "Decision Tree decision threshold"]
    plt.legend(legend_elements, legend_labels, loc='upper right', fontsize=15)

    # Set title, X label, Y label and X ticks labels
    plt.title("Anchor-positive and anchor-negative distances", fontsize=20)
    if title != None:
        plt.title(title)
        
    plt.xlabel("L2 distance between pairs of images", fontsize=15)
    plt.ylabel("Number of pairs", fontsize=15)
    plt.xticks(ticks, rotation=-45)

    if path_savefig != None:
        plt.savefig(path_savefig, dpi=300, bbox_inches = "tight")
    
    

"""
###############################################################################
PLOT CMC CURVE
"""

def show_cmc_curve(cmc_values, sample_size, path_savefig=None, title=None):
    x_values = np.array([int(sample_size*i/10) for i in range(1,11)]) # Every 10% of Data base size
    
    # Append first top-k acc = 1.0
    x_values = np.sort(np.append(max(cmc_values), x_values))
    
    # Append top ranking if not already included
    if 1 not in x_values:
        x_values = np.insert(x_values, 0, 1)

    y_values = np.ones(len(x_values), dtype='float32')

    for i in range(0, len(x_values)):
        k = x_values[i] # Top ranking

        y_values[i] = np.count_nonzero(cmc_values <= k) / len(cmc_values) # Top-k accuracy
    

    x_labels = np.array(["1\n(" + "{:.2f}".format(1/sample_size*100) + "%)"] +
                        [str(int(sample_size*i/10)) + "\n(" + str(i*10)+"%)" for i in range(1,11)])
    
    y_ticks  = np.arange(0,1.1,0.1)

    fig = plt.figure(figsize=(10,5))

    plt.plot(x_values, y_values, 'bo-')

    plt.ylim([0, 1.1])
    plt.xlim([0, sample_size*1.1])
    plt.xticks(x_values, x_labels)
    plt.yticks(y_ticks)
    plt.title("CMC curve in test", fontsize=20)
    if title != None:
        plt.title("CMC curve in test " + title)
    plt.xlabel("K rankings\n(Percentaje in sample)", fontsize=15)
    plt.ylabel("Top-K accuracy", fontsize=15)

    plt.grid()
    if path_savefig != None:
        plt.savefig(path_savefig, dpi=300, bbox_inches = "tight")
    


"""
###############################################################################
PLOT NFAS CURVE
"""

def show_nfa_curve(nfa_values, sample_size, path_savefig=None, title=None):
    # X-axis are the NFAs
    x_values = np.array([int(sample_size*i/10) for i in range(0,11)]) # Every 10% of sample size

    # Y-Axis are the percentages of individuals with lower than specfific NFA
    y_values = np.ones(len(x_values), dtype='float32')

    # For each point
    for i in range(0, len(x_values)):
        k = x_values[i] # Get nfa ranking

        # Count number of individuals with nfa<=k
        y_values[i] = np.count_nonzero(nfa_values <= k) / len(nfa_values) * 100

    x_labels = np.array([str(int(sample_size*i/10)) + "\n(" + str(i*10)+"%)" for i in range(0,11)])
    
    y_ticks  = np.arange(0,110,10)

    fig = plt.figure(figsize=(10,5))

    plt.plot(x_values, y_values, 'bo-')

    plt.ylim([0, 110])
    plt.xlim([-1, sample_size*1.1])
    plt.xticks(x_values, x_labels)
    plt.yticks(y_ticks)
    plt.title("NFA curve in test", fontsize=20)
    if title != None:
        plt.title("NFA curve in test " + title)
    plt.xlabel("NFA\n(Percentaje of sample)", fontsize=15)
    plt.ylabel("% of individuals with lower NFA", fontsize=15)

    plt.grid()
    if path_savefig != None:
        plt.savefig(path_savefig, dpi=300, bbox_inches = "tight")
