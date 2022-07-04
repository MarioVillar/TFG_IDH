# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 12:49:17 2022

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



"""
###############################################################################
DISTANCE BETWEEN TRIPLETS

The distance between two embeddings is defined as the L2 distance between their
feature vectors. The distance of a triplet is the pair of distances between the
anchor and the positive images and between the anchor and the negative images.
"""


def l2_unit_vector(x):
    norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=1, keepdims=True))
    return x / norm
    
def l2_distance(a,b):
    return tf.reduce_sum(tf.square(a - b), -1)

def cosine_distance(a,b):
    dotproduct = tf.reduce_sum(tf.multiply(a,b), 1, keepdims=True )
    norm_a = tf.norm(a)
    norm_b = tf.norm(b)

    return dotproduct / (norm_a * norm_b)

"""
Compute the distance between two embeddings.
"""
def emb_distance(emb1, emb2):
    # emb1 = l2_unit_vector(emb1)
    # emb2 = l2_unit_vector(emb2)
    return l2_distance(emb1, emb2)
    

"""
Compute L2 distance between anchor-positive embeddings and 
anchor-negative embeddings.
"""
def triplets_distance(anchor, positive, negative):
    ap_distance = emb_distance(anchor, positive)
    an_distance = emb_distance(anchor, negative)

    return (ap_distance, an_distance)