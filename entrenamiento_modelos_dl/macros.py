# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 12:33:37 2022

@author: mario
"""

# Shape of images
input_shape = (224, 224)
faceNet_shape = (160,160)

# Set seed for random operations
SEED = 42

# Train batch size
train_batch_size  = 64

validation_batch_size = 1

test_batch_size = 1

train_batch_size_online = 256 # SNN with online generation of triplets


# Number of folds in Cross Validation
n_folds = 5


# Validation split
val_split = 0.1


# Test split for Hold-out experiments
test_split = 0.25


# Face data augmentation
ctrst_factor = 0.3 # Contrast factor
brt_factor   = 0.15 # Brightness factor
hue_factor   = 0.3 # Hue factor
sat_upper_factor = 0.5 # Saturation upper factor
sat_lower_factor = 1.5 # Saturation lower factor



# Directories of data 
face_im_path  = "/mnt/homeGPU/mvillar/dataset/FOTOS_AM"
skull_im_path = "/mnt/homeGPU/mvillar/dataset/PM_DATA_IMAGES"
info_path     = "/mnt/homeGPU/mvillar/dataset/informacion_dataset.xls"
UTKFace_path  = "/mnt/homeGPU/mvillar/dataset/UTKFace"


# FaceNet pretrained model path
faceNet_model_path = "/mnt/homeGPU/mvillar/FaceNet_pretrained/facenet_keras.h5"




#######################################
# PATH OF RESULTS FILES

# SNNTLBASIC
results_snntlbasic_hold_out_file_path = "/mnt/homeGPU/mvillar/Modelos_entrenados/results_snntlbasic_hold_out.pkl"

# SNNTLSEL
results_snntlsel_hold_out_file_path = "/mnt/homeGPU/mvillar/Modelos_entrenados/results_snntlsel_hold_out.pkl"

# SNNTLC
results_snntlc_hold_out_file_path = "/mnt/homeGPU/mvillar/Modelos_entrenados/results_snntlc_hold_out.pkl"

# SNNTLONLINE
results_snntlonline_hold_out_file_path = "/mnt/homeGPU/mvillar/Modelos_entrenados/results_snntlonline_hold_out.pkl"

# SNNSIGMOID LESS REG
results_snnsigmoid_hold_out_file_path = "/mnt/homeGPU/mvillar/Modelos_entrenados/results_snnsigmoid_hold_out.pkl"

# SNNSIGMOID STRONGER REG
results_snnsigmoid_reg_hold_out_file_path = "/mnt/homeGPU/mvillar/Modelos_entrenados/results_snnsigmoid_reg_hold_out.pkl"


