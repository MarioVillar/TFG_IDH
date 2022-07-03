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


faceNet_model_path = "/mnt/homeGPU/mvillar/FaceNet_pretrained/facenet_keras.h5"


path_fig_hist_thresholds = "/mnt/homeGPU/mvillar/Modelos_entrenados/AP_AN_dists/"

path_fig_hist_thresholds_trad_nn = "/mnt/homeGPU/mvillar/Modelos_entrenados/AP_AN_dists/Trad_NN/"


# Result files paths
results_online_file_path = "/mnt/homeGPU/mvillar/Modelos_entrenados/resultados_online_gen_hold_out.pkl"



results_cv_without_on_file_path = "/mnt/homeGPU/mvillar/Modelos_entrenados/resultados_CV_without_online.pkl"

results_cv_online_file_path = "/mnt/homeGPU/mvillar/Modelos_entrenados/resultados_CV_online_gen.pkl"



results_traditional_nn = "/mnt/homeGPU/mvillar/Modelos_entrenados/resultados_trad_nn_hold_out.pkl"

results_traditional_nn_2 = "/mnt/homeGPU/mvillar/Modelos_entrenados/resultados_trad_nn_2_hold_out.pkl"

results_traditional_nn_reg = "/mnt/homeGPU/mvillar/Modelos_entrenados/resultados_trad_nn_reg_hold_out.pkl"

results_cv_traditional_nn_file_path = "/mnt/homeGPU/mvillar/Modelos_entrenados/resultados_CV_trad_nn.pkl"

results_cv_traditional_nn_reg_file_path = "/mnt/homeGPU/mvillar/Modelos_entrenados/resultados_CV_trad_nn_reg.pkl"


results_traditional_nn_mean_rank = "/mnt/homeGPU/mvillar/Modelos_entrenados/resultados_trad_nn_mean_rank_hold_out.pkl"

results_traditional_nn_reg_mean_rank = "/mnt/homeGPU/mvillar/Modelos_entrenados/resultados_trad_nn_reg_mean_rank_hold_out.pkl"


results_efficentNet = "/mnt/homeGPU/mvillar/Modelos_entrenados/resultados_efficentNet_hold_out.pkl"

results_efficentNet_reg = "/mnt/homeGPU/mvillar/Modelos_entrenados/resultados_efficentNet_reg_hold_out.pkl"