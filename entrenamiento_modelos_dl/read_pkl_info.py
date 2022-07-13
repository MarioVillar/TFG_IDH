# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 10:25:38 2022

@author: mario
"""


from utilities import *


snn = False


file_name = 'results_snnsigmoid_reg_hold_out'


# resultados_CV_online_gen
# resultados_CV_online_gen_adagrad
# resultados_CV_trad_nn
# resultados_CV_trad_nn_reg
# resultados_CV_without_online
# resultados_CV_without_online_adagrad


path_pkl = 'C:/Users/mario/Documents/mario/ingenieria_informatica/TFG/codigo/' + \
           'cross_validation_servidores_UGR/Modelos_entrenados/' + \
           'resultados_CV_finales/' + file_name + '.pkl'
           
           
path_pkl = 'C:/Users/mario/Documents/mario/ingenieria_informatica/TFG/codigo/' + \
           'cross_validation_servidores_UGR/Modelos_entrenados/' + \
           'resultados_hold_out/' + file_name + '.pkl'
           
path_nfa_hold_out = 'C:/Users/mario/Documents/mario/ingenieria_informatica/TFG/codigo/' + \
                    'cross_validation_servidores_UGR/Modelos_entrenados/NFA/'

path_cmc_hold_out = 'C:/Users/mario/Documents/mario/ingenieria_informatica/TFG/codigo/' + \
                    'cross_validation_servidores_UGR/Modelos_entrenados/CMC/'
           
path_nfa_CV = 'C:/Users/mario/Documents/mario/ingenieria_informatica/TFG/codigo/' + \
              'cross_validation_servidores_UGR/Modelos_entrenados/NFA_CV/'

path_cmc_CV = 'C:/Users/mario/Documents/mario/ingenieria_informatica/TFG/codigo/' + \
              'cross_validation_servidores_UGR/Modelos_entrenados/CMC_CV/'


##############
# Obtain object to save the results from file
exp_results = None

if snn:
    exp_results = expResults.get_results_from_disk(path_pkl)
else:
    exp_results = expResultsTrad.get_results_from_disk(path_pkl)


sample_size = exp_results.face_sample_size

nmodels_show = 3

########################################################
print('\n\n', '*'*70, "\nTop " + str(nmodels_show) + " models referring first top-k acc = 1.0", sep='')
best_topk_models = exp_results.get_n_best_topk_acc_models(nmodels_show)

for i in best_topk_models:
    modelo_i = exp_results.get_model(i)
    print("\nModelo", i)
    print(modelo_i['parameters'])
    
    if snn:
        print("Epochs before Early Stopping =", len(modelo_i['val_history']))
        print("Positive accuracy", "{0:.4f}".format(modelo_i['acc_tree'][0]),
              "Negative accuracy", "{0:.4f}".format(modelo_i['acc_tree'][1]),
              "Overall accuracy",  "{0:.4f}".format(modelo_i['acc_tree'][2]))
        print("First Top-k accuracy = 1.0:", modelo_i['first_top_k'][2])
    else:
        print("Epochs before Early Stopping =", len(modelo_i['val_history']))
        print("Positive accuracy", "{0:.4f}".format(modelo_i['accuracies'][0]),
              "Negative accuracy", "{0:.4f}".format(modelo_i['accuracies'][1]),
              "Overall accuracy",  "{0:.4f}".format(modelo_i['accuracies'][2]))
        print("First Top-k accuracy = 1.0:", modelo_i['first_top_k'])
    

########################################################
print('\n\n', '*'*70, "\nTop " + str(nmodels_show) + " models referring overall accuracy", sep='')
best_acc_models = exp_results.get_n_best_acc_models(nmodels_show)

for j in best_acc_models:
    modelo_j = exp_results.get_model(j)
    print("\nModelo", j)
    print(modelo_j['parameters'])
    
    if snn:
        print("Epochs before Early Stopping =", len(modelo_j['val_history']))
        print("Positive accuracy", "{0:.4f}".format(modelo_j['acc_tree'][0]),
              "Negative accuracy", "{0:.4f}".format(modelo_j['acc_tree'][1]),
              "Overall accuracy",  "{0:.4f}".format(modelo_j['acc_tree'][2]))
        print("First Top-k accuracy = 1.0:", modelo_j['first_top_k'][2])
    else:
        print("Epochs before Early Stopping =", len(modelo_j['val_history']))
        print("Positive accuracy", "{0:.4f}".format(modelo_j['accuracies'][0]),
              "Negative accuracy", "{0:.4f}".format(modelo_j['accuracies'][1]),
              "Overall accuracy",  "{0:.4f}".format(modelo_j['accuracies'][2]))
        print("First Top-k accuracy = 1.0:", modelo_j['first_top_k'])



# ########################################################
# print('\n\n', '*'*70, "\nTop " + str(nmodels_show) + " models referring positive accuracy", sep='')
# best_acc_models = exp_results.get_n_best_pos_acc_models(nmodels_show)

# for j in best_acc_models:
#     modelo_j = exp_results.get_model(j)
#     print("\nModelo", j)
#     print(modelo_j['parameters'])
    
#     if snn:
#         print("Epochs before Early Stopping =", len(modelo_j['val_history']))
#         print("Positive accuracy", "{0:.4f}".format(modelo_j['acc_tree'][0]),
#               "Negative accuracy", "{0:.4f}".format(modelo_j['acc_tree'][1]),
#               "Overall accuracy",  "{0:.4f}".format(modelo_j['acc_tree'][2]))
#         print("First Top-k accuracy = 1.0:", modelo_j['first_top_k'][2])
#     else:
#         print("Epochs before Early Stopping =", len(modelo_j['val_history']))
#         print("Positive accuracy", "{0:.4f}".format(modelo_j['accuracies'][0]),
#               "Negative accuracy", "{0:.4f}".format(modelo_j['accuracies'][1]),
#               "Overall accuracy",  "{0:.4f}".format(modelo_j['accuracies'][2]))
#         print("First Top-k accuracy = 1.0:", modelo_j['first_top_k'])


# ########################################################
# print('\n\n', '*'*70, "\nTop " + str(nmodels_show) + " models referring negative accuracy", sep='')
# best_acc_models = exp_results.get_n_best_neg_acc_models(nmodels_show)

# for j in best_acc_models:
#     modelo_j = exp_results.get_model(j)
#     print("\nModelo", j)
#     print(modelo_j['parameters'])
    
#     if snn:
#         print("Epochs before Early Stopping =", len(modelo_j['val_history']))
#         print("Positive accuracy", "{0:.4f}".format(modelo_j['acc_tree'][0]),
#               "Negative accuracy", "{0:.4f}".format(modelo_j['acc_tree'][1]),
#               "Overall accuracy",  "{0:.4f}".format(modelo_j['acc_tree'][2]))
#         print("First Top-k accuracy = 1.0:", modelo_j['first_top_k'][2])
#     else:
#         print("Epochs before Early Stopping =", len(modelo_j['val_history']))
#         print("Positive accuracy", "{0:.4f}".format(modelo_j['accuracies'][0]),
#               "Negative accuracy", "{0:.4f}".format(modelo_j['accuracies'][1]),
#               "Overall accuracy",  "{0:.4f}".format(modelo_j['accuracies'][2]))
#         print("First Top-k accuracy = 1.0:", modelo_j['first_top_k'])


# ########################################################
# # Plot CMCs and NFAs
# for i in np.append(best_topk_models, best_acc_models):
#     modelo_i = exp_results.get_model(i)
    
#     if snn:
#         show_cmc_curve(modelo_i['cmc_tree'], sample_size,
#                         #path_savefig=path_cmc+str(i), 
#                         title=modelo_i['parameters'])
        
#         show_nfa_curve(modelo_i['nfa_tree'], sample_size,
#                         #path_savefig=path_nfa+str(i), 
#                         title=modelo_i['parameters'])
#     else:
#         show_cmc_curve(modelo_i['cmc_values'], sample_size,
#                         path_savefig=path_cmc+str(i), 
#                         title=modelo_i['parameters'])
        
#         show_nfa_curve(modelo_i['nfa_values'], sample_size,
#                         path_savefig=path_nfa+str(i), 
#                         title=modelo_i['parameters'])



# ########################################################
# # Plot CMCs and NFAs for Cross Validation
# for i in np.append(best_topk_models, best_acc_models):
#     modelo_i = exp_results.get_model(i)
    
#     if snn:
#         show_cmc_curve(modelo_i['cmc_tree'], sample_size,
#                         path_savefig = path_cmc_CV + file_name + '_' + str(i), 
#                         title=modelo_i['parameters'])
        
#         show_nfa_curve(modelo_i['nfa_tree'], sample_size,
#                         path_savefig = path_nfa_CV + file_name + '_' + str(i), 
#                         title=modelo_i['parameters'])
#     else:
#         show_cmc_curve(modelo_i['cmc_values'], sample_size,
#                         path_savefig = path_cmc_CV + file_name + '_' + str(i), 
#                         title=modelo_i['parameters'])
        
#         show_nfa_curve(modelo_i['nfa_values'], sample_size,
#                         path_savefig = path_nfa_CV + file_name + '_' + str(i), 
#                         title=modelo_i['parameters'])
    
    
    
    