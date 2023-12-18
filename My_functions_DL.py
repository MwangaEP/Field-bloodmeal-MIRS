
#%%

# Import libraries

import os

import json
import itertools
import collections
from time import time

import numpy as np 
import pandas as pd

from random import randint
from collections import Counter 

from sklearn.model_selection import (
                                        ShuffleSplit, 
                                        train_test_split, 
                                        KFold
                                    )
from sklearn.metrics import (
                                confusion_matrix, 
                                classification_report, 
                                f1_score, 
                                recall_score, 
                                precision_score
                            )

import matplotlib.pyplot as plt # for making plots
import seaborn as sns

sns.set(
            context = "paper",
            style = "white",
            palette = "deep",
            font_scale = 2.0,
            color_codes = True,
            rc = ({"font.family": "Dejavu Sans"})
        )
# %matplotlib inline

#%%

# create a new folder for the CNN outputs

def build_folder(Fold, to_build = False):
    if not os.path.isdir(Fold):
        if to_build == True:
            os.mkdir(Fold)
        else:
            print('Directory does not exists, not creating directory!')
    else:
        if to_build == True:
            raise NameError('Directory already exists, cannot be created!')

#%%

# This normalizes the confusion matrix and ensures neat plotting for all outputs.
# Function for plotting confusion matrcies

def plot_confusion_matrix(
                            cm, 
                            classes,  
                            save_path, 
                            model_name, 
                            fold,
                            normalize = True,
                            title = 'Confusion matrix',
                            xrotation = 0,
                            yrotation = 0,
                            cmap = plt.cm.Blues,
                            printout = False
                        ):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        if printout:
            print("Normalized confusion matrix")
    else:
        if printout:
            print('Confusion matrix')

    if printout:
        print(cm)
    
    # plt.figure(figsize=(6,4))

    plt.imshow(cm, interpolation='nearest', vmin = 0.2, vmax = 1.0, cmap = cmap)
    # plt.title([title +' - '+ model_name])
    plt.colorbar()
    classes = classes[0]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = xrotation)
    plt.yticks(tick_marks, classes, rotation = yrotation)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j, 
            i, 
            format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black"
                )

    plt.tight_layout()
    plt.ylabel('True Host', weight = 'bold')
    plt.xlabel('Predicted Host', weight = 'bold')
    plt.savefig(
                    (
                        save_path 
                        + "Confusion_Matrix_" 
                        + model_name 
                        + "_" 
                        + fold 
                        + "_" 
                        + ".png"
                    ), 
                        dpi = 500, 
                        bbox_inches = "tight"
                )

    plt.close()

#%%
# Visualizing outputs

# for visualizing losses and metrics once the neural network fold is trained
def visualize(
                histories, 
                save_path, 
                model_name, 
                fold, 
                classes, 
                predicted, 
                true
            ):

    # Sort out predictions and true labels

    classes_pred = np.argmax(predicted, axis=-1)
    classes_true = np.argmax(true, axis=-1)
 
    cnf_matrix = confusion_matrix(classes_true, classes_pred)
    plot_confusion_matrix(cnf_matrix, classes, save_path, model_name, fold)

#%%

#%%

# Data logging
# for logging data associated with the model

def log_data(log, name, fold, save_path):
    f = open((save_path+name+'_'+str(fold)+'_log.txt'), 'w')
    np.savetxt(f, log)
    f.close()

#%%

# Graphing the training data and validation
 
def graph_history(
                    history, 
                    model_name, 
                    model_ver_num, 
                    fold, 
                    save_path
                ):

    #not_validation = list(filter(lambda x: x[0:3] != "val", history.history.keys()))
    
    print('history.history.keys : {}'.format(history.history.keys()))
    filtered = filter(lambda x: x[0:3] != "val", history.history.keys())
    not_validation = list(filtered)
    for i in not_validation:
        plt.figure(figsize=(6, 4))
        # plt.title(i+"/ "+"val_"+i)
        plt.plot(history.history[i], label=i)
        plt.plot(history.history["val_"+i], label="val_"+i)
        plt.legend()
        plt.tight_layout()
        plt.grid(False)
        plt.xlabel("epoch", weight = 'bold')
        plt.ylabel(i)
        plt.savefig(
                        save_path 
                        + model_name 
                        + "_" 
                        + str(model_ver_num) 
                        + "_"+str(fold) 
                        + "_" 
                        + i 
                        + ".png", 
                        dpi = 500, 
                        bbox_inches = "tight"
                    )
        # plt.savefig(save_path +model_name+"_"+str(model_ver_num)+"_"+str(fold)+"_"+i + ".pdf", dpi = 500, bbox_inches="tight")
        plt.close()

#%%
# Graphing the averaged training and validation histories 
 
# when plotting, smooth out the points by some factor (0.5 = rough, 0.99 = smooth)
# method taken from `Deep Learning with Python` by François Chollet

def smooth_curve(points, factor = 0.75):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def set_plot_history_data(ax, history, which_graph):

    if which_graph == 'accuracy':
        train = smooth_curve(history['accuracy'])
        valid = smooth_curve(history['val_accuracy'])

    epochs = range(1, len(train) + 1)
        
    trim = 0 # remove first 5 epochs
    # when graphing loss the first few epochs may skew the (loss) graph
    
    ax.plot(
                epochs[trim:], 
                train[trim:], 
                'b', 
                label = ('accuracy')
            )

    ax.plot(
                epochs[trim:], 
                train[trim:], 
                'b', 
                linewidth = 15, 
                alpha = 0.1
            )
    
    ax.plot(
                epochs[trim:], 
                valid[trim:], 
                'orange', 
                label = ('val_accuracy')
            )

    ax.plot(
                epochs[trim:], 
                valid[trim:], 
                'orange', 
                linewidth = 15, 
                alpha = 0.1
            )


def graph_history_averaged(combined_history):

    print('averaged_histories.keys : {}'.format(combined_history.keys()))
    
    fig, (ax1) = plt.subplots(
                                nrows = 1,
                                ncols = 1,
                                figsize = (6, 4),
                                sharex = True
                            )

    set_plot_history_data(ax1, combined_history, 'accuracy')
    
    # Accuracy graph
    ax1.set_ylabel('Accuracy', weight = 'bold')
    plt.xlabel('Epoch', weight = 'bold')
    # ax1.set_ylim(bottom = 0.3, top = 1.0)
    ax1.legend(loc = 'lower right')
    ax1.set_yticks(np.arange(0.2, 1.0 + 0.05, step = 0.1))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.spines['bottom'].set_visible(True)

    plt.tight_layout()
    plt.grid(False)
    plt.savefig(
                "C:\Mannu\Projects\Mannu Phd\Transfer_learning MLP\_transfer_learning\Averaged_graph_base_model.png", 
                dpi = 500, 
                bbox_inches = "tight"
            )
    plt.close()

#%%
# This function takes a list of dictionaries, and combines them into a single dictionary in which each key maps to a 
# list of all the appropriate values from the parameters of the dictionaries

def combine_dictionaries(list_of_dictionaries):
    
    combined_dictionaries = {}
    
    for individual_dictionary in list_of_dictionaries:
        
        for key_value in individual_dictionary:
            
            if key_value not in combined_dictionaries:
                
                combined_dictionaries[key_value] = []
            combined_dictionaries[key_value].append(individual_dictionary[key_value])

    return combined_dictionaries


#%%

# This function calculates the average of the combined dictionaries either of same length or not the same length, 
# and return the mean

def find_mean_from_combined_dicts(combined_dicts):
    
    dict_of_means = {}

    for key_value in combined_dicts:
        dict_of_means[key_value] = []

        # Length of longest list return the longest list within the list of a dictionary item
        length_of_longest_list = max([len(a) for a in combined_dicts[key_value]])
        temp_array = np.empty([len(combined_dicts[key_value]), length_of_longest_list])
        temp_array[:] = np.NaN

        for i, j in enumerate(combined_dicts[key_value]):
            temp_array[i][0:len(j)] = j
        mean_value = np.nanmean(temp_array, axis=0)

        dict_of_means[key_value] = mean_value.tolist()
    
    return dict_of_means

#%%

# Function to calculate Human bloodmeal index (HBI)

def human_blood_index(beta, alpha):

    '''
    The formular to calculate blood index

    beta: for PCR; Number of mosquitoes blood-fed on human/
          for ML; Number of mosquitoes predicted as human blood-fed
    alpha: for PCR; Total number of mosquitoes for ML testset (human & bovine)/
           for ML; Total number of predicted blood-fed mosquitoes
    '''
    bi = beta/alpha

    return bi