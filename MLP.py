#%%
# Import libraries

import os
import io
import json
import joblib
import collections
from time import time
from tqdm import tqdm

from itertools import cycle
import random as rn
import datetime

import numpy as np 
import pandas as pd

from random import randint
from collections import Counter 

from My_functions_DL import (
                                build_folder, 
                                log_data, 
                                visualize, 
                                graph_history, 
                                graph_history_averaged,
                                combine_dictionaries, 
                                find_mean_from_combined_dicts
                            )

from sklearn.model_selection import (
                                        ShuffleSplit, 
                                        train_test_split, 
                                        KFold 
                                    )
    
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MultiLabelBinarizer
from imblearn.under_sampling import RandomUnderSampler

from sklearn.metrics import accuracy_score

from sklearn import decomposition
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline

import tensorflow as tf
from tensorflow import keras
from keras import regularizers
from keras import initializers
from keras.models import Sequential, Model
from keras import layers, metrics
from keras.layers import Input
from keras.layers import Concatenate
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import BatchNormalization
from keras.layers import Conv1D, MaxPooling1D
from keras.models import model_from_json, load_model
from keras.regularizers import *
from keras.callbacks import CSVLogger
from keras import backend as K

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

# % matplotlib inline
plt.rcParams["figure.figsize"] = [6,4]


#%%

# Function to create deep learning model

# This function takes as an input a list of dictionaries. Each element in the list is a new hidden layer in the model. For each 
# layer the dictionary defines the layer to be used.

def create_models(model_shape, input_layer_dim):
    
    # parameter rate for l2 regularization
    regConst = 0.02
    
    # defining a stochastic gradient boosting optimizer
    sgd = tf.keras.optimizers.SGD(lr = 0.001, momentum = 0.9, 
                                    nesterov = True, clipnorm = 1.)
    
    # define categorical_crossentrophy as the loss function (multi-class problem i.e. 3 age classes)
    cce = 'categorical_crossentropy'
    # bce = 'binary_crossentropy'

    # input shape vector

    # change the input shape to avoid learning feautures independently. By changing the input shape to 
    # (input_layer_dim, ) it will learn some combination of feautures with the learnable weights of the 
    # network

    input_vec = tf.keras.Input(name = 'input', shape = (input_layer_dim, )) 

    for i, layerwidth in zip(range(len(model_shape)),model_shape):
        if i == 0:
            if model_shape[i]['type'] == 'c':

                # Convolution1D layer, which will learn filters from spectra 
                # signals with maxpooling1D and batch normalization:

                xd = tf.keras.layers.Conv1D(name=('Conv'+str(i+1)), filters=model_shape[i]['filter'], 
                 kernel_size = model_shape[i]['kernel'], strides = model_shape[i]['stride'],
                 activation = 'relu',
                 kernel_regularizer = regularizers.l2(regConst), 
                 kernel_initializer = 'he_normal')(input_vec)
                xd = tf.keras.layers.BatchNormalization(name=('batchnorm_'+str(i+1)))(xd)
                xd = tf.keras.layers.MaxPooling1D(pool_size=(model_shape[i]['pooling']))(xd)
                
                # A hidden layer

            elif model_shape[i]['type'] == 'd':
                xd = tf.keras.layers.Dense(name=('d'+str(i+1)), units=model_shape[i]['width'], activation='relu', 
                 kernel_regularizer = regularizers.l2(regConst), 
                 kernel_initializer='he_normal')(input_vec)
                xd = tf.keras.layers.BatchNormalization(name=('batchnorm_'+str(i+1)))(xd) 
                xd = tf.keras.layers.Dropout(name=('dout'+str(i+1)), rate=0.5)(xd)

        else:
            if model_shape[i]['type'] == 'c':
                
                # convulational1D layer

                xd = tf.keras.layers.Conv1D(name=('Conv'+str(i+1)), filters=model_shape[i]['filter'], 
                 kernel_size = model_shape[i]['kernel'], strides = model_shape[i]['stride'],
                 activation = 'relu',
                 kernel_regularizer = regularizers.l2(regConst), 
                 kernel_initializer='he_normal')(xd)
                xd = tf.keras.layers.BatchNormalization(name=('batchnorm_'+str(i+1)))(xd)
                xd = tf.keras.layers.MaxPooling1D(pool_size=(model_shape[i]['pooling']))(xd)
                
            elif model_shape[i]['type'] == 'd':
                if model_shape[i-1]['type'] == 'c':
                    xd = tf.keras.layers.Flatten()(xd)
                    
                xd = tf.keras.layers.Dropout(name=('dout'+str(i+1)), rate = 0.5)(xd)
                xd = tf.keras.layers.Dense(name=('d'+str(i+1)), units=model_shape[i]['width'], activation='relu', 
                 kernel_regularizer = regularizers.l2(regConst), 
                 kernel_initializer = 'he_normal')(xd)
                xd = tf.keras.layers.BatchNormalization(name=('batchnorm_'+str(i+1)))(xd) 
        
    # Project the vector onto a 2 unit output layer, and squash it with a 
    # softmax activation:

    x_host_group    = tf.keras.layers.Dense(name = 'host_group', units = 2, 
                     activation = 'softmax', 
                    #  activation = 'sigmoid',
                     kernel_regularizer = regularizers.l2(regConst), 
                     kernel_initializer = 'he_normal')(xd)

    outputs = []
    for i in ['x_host_group']:
        outputs.append(locals()[i])
    model = Model(inputs = input_vec, outputs = outputs)
    
    model.compile(loss = cce, metrics = ['accuracy'], 
                  optimizer=sgd)
    model.summary()
    return model

#%%

# Load lab data

blood_meal_lab_df = pd.read_csv(
                                "C:\Mannu\Projects\Mannu Phd\lab data\Blood_meal_lab.dat", 
                                delimiter= '\t'
                            )

blood_meal_lab_df['Cat3'] = blood_meal_lab_df['Cat3'].str.replace('BF', 'Bovine')
blood_meal_lab_df['Cat3'] = blood_meal_lab_df['Cat3'].str.replace('HF', 'Human')

# Drop unused columns
blood_meal_lab_df = blood_meal_lab_df.drop(['Cat1', 'Cat2', 'Cat5'], axis=1)
blood_meal_lab_df.rename(columns = {'Cat3':'blood_meal'}, inplace = True) 

print('Size of blood meal by count', Counter(blood_meal_lab_df['blood_meal']))


#%% 

# Load hours blood meal data

blood_hours = pd.read_csv(
                            'C:\Mannu\Projects\JACQUEZ MIRS EXP\Bloodfed_hours.dat', 
                            delimiter = '\t'
                        )

# Rename items in the column
blood_hours['Cat3'] = blood_hours['Cat3'].str.replace('CW', 'Bovine')
blood_hours['Cat3'] = blood_hours['Cat3'].str.replace('HN', 'Human')

# Drop unused columns
blood_hours = blood_hours.drop(
                                [
                                    'Cat1', 
                                    'Cat2', 
                                    'Cat4', 
                                    'StoTime'
                                ], 
                                axis=1
                            )

# rename the column
blood_hours.rename(
    columns = {'Cat3':'blood_meal'}, 
    inplace = True
    ) 

# concat field data into training data
training_df = pd.concat([blood_meal_lab_df, blood_hours], axis = 0)
print('Size of blood meal by count', Counter(training_df['blood_meal']))
training_df

#%%

# define X (matrix of features) and y (list of labels)

X = np.asarray(training_df.iloc[:,1:]) # select all columns except the last one
# features = X 
y = np.asarray(training_df["blood_meal"])

# Scale data
scaler = StandardScaler().fit(X = X)
X_trans = scaler.transform(X = X) # transform X

# Serialize the scaler to a file using joblib

joblib.dump(
                scaler, 
                'C:\Mannu\Projects\Mannu Phd\Transfer_learning MLP\_transfer_learning\data_scaler.joblib'
            )

host_list = [[host] for host in y]
hosts = MultiLabelBinarizer().fit_transform(host_list)
y_classes = list(np.unique(host_list))
print(y_classes)
print(hosts)

# Labels default - all classification
labels_default, classes_default, outputs_default = [hosts], [y_classes], ['x_host_group']


#%%

# Function to train the model

# This function will split the data into training and validation, and call the create models function. 
# This fucntion returns the model and training history.

def train_models(model_to_test, save_path):

    model_shape = model_to_test["model_shape"][0]
    model_name = model_to_test["model_name"][0]
    input_layer_dim = model_to_test["input_layer_dim"][0]
    model_ver_num = model_to_test["model_ver_num"][0]
    fold = model_to_test["fold"][0]
    y_train = model_to_test["labels"][0]
    X_train = model_to_test["features"][0]
    classes = model_to_test["classes"][0]
    outputs = model_to_test["outputs"][0]
    compile_loss = model_to_test["compile_loss"][0]
    compile_metrics = model_to_test["compile_metrics"][0]

    model = create_models(model_shape, input_layer_dim)

    model.summary()
    
    history = model.fit(x = X_train, 
                        y = y_train,
                        batch_size = 256, 
                        verbose = 1, 
                        epochs = 8000,
                        validation_data = (X_val, y_val),
                        callbacks = [
                                    tf.keras.callbacks.EarlyStopping(
                                        monitor = 'val_loss', 
                                        patience = 400, 
                                        verbose = 1, 
                                        mode = 'auto'
                                        ), 
                                    CSVLogger(
                                        save_path 
                                        + model_name 
                                        + "_" 
                                        + str(model_ver_num) 
                                        + '.csv', 
                                        append = True, 
                                        separator = ';'
                                        )
                                    ]
                            )

    model.save(
                (
                    save_path 
                    + model_name 
                    + "_" 
                    + str(model_ver_num) 
                    + "_"
                    + str(fold) 
                    + "_" 
                    + 'Model.tf'
                )
            )
    
    graph_history(history, model_name, model_ver_num, fold, save_path)
            
    return model, history


# Main training and prediction section 

# Functionality:
# Define the CNN to be built.
# Build a folder to output data into.
# Call the model training.
# Organize outputs and call visualization for plotting and graphing.

outdir = "C:\Mannu\Projects\Mannu Phd\Transfer_learning MLP"
build_folder(outdir, False)  

# set model parameters
# model size when data dimension is reduced to 8 principle componets 

# Options
# Convolutional Layer:

#     type = 'c'
#     filter = optional number of filters
#     kernel = optional size of the filters
#     stride = optional size of stride to take between filters
#     pooling = optional width of the max pooling

# dense layer:

#     type = 'd'
#     width = option width of the layer

model_size = [#{'type':'c', 'filter':8, 'kernel':2, 'stride':1, 'pooling':1}, 
            #  {'type':'c', 'filter':8, 'kernel':2, 'stride':1, 'pooling':1},
            #  {'type':'c', 'filter':8, 'kernel':2, 'stride':1, 'pooling':1},
            #  {'type':'d', 'width':500},
            #  {'type':'d', 'width':500},
            #  {'type':'d', 'width':500},
             {'type':'d', 'width':500},
             {'type':'d', 'width':500},
             {'type':'d', 'width':500}]
             


# # Name the model
model_name = 'Baseline_CNN'
label = labels_default
    
# Split data into 10 folds for training/testing
# Define cross-validation strategy 

num_folds = 5
random_seed = np.random.randint(0, 81470)
# seed = 42
kf = KFold(n_splits = num_folds, shuffle = True, random_state = random_seed)

# Features
features = X_trans 
    
histories = []
averaged_histories = []
fold = 1
train_model = True

# Name a folder for the outputs to go into

savedir = (outdir+"\_transfer_learning")            
build_folder(savedir, False)
savedir = (outdir+"\_transfer_learning\l")            
           
# start model training on standardized data

start_time = time()
save_predicted = []
save_true = []
save_hist = []

for train_index, test_index in kf.split(features):

    # Split data into test and train

    X_trainset, X_test = features[train_index], features[test_index]
    y_trainset, y_test = list(map(lambda y:y[train_index], label)), list(map(lambda y:y[test_index], label))

    # Further divide training dataset into train and validation dataset 
    # with an 90:10 split
    
    validation_size = 0.1
    X_train, X_val, y_train, y_val = train_test_split(X_trainset,
                                        *y_trainset, test_size = validation_size, random_state = seed)

    
    # expanding to one dimension, because the conv layer expcte to, 1
    X_train = X_train.reshape([X_train.shape[0], -1])
    X_val = X_val.reshape([X_val.shape[0], -1])
    X_test = X_test.reshape([X_test.shape[0], -1])

    # Check the sizes of all newly created datasets
    print("Shape of X_train:", X_train.shape)
    print("Shape of X_val:", X_val.shape)
    print("Shape of X_test:", X_test.shape)
    print("Shape of y_train:", y_train.shape)
    print("Shape of y_val:", y_val.shape)
    # print("Shape of y_test:", y_test.shape)

    input_layer_dim = len(X[0])

    model_to_test = {
                        "model_shape" : [model_size], # defines the hidden layers of the model
                        "model_name"  : [model_name],
                        "input_layer_dim"  : [input_layer_dim], # size of input layer
                        "model_ver_num"  : [0],
                        "fold"  : [fold], # kf.split number on
                        "labels"   : [y_train],
                        "features" : [X_train],
                        "classes"  : [classes_default],
                        "outputs"   : [outputs_default],
                        "compile_loss": [{'host_group': 'categorical_crossentropy'}],
                        "compile_metrics" :[{'host_group': 'accuracy'}]
                    }

    # Call function to train all the models from the dictionary
    model, history = train_models(model_to_test, savedir)
    histories.append(history)

    print(X_test.shape)

    # predict the unseen dataset/new dataset
    y_predicted = model.predict(X_test)

    # change the dimension of y_test to array
    y_test = np.asarray(y_test)
    y_test = np.squeeze(y_test) # remove any single dimension entries from the arrays

    print('y predicted shape', y_predicted.shape)
    print('y_test', y_test.shape)

    # save predicted and true value in each iteration for plotting averaged confusion matrix

    for pred, tru in zip(y_predicted, y_test):
        save_predicted.append(pred)
        save_true.append(tru)

    hist = history.history
    averaged_histories.append(hist)

    # Plotting confusion matrix for each fold/iteration

    visualize(histories, savedir, model_name, str(fold), classes_default, outputs_default, y_predicted, y_test)
    # log_data(X_test, 'test_index', fold, savedir)

    fold += 1

    # Clear the Keras session, otherwise it will keep adding new
    # models to the same TensorFlow graph each time we create
    # a model with a different set of hyper-parameters.

    K.clear_session()

    # Delete the Keras model with these hyper-parameters from memory.
    del model

save_predicted = np.asarray(save_predicted)
save_true = np.asarray(save_true)
print('save predicted shape', save_predicted.shape)
print('save.true shape', save_true.shape)

# Plotting an averaged confusion matrix

visualize(1, savedir, model_name, "Averaged_training", classes_default, outputs_default, save_predicted, save_true)

end_time = time()
print('Run time : {} s'.format(end_time-start_time))
print('Run time : {} m'.format((end_time-start_time)/60))
print('Run time : {} h'.format((end_time-start_time)/3600))


#%%

# combine all dictionaries together for the base model training (using Ifakara data)

combn_dictionar = combine_dictionaries(averaged_histories)

with open(
    'C:\Mannu\Projects\Mannu Phd\Transfer_learning MLP\_transfer_learning\combined_history_dictionaries_base_model.txt',
     'w'
     ) as outfile:
     json.dump(
         combn_dictionar, 
         outfile
         )

# find the average of all dictionaries 

combn_dictionar_average = find_mean_from_combined_dicts(combn_dictionar)

# Plot averaged histories

graph_history_averaged(combn_dictionar_average)
