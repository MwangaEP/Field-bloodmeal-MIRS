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

from MLP_functions import (
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

from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow import keras
from keras import regularizers
from keras import optimizers
from keras import initializers
from keras.models import Sequential, Model
from keras import layers, metrics
from keras.layers import Input
from keras.layers import Concatenate
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import BatchNormalization, LayerNormalization
from keras.layers import Conv1D, MaxPooling1D
from keras.callbacks import EarlyStopping
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

class AttentionPooling(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.attention = layers.Dense(1)
        super().build(input_shape)

    def call(self, inputs):
        weights = tf.nn.softmax(self.attention(inputs), axis=1)
        return tf.reduce_sum(inputs * weights, axis=1)
    

def create_models(model_shape, input_layer_dim):

    '''
    Docstring for create_models
    
    Parameters
    ----------  
    :param model_shape: The model architecture defined as a list of dictionaries
    :param input_layer_dim: model input layer dimension (number of wavelengths)
    '''
    
    # parameter rate for l2 regularization
    regConst = 0.02
    
    # defining a stochastic gradient boosting optimizer
    sgd = optimizers.SGD(
        learning_rate = 0.001, 
        momentum = 0.9, 
        nesterov = True, 
        clipnorm = 1.
        )
    
    # define categorical_crossentrophy as the loss function 
    cce = 'categorical_crossentropy'

    # ------------------------------------------------------------------------------
    # input shape vector
    # ------------------------------------------------------------------------------

    input_vec = Input(
        name = 'input', 
        shape = (input_layer_dim, 1) # (wavelengths, channels)
        ) 
    
    xd = input_vec
    conv_seen = False # track when we exist sequence domain

    # ------------------------------------------------------------------------------
    # Build feature extractor
    # ------------------------------------------------------------------------------

    for i, layer_cfg in enumerate(model_shape):

        if i == 0:
            if layer_cfg['type'] == 'c':

                # --------------------Conv1D block ---------------------------

                # Convolutional layer to learn from spectra using filters

                xd = Conv1D(
                filters = layer_cfg['filter'],
                kernel_size = layer_cfg['kernel'],
                strides = layer_cfg['stride'],
                padding = 'same',
                activation = 'relu',
                kernel_initializer = 'he_normal',
                kernel_regularizer = regularizers.l2(regConst),
                name = f'Conv{i+1}'
                )(xd)

                xd = BatchNormalization(name = f'batchnorm_{i+1}')(xd)

                xd = MaxPooling1D(
                    pool_size = layer_cfg['pooling'],
                    name = f'MaxPool{i+1}'
                )(xd)

                # A hidden layer 

            elif layer_cfg['type'] == 'd':
                xd = Dense(
                    units = layer_cfg['width'],
                    activation = 'relu',
                    kernel_initializer = 'he_normal',
                    kernel_regularizer = regularizers.l2(regConst),
                    name = f'Dense{i+1}'
                )(xd)

                xd = BatchNormalization(name = f'batchnorm_{i+1}')(xd)
                xd = Dropout(rate = 0.4, name = f'Dropout{i+1}')(xd)
        
        else:
            if layer_cfg['type'] == 'c':
                
                # --------------------Conv1D block ---------------------------

                xd = Conv1D(
                    filters = layer_cfg['filter'],
                    kernel_size = layer_cfg['kernel'],
                    strides = layer_cfg['stride'],
                    padding = 'same',
                    activation = 'relu',
                    kernel_initializer = 'he_normal',
                    kernel_regularizer = regularizers.l2(regConst),
                    name = f'Conv{i+1}'
                )(xd)

                xd = BatchNormalization(name = f'batchnorm_{i+1}')(xd)

                xd = MaxPooling1D(
                    pool_size = layer_cfg['pooling'],
                    name = f'MaxPool{i+1}'
                )(xd)

            elif layer_cfg['type'] == 'd':
                # Check if the previous layer in the model shape was a conv
                # if layer_cfg[i-1]['type'] == 'c':
                if 1 > 0 and model_shape[i-1]['type'] == 'c':
                    xd = Flatten()(xd)
                    # xd = AttentionPooling(name = f'AttentionPool_{i+1}')(xd)
                
                xd = Dense(
                    units = layer_cfg['width'],
                    activation = 'relu',
                    kernel_initializer = 'he_normal',
                    kernel_regularizer = regularizers.l2(regConst),
                    name = f'Dense{i+1}'
                )(xd)

                xd = Dropout(rate = 0.4, name = f'Dropout{i+1}')(xd)
                xd = BatchNormalization(name = f'batchnorm_{i+1}')(xd)

    # Project the vector into the output layer
    # softmax activation is used for multi-class classification

    x_host_group = Dense(
        units = 2,
        activation = 'softmax',
        kernel_initializer = 'he_normal',
        kernel_regularizer = regularizers.l2(regConst),
        name = 'x_host_group'
    )(xd)

    outputs = [x_host_group]
    # for i in [x_host_group]:
    #     outputs.append(locals()[i])
    model = Model(inputs = input_vec, outputs = outputs)

    # Compile the model
    model.compile(
        loss = cce, 
        metrics = ['accuracy'], 
        optimizer = sgd
    )

    model.summary()
    return model
 
    #     # --------------------Conv1D block ---------------------------
    #     if layer_cfg['type'] == 'c':
    #         conv_seen = True

    #         xd = Conv1D(
    #             filters = layer_cfg['filter'],
    #             kernel_size = layer_cfg['kernel'],
    #             strides = layer_cfg['stride'],
    #             padding = 'same',
    #             activation = 'relu',
    #             kernel_initializer = 'he_normal',
    #             kernel_regularizer = regularizers.l2(regConst),
    #             name = f'Conv{i+1}'
    #         )(xd)

    #         xd = BatchNormalization(name = f'batchnorm_{i+1}')(xd)

    #         xd = MaxPooling1D(
    #             pool_size = layer_cfg['pooling'],
    #             name = f'MaxPool{i+1}'
    #         )(xd)

    #     # --------------------Dense block ---------------------------

    #     elif layer_cfg['type'] == 'd':

    #         # First layer after conv layers → attention pooling
    #         if conv_seen:
    #             xd = Flatten()(xd)
    #             xd = AttentionPooling(name = f'AttentionPool_{i+1}')(xd)
    #             conv_seen = False # ensure pooling happens only once

    #         xd = Dense(
    #             units = layer_cfg['width'],
    #             activation = 'relu',
    #             kernel_initializer = 'he_normal',
    #             kernel_regularizer = regularizers.l2(regConst),
    #             name = f'Dense{i+1}'
    #         )(xd)

    #         xd = LayerNormalization(name = f'layernorm_{i+1}')(xd)
    #         xd = Dropout(rate = 0.4, name = f'Dropout{i+1}')(xd)

    #     else:
    #         raise ValueError(f"Unknown layer type: {layer_cfg['type']}")
        
    # # ------------------------------------------------------------------------------
    # # Output layers
    # # ------------------------------------------------------------------------------

    # output  = Dense(
    #     units = 2,
    #     activation = 'softmax',
    #     kernel_initializer = 'he_normal',
    #     kernel_regularizer = regularizers.l2(regConst),
    #     name = 'x_host_group'
    # )(xd)

    # model = Model(inputs = input_vec, outputs = output)

    # # Compile the model
    # model.compile(
    #     loss = cce, 
    #     metrics = ['accuracy'], 
    #     optimizer = sgd
    # )

    # model.summary()
    # return model
    
#%%

# Load lab data

blood_meal_df = pd.read_csv(
    os.path.join("..", "Data", "Blood_meal_lab.dat"), 
    delimiter= '\t'
    )

blood_meal_df['Cat3'] = blood_meal_df['Cat3'].str.replace('BF', 'Bovine')
blood_meal_df['Cat3'] = blood_meal_df['Cat3'].str.replace('HF', 'Human')

# Drop unused columns
blood_meal_df = blood_meal_df.drop(
    [
        'Cat1', 
        'Cat2', 
        'Cat5'
    ], axis=1
    )

blood_meal_df.rename(
    columns = {'Cat3':'blood_meal'}, 
    inplace = True
    ) 

print('Size of blood meal by count', Counter(blood_meal_df['blood_meal']))


#%%

# define X (matrix of features) and y (list of labels)
X = np.asarray(blood_meal_df.iloc[:,1:]) # select all columns except the first one
# features = X 
y = np.asarray(blood_meal_df["blood_meal"])

# Scale data
scaler = StandardScaler().fit(X = X)
X_trans = scaler.transform(X = X) # transform X

# Serialize the scaler to a file using joblib

joblib.dump(
    scaler, 
    os.path.join("..", "Results", "data_scaler_cnn.joblib")
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
# This function returns the model and training history.

def train_models(model_to_test, save_path, X_val, y_val):

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
    
    history = model.fit(
        x = X_train, 
        y = y_train,
        batch_size = 32, 
        verbose = 1, 
        epochs = 500,
        validation_data = (X_val, y_val),
        callbacks = [
            EarlyStopping(
                monitor = 'val_loss', 
                patience = 30,
                restore_best_weights = True, 
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
        save_path 
        + model_name 
        + "_" 
        + str(model_ver_num) 
        + "_"
        + str(fold) 
        + "_" 
        + 'Model.keras'
    )
    
    graph_history(history, model_name, model_ver_num, fold, save_path)
            
    return model, history

#-----------------------------------------------------------------------------------
# Main training and prediction section 
#-----------------------------------------------------------------------------------

# Functionality:
# Define the CNN to be built.
# Build a folder to output data into.
# Call the model training.
# Organize outputs and call visualization for plotting and graphing.

outdir = os.path.join("..", "Results")
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

model_size = [
    {'type':'c', 'filter':8, 'kernel':4, 'stride':1, 'pooling':1}, 
    {'type':'c', 'filter':4, 'kernel':2, 'stride':1, 'pooling':1},
    # {'type':'c', 'filter':32, 'kernel':3, 'stride':1, 'pooling':2},
    {'type':'d', 'width':50},
    # {'type':'d', 'width':128}
    ]

# # Name the model
model_name = 'CNN_attentionPooling'
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

savedir = (outdir + r"\CNN")            
build_folder(savedir, False)
savedir = (outdir + r"\CNN\l")            
           
# start model training on standardized data

start_time = time()
save_predicted = []
save_true = []

for train_index, test_index in kf.split(features):

    # Split data into test and train

    X_trainset, X_test = features[train_index], features[test_index]
    y_trainset, y_test = list(map(lambda y:y[train_index], label)), list(map(lambda y:y[test_index], label))

    # Further divide training dataset into train and validation dataset 
    # with an 90:10 split
    
    validation_size = 0.1
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainset,
        *y_trainset, 
        test_size = validation_size, 
        random_state = 42
    )

    
    # expanding dimesions of input data to 3D for CNN input
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # # Data augmentation on training data only
    # def augment_spectra(
    #         X,
    #         noise_std=0.003,
    #         scale_range=0.05,
    #         seed=None
    #     ):

    #     """
    #     X shape: (n_samples, n_wavelengths, 1)
    #     """
    #     rng = np.random.default_rng(seed)

    #     noise = rng.normal(
    #         loc=0.0,
    #         scale=noise_std,
    #         size=X.shape
    #     )

    #     scale = rng.uniform(
    #         1.0 - scale_range,
    #         1.0 + scale_range,
    #         size=(X.shape[0], 1, 1)
    #     )

    #     return (X + noise) * scale

    # X_train_aug = augment_spectra(X_train, noise_std=0.003, scale_range=0.05)

    # X_train = np.concatenate([X_train, X_train_aug], axis=0)
    # y_train = np.concatenate([y_train, y_train], axis=0)

    # Check the sizes of all newly created datasets
    print("Shape of X_train:", X_train.shape)
    print("Shape of X_val:", X_val.shape)
    print("Shape of X_test:", X_test.shape)
    print("Shape of y_train:", y_train.shape)
    print("Shape of y_val:", y_val.shape)
    # print("Shape of y_test:", y_test.shape)

    input_layer_dim = X_train.shape[1] # (avoids accidental reference to global X)  #len(X[0])

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
        "compile_loss": [{'hosts': 'categorical_crossentropy'}],
        "compile_metrics" :[{'hosts': 'accuracy'}]
    }

    # Call function to train all the models from the dictionary
    model, history = train_models(model_to_test, savedir, X_val, y_val)
    histories.append(history)

    print(X_test.shape)

    # predict the unseen dataset/new dataset
    y_predicted = model.predict(X_test)

    # change the dimension of y_test to array
    y_test = np.asarray(y_test)
    # y_test = y_test.reshape(y_test.shape[0], -1) # safer alternative to preserve the shape/squeeze 
    # (n_samples, n_classes) consistently.
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

    visualize(histories, savedir, model_name, str(fold), classes_default, y_predicted, y_test)
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

visualize(1, savedir, model_name, "Averaged_training", classes_default, save_predicted, save_true)

end_time = time()
print('Run time : {} s'.format(end_time-start_time))
print('Run time : {} m'.format((end_time-start_time)/60))
print('Run time : {} h'.format((end_time-start_time)/3600))

# %%
# combine all dictionaries together for the base model training (using Ifakara data)

combn_dictionar = combine_dictionaries(averaged_histories)

with open(
    os.path.join("..", "Results", "combined_history_dictionaries_base_model.txt"),
    'w'
    ) as outfile:
    json.dump(
        combn_dictionar, 
        outfile
        )

# with open(
#     os.path.join("..", "Results", "combined_history_dictionaries_base_model.txt"), 
#     ) as json_file:
#     combn_dictionar = json.load(json_file)

# find the average of all dictionaries 

combn_dictionar_average = find_mean_from_combined_dicts(combn_dictionar)

# Plot averaged histories

sns.set(
        context = "paper",
        style = "white",
        palette = "deep",
        font_scale = 1.5,
        color_codes = True,
        rc = ({"font.family": "Dejavu Sans"})
    )

graph_history_averaged(combn_dictionar_average)

# %%
