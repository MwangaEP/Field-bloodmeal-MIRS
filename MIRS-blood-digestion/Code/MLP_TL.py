#%%
# Import libraries

import os
import io
import json
import joblib
from time import time

from itertools import cycle
import random as rn

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
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
)

from sklearn.metrics import accuracy_score

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

# set working directory

outdir = os.path.join("..", "Results")
build_folder(outdir, False)  

model_name = 'Baseline_CNN'
# label = labels_default
    
# Split data into 10 folds for training/testing
# Define cross-validation strategy 

num_folds = 5
random_seed = np.random.randint(0, 81470)
kf = KFold(n_splits = num_folds, shuffle = True, random_state = random_seed)
train_model = True

# Name a folder for the outputs to go into

savedir = (outdir + r"\transfer_learning_2")            
build_folder(savedir, False)
savedir = (outdir+ r"\transfer_learning_2\l")

#%%
# Load the scaler used to scale the training data
scaler = joblib.load(os.path.join("..", "Results", "data_scaler.joblib"))

# Load the important wavenumbers identified by SVC model
# load SVC important wavenumbers from previous analysis 
with open(
    os.path.join("..", "Results", "top_svc_coefficients_wavenumbers.txt"), 
    ) as json_file:
    sn_wn = json.load(json_file)

# Load the trained MLP model from disk
clf = tf.keras.models.load_model(os.path.join("..", "Results", "lBaseline_CNN_0_3_Model.keras"))

#%%
# Making prediction on the blood meal hours data
# Load hours blood meal data

blood_hours_df = pd.read_csv(
    os.path.join("..", "Data", "Bloodfed_hours.dat"), 
    delimiter = '\t'
)

# Rename items in the column
blood_hours_df['Cat3'] = blood_hours_df['Cat3'].str.replace('CW', 'Bovine')
blood_hours_df['Cat3'] = blood_hours_df['Cat3'].str.replace('HN', 'Human')

# view the data
blood_hours_df.head()

#%%

# count the number of blood hours post feeding
Counter(blood_hours_df['Cat4'])

#%% 
# filter data with blood meal hours (To be used for model testing)
blood_6hours = blood_hours_df[blood_hours_df['Cat4'] == '6H']
blood_12hours = blood_hours_df[blood_hours_df['Cat4'] == '12H']
blood_24hours = blood_hours_df[blood_hours_df['Cat4'] == '24H']
blood_48hours = blood_hours_df[blood_hours_df['Cat4'] == '48H']

#%%
# 6 hours data
# define the features and target variable
X_6h = blood_6hours.drop(
    [
        'Cat1', 
        'Cat2',
        'Cat3', 
        'Cat4', 
        'StoTime'
    ], 
    axis = 1
)

X_6h = X_6h.loc[:, sn_wn]  # select only important wavenumbers identified by SVC

# target variable
y_6h = blood_6hours['Cat3']

# scale the data
X_6h_scl = scaler.transform(np.asarray(X_6h))

#%%
# 12 hours data
# define the features and target variable
X_12h = blood_12hours.drop(
    [
        'Cat1', 
        'Cat2',
        'Cat3', 
        'Cat4', 
        'StoTime'
    ], 
    axis = 1
)

X_12h = X_12h.loc[:, sn_wn]  # select only important wavenumbers identified by SVC

# target variable
y_12h = blood_12hours['Cat3']

# scale the data
X_12h_scl = scaler.transform(np.asarray(X_12h))

#%%
# 24 hours data
# define the features and target variable
X_24h = blood_24hours.drop(
    [
        'Cat1', 
        'Cat2',
        'Cat3', 
        'Cat4', 
        'StoTime'
    ], 
    axis = 1
)

X_24h = X_24h.loc[:, sn_wn]  # select only important wavenumbers identified by SVC

# target variable
y_24h = blood_24hours['Cat3']

# scale the data
X_24h_scl = scaler.transform(np.asarray(X_24h))

#%%
# 48 hours data
# define the features and target variable
X_48h = blood_48hours.drop(
    [
        'Cat1', 
        'Cat2',
        'Cat3', 
        'Cat4', 
        'StoTime'
    ], 
    axis = 1
)

X_48h = X_48h.loc[:, sn_wn]  # select only important wavenumbers identified by SVC

# target variable
y_48h = blood_48hours['Cat3']

# scale the data
X_48h_scl = scaler.transform(np.asarray(X_48h))

#%%

# Create a function to predict blood meal hours using the trained MLP model. 

def predict_bloodmeal_hours(X_data, y_data, model):
    
    """
    Predicts blood meal source using trained model
    Parameters:
    X_data (array-like): Scaled feature data
    y_data (array-like): True labels
    model (keras.Model): Trained Keras model for prediction

    Returns:
    dict: Dictionary containing accuracy and classification report

    """
    # prepare Y_values for the transfer learning - training

    host_list = [[host] for host in y_data]
    hosts_trans = MultiLabelBinarizer().fit_transform(host_list)
    y_classes = list(np.unique(host_list))
    print('Y classes ', y_classes)
    # print('y classes binarized', hosts_trans)

    labels_default, classes_default = [hosts_trans], [y_classes]

    # reshape data to match the training shape
    X_data_2 = X_data.reshape([X_data.shape[0], -1])
    y_val = np.squeeze(labels_default) # remove any single dimension entries from the arrays
    # print(type(y_val))
    print('Y validation', y_val)
    y_predictions = model.predict(X_data_2)

    # computes the loss based on the X_input you passed, along with any other metrics requested in the metrics param 
    # when model was compiled
 
    score = model.evaluate(X_data_2, y_val, verbose = 1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # predictions = np.argmax(y_predictions, axis = -1)
    # y_true = np.argmax(y_val, axis = -1)

    return y_predictions, y_val, classes_default


#%%
# 6 hours

pred_6h, true_6h, classes_6h = predict_bloodmeal_hours(
    X_6h_scl, 
    np.asarray(y_6h), 
    clf
)

visualize(
    1,
    savedir, 
    "base_model",
    "6hrs_bloodmeal",
    classes_6h,
    pred_6h,
    true_6h
)

#%%

# 12 hours

pred_12h, true_12h, classes_12h = predict_bloodmeal_hours(
    X_12h_scl, 
    np.asarray(y_12h), 
    clf
)

visualize(
    1,
    savedir, 
    "base_model",
    "12hrs_bloodmeal",
    classes_12h,
    pred_12h,
    true_12h
)
    
#%%
# 24 hours

pred_24h, true_24h, classes_24h = predict_bloodmeal_hours(
    X_24h_scl, 
    np.asarray(y_24h), 
    clf
)

visualize(
    1,
    savedir, 
    "base_model",
    "24hrs_bloodmeal",
    classes_24h,
    pred_24h,
    true_24h
)

#%%
# 48 hours

pred_48h, true_48h, classes_48h = predict_bloodmeal_hours(
    X_48h_scl, 
    np.asarray(y_48h), 
    clf
)

visualize(
    1,
    savedir, 
    "base_model",
    "48hrs_bloodmeal",
    classes_48h,
    pred_48h,
    true_48h
)

#%%
# Prepare data for transfer learning, based on findings from mwanga et al., 2024,
# we will use 13% of the data for transfer learning from each dataset

# create a function that split data into training and test set, then return the two dataframes

def split_data(X, y, test_size = 0.85, random_state = 42):
    
    """
    Splits the data into training and test sets.

    Parameters:
    X (array-like): Feature data
    y (array-like): Target labels
    test_size (float): Proportion of the dataset to include in the test split
    random_state (int): Random seed for reproducibility

    Returns:
    train_set: dataframe containing training features and labels 
    test_set: dataframe containing test features and labels 
    
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size = test_size, 
        random_state = random_state, 
        shuffle = True
    )
    # combine X and y into a single dataframe for both training and test sets
    train_set = pd.DataFrame(X_train)
    train_set['blood_meal'] = y_train

    test_set = pd.DataFrame(X_test)
    test_set['blood_meal'] = y_test

    return train_set, test_set

#%%

# Extract the train and test splits from each blood meal hours dataset
train_6h, test_6h = split_data(X_6h, y_6h, test_size = 0.85, random_state = 42)
train_12h, test_12h = split_data(X_12h, y_12h, test_size = 0.85, random_state = 42)
train_24h, test_24h = split_data(X_24h, y_24h, test_size = 0.85, random_state = 42)
train_48h, test_48h = split_data(X_48h, y_48h, test_size = 0.85, random_state = 42)

#%%
# combine all train sets into a single dataframe

train_trans_df = pd.concat(
    [
        train_6h,
        train_12h,
        train_24h,
        train_48h
    ],
    axis = 0,
).reset_index(drop = True)

# combine all test sets into a single dataframe
test_trans_df = pd.concat(
    [
        test_6h,
        test_12h,
        test_24h,
        test_48h
    ],
    axis = 0,
).reset_index(drop = True)

# %%
# Prepare data by defining features and target variable for transfer learning

# train data
X_train_t = np.asarray(train_trans_df.drop('blood_meal', axis = 1))
y_train_t = np.asarray(train_trans_df['blood_meal'])

# validation data
X_val_t = np.asarray(test_trans_df.drop('blood_meal', axis = 1))
y_val_t = np.asarray(test_trans_df['blood_meal'])

#%%
# Create a function to train and evaluate transfer learning model

def train_evaluate_tf(model, scaler, X, y, X_val, y_val):
    
    """
    Train and evaluate transfer learning model

    Parameters:
    model (keras.Model): Pre-trained Keras model for transfer learning
    scaler: saved scaler object for data scaling
    X (array-like): Feature data for training
    y (array-like): Target labels for training
    X_val (array-like): Feature data for validation
    y_val (array-like): Target labels for validation
    
    Returns:
    history: Training history object

    """

    # Prepare the model for transfer learning

    inputs = model.input
    output = model.output
    transfer_lr_model = Model(inputs = inputs, outputs = output)

    sgd_tl = keras.optimizers.SGD(
        learning_rate = 0.00001, 
        # decay = 1e-5, 
        momentum = 0.9, 
        nesterov = True, 
        clipnorm = 1.
    )

    cce_tl = 'categorical_crossentropy'

    transfer_lr_model.compile(
        loss = cce_tl, 
        metrics = ['acc'], 
        optimizer = sgd_tl
    )

    # scale X and X_val
    X = scaler.transform(X)
    X_val = scaler.transform(X_val)

    # prepare Y_values for the transfer learning - training

    host_t = [[host] for host in y]
    hosts_t = MultiLabelBinarizer().fit_transform(host_t)
    y_classes_t = list(np.unique(host_t))
    labels_default_t, classes_default_t = [hosts_t], [y_classes_t]

    # prepare Y_values for the transfer learning - validation
    host_val = [[host] for host in y_val]
    hosts_val = MultiLabelBinarizer().fit_transform(host_val)
    y_classes_val = list(np.unique(host_val))
    labels_default_val, classes_default_val = [hosts_val], [y_classes_val]

    # reshape data, and train the model
    X = X.reshape([X.shape[0], -1])
    X_val = X_val.reshape([X_val.shape[0], -1])

    history_transfer_lr = transfer_lr_model.fit(
        x = X, 
        y = np.squeeze(labels_default_t),
        batch_size = 256, 
        verbose = 1, 
        epochs = 5000,
        validation_data = (X_val, np.squeeze(labels_default_val)),
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor = 'val_loss', 
                patience = 50, 
                verbose = 1, 
                mode = 'auto'
            ), 
            CSVLogger(
                savedir + 'transfer_logger.csv', 
                append = True, 
                separator = ';'
            )
        ]
    )
 
    transfer_lr_model.save(
        os.path.join(
            "..", "Results", 
            'transfer_lr_model.keras'
        )
    )

    return history_transfer_lr, transfer_lr_model


#%%

# Train and evaluate the transfer learning
start_time = time()

history, transfer_model = train_evaluate_tf(
    clf, 
    scaler, 
    X_train_t, 
    y_train_t, 
    X_val_t, 
    y_val_t
)

end_time = time()
print('Run time : {} s'.format(end_time-start_time))
print('Run time : {} m'.format((end_time-start_time)/60))
print('Run time : {} h'.format((end_time-start_time)/3600))

#%%
# Predict hourly blood meal source using the transfer learning model
# 6 hours data

X_test_6h = np.asarray(test_6h.drop('blood_meal', axis = 1))

# target variable
y_test_6h = np.asarray(test_6h['blood_meal'])

# scale the data
X_test_6h_scl = scaler.transform(X_test_6h)

# Predictions
pred_test_6h, true_test_6h, classes_test_6h = predict_bloodmeal_hours(
    X_test_6h_scl, 
    y_test_6h, 
    transfer_model  
)

# Visualize the confusion matrix
visualize(
    1,
    savedir, 
    "tl_model",
    "6hrs_bloodmeal",
    classes_test_6h,
    pred_test_6h,
    true_test_6h
)

# Calculate classification report
cr_6h = classification_report(
    np.argmax(true_test_6h, axis = -1), 
    np.argmax(pred_test_6h, axis = -1),
    target_names = ['Bovine', 'Human']
)

# save classification report to disk 
cr_6h = pd.read_fwf(io.StringIO(cr_6h), header = 0)
cr_6h = cr_6h.iloc[1:]
cr_6h.to_csv(
    os.path.join("..", 
                 "Results", 
                 "transfer_learning_2", 
                 "cr_report_TL_6h_bloodmeal.csv"
                 )
)

#%%
# 12 hours data
X_test_12h = np.asarray(test_12h.drop('blood_meal', axis = 1))

# target variable
y_test_12h = np.asarray(test_12h['blood_meal'])

# scale the data
X_test_12h_scl = scaler.transform(X_test_12h)

# Predictions
pred_test_12h, true_test_12h, classes_test_12h = predict_bloodmeal_hours(
    X_test_12h_scl, 
    y_test_12h, 
    transfer_model  
)

# Visualize the confusion matrix
visualize(
    1,
    savedir, 
    "tl_model",
    "12hrs_bloodmeal",
    classes_test_12h,
    pred_test_12h,
    true_test_12h
)

# Calculate classification report
cr_12h = classification_report(
    np.argmax(true_test_12h, axis = -1), 
    np.argmax(pred_test_12h, axis = -1),
    target_names = ['Bovine', 'Human']
)

# save classification report to disk 
cr_12h = pd.read_fwf(io.StringIO(cr_12h), header = 0)
cr_12h = cr_12h.iloc[1:]
cr_12h.to_csv(
    os.path.join("..", 
                 "Results", 
                 "transfer_learning_2", 
                 "cr_report_TL_12h_bloodmeal.csv"
                 )
)


#%%
# 24 hours data
X_test_24h = np.asarray(test_24h.drop('blood_meal', axis = 1))

# target variable
y_test_24h = np.asarray(test_24h['blood_meal'])

# scale the data
X_test_24h_scl = scaler.transform(X_test_24h)

# Predictions
pred_test_24h, true_test_24h, classes_test_24h = predict_bloodmeal_hours(
    X_test_24h_scl, 
    y_test_24h, 
    transfer_model  
)

# Visualize the confusion matrix
visualize(
    1,
    savedir, 
    "tl_model",
    "24hrs_bloodmeal",
    classes_test_24h,
    pred_test_24h,
    true_test_24h
)

# Calculate classification report
cr_24h = classification_report(
    np.argmax(true_test_24h, axis = -1), 
    np.argmax(pred_test_24h, axis = -1),
    target_names = ['Bovine', 'Human']
)

# save classification report to disk 
cr_24h = pd.read_fwf(io.StringIO(cr_24h), header = 0)
cr_24h = cr_24h.iloc[1:]
cr_24h.to_csv(
    os.path.join("..", 
                 "Results", 
                 "transfer_learning_2", 
                 "cr_report_TL_24h_bloodmeal.csv"
                 )
)

#%%
# 48 hours data
X_test_48h = np.asarray(test_48h.drop('blood_meal', axis = 1))

# target variable
y_test_48h = np.asarray(test_48h['blood_meal'])

# scale the data
X_test_48h_scl = scaler.transform(X_test_48h)

# Predictions
pred_test_48h, true_test_48h, classes_test_48h = predict_bloodmeal_hours(
    X_test_48h_scl, 
    y_test_48h, 
    transfer_model  
)

# Visualize the confusion matrix
visualize(
    1,
    savedir, 
    "tl_model",
    "48hrs_bloodmeal",
    classes_test_48h,
    pred_test_48h,
    true_test_48h
)

# Calculate classification report
cr_48h = classification_report(
    np.argmax(true_test_48h, axis = -1), 
    np.argmax(pred_test_48h, axis = -1),
    target_names = ['Bovine', 'Human']
)

# save classification report to disk 
cr_48h = pd.read_fwf(io.StringIO(cr_48h), header = 0)
cr_48h = cr_48h.iloc[1:]
cr_48h.to_csv(
    os.path.join("..", 
                 "Results", 
                 "transfer_learning_2", 
                 "cr_report_TL_48h_bloodmeal.csv"
                 )
)

#%%

# Plot training history for transfer learning model
def plot_tl_accuracy_history(history):
    plt.figure(figsize=(6, 4))

    plt.plot(history.history['acc'], label='accuracy')
    plt.plot(history.history['val_acc'], label='val_accuracy')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.yticks(np.arange(0.0, 1.0 + 0.05, step = 0.2))
    plt.legend(loc = 'lower right')
    plt.grid(False)

    plt.tight_layout()

    plt.savefig(
        os.path.join("..", 
                     "Results", 
                     "transfer_learning_2", 
                     "tl_accuracy_history.png"
                    ), 
        dpi = 500, 
        bbox_inches = "tight"
    )
    plt.show()

     

sns.set(
        context = "paper",
        style = "white",
        palette = "deep",
        font_scale = 1.5,
        color_codes = True,
        rc = ({"font.family": "Dejavu Sans"})
    )

plot_tl_accuracy_history(history)


#%%
# Collect all accuracies from all blood meal hours into a single dataframe

accuracy_df = pd.DataFrame({
    "Blood meal hours": ['6 hours', '12 hours', '24 hours', '48 hours'],
    "Accuracy": [np.round(accuracy_score(true_test_6h.argmax(axis=-1), pred_test_6h.argmax(axis=-1)), 2),
                 np.round(accuracy_score(true_test_12h.argmax(axis=-1), pred_test_12h.argmax(axis=-1)), 2),
                 np.round(accuracy_score(true_test_24h.argmax(axis=-1), pred_test_24h.argmax(axis=-1)), 2),
                 np.round(accuracy_score(true_test_48h.argmax(axis=-1), pred_test_48h.argmax(axis=-1)), 2)]
})

accuracy_df.to_csv(
    os.path.join("..",
                "Results", 
                "transfer_learning_2", 
                "tl_bloodmeal_hours_accuracies.csv"
            )
)

# %%

# Plot the results
plt.figure(figsize = (8, 4))

sns.barplot(
    x = 'Accuracy',
    y = 'Blood meal hours',
    data = accuracy_df,
    palette = 'colorblind',
    width = 0.5, 
    # orient='h'
)

# Add a horizontal dotted line at y=0.7
plt.axvline(0.7, color='gray', linestyle='--', linewidth=2)

# sns.despine(offset = 5, trim = False)
# plt.xticks(rotation = 90)
plt.xticks(np.arange(0.0, 1.0 + .1, step = 0.2))
plt.ylabel('Hours', weight = 'bold')
plt.xlabel("Accuracy", weight = 'bold')
# plt.legend().remove()
plt.tight_layout()

plt.savefig(
    os.path.join("..",
                 "Results", 
                "transfer_learning_2",
                "tl_bloodmeal_hours_accuracies.png"
                ), 
    dpi = 500, 
    bbox_inches = "tight"
)

# %%

