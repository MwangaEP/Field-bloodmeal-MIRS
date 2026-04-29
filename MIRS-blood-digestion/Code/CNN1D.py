# =============================================================================
# CNN Blood Meal Classification — Base Model (6 hr)
# =============================================================================
# Fixes applied vs previous version:
#   1. StandardScaler now fit inside CV loop (no leakage)
#   2. Data augmentation enabled (noise + amplitude scaling)
#   3. SpatialDropout1D added after each Conv block
#   4. L2 regularisation increased on Dense layers (0.02 → 0.05)
#   5. Dense block layer order corrected: Dense → BN → Dropout
#   6. EarlyStopping patience reduced (30 → 15)
#   7. ReduceLROnPlateau added (halves LR after 8 stagnant epochs)
#   8. Fixed random seed for reproducibility
# =============================================================================
 
#%%

# ----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

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
from keras.layers import Input, Concatenate
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import BatchNormalization, LayerNormalization
from keras.layers import Conv1D, MaxPooling1D, SpatialDropout1D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.models import model_from_json, load_model
from keras.regularizers import *
from keras import backend as K
 
import matplotlib.pyplot as plt
import seaborn as sns
 
sns.set(
    context="paper",
    style="white",
    palette="deep",
    font_scale=2.0,
    color_codes=True,
    rc=({"font.family": "Dejavu Sans"})
)
 
plt.rcParams["figure.figsize"] = [6, 4]

#%%

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
 
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
rn.seed(RANDOM_SEED)

#%%

# =============================================================================
# Model definition
# =============================================================================
 
# Regularisation constants
# Conv layers use a lighter penalty; Dense layers use a stronger one to
# counter the higher overfitting risk in the fully-connected part.
CONV_REG  = 0.02
DENSE_REG = 0.05

def create_models(model_shape, input_layer_dim):

    """
    Build a 1-D CNN for spectra blood-meal identification
    
    
    Parameters
    ----------  
    model_shape: list of dict
        Architecture definition. Each dict describes one layer:
          Conv  → {'type':'c', 'filter':int, 'kernel':int,
                   'stride':int, 'pooling':int}
          Dense → {'type':'d', 'width':int}
    input_layer_dim: int
        Number of spectral wavelengths (input features).

    
    Returns
    -------
    Keras.model (compiled)
    """
    
    sgd = optimizers.SGD(
        learning_rate = 0.001, 
        momentum = 0.9, 
        nesterov = True, 
        clipnorm = 1.0
        )

    # ------------------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------------------

    input_vec = Input(name = 'input', shape = (input_layer_dim, 1)) # (wavelengths, channels)  
    xd = input_vec

    # ------------------------------------------------------------------------------
    # Feature extractor
    # ------------------------------------------------------------------------------

    for i, layer_cfg in enumerate(model_shape):

        if layer_cfg['type'] == 'c':
            # --------------------Conv1D block -------------------------------------
            # Order: Conv1D (with relu) → BatchNorm → MaxPool
            #        → SpatialDropout1D
            # SpatialDropout1D drops entire feature maps rather than
            # individual activations, which is far more effective at
            # regularising convolutional layers on spectral data.

            xd = Conv1D(
            filters = layer_cfg['filter'],
            kernel_size = layer_cfg['kernel'],
            strides = layer_cfg['stride'],
            padding = 'same',
            activation = 'relu',
            kernel_initializer = 'he_normal',
            kernel_regularizer = regularizers.l2(CONV_REG),
            name = f'Conv{i+1}'
            )(xd)

            xd = BatchNormalization(name = f'BN_conv{i+1}')(xd)

            xd = MaxPooling1D(
                pool_size = layer_cfg['pooling'],
                name = f'MaxPool{i+1}'
            )(xd)

            # SpatialDropout1D: drops entire channels (feature maps)
            # rather than individual time-steps — stronger regulariser
            # for 1-D CNNs on spectral data.

            xd = SpatialDropout1D(
                rate = 0.2,
                name = f'SpatialDrop{i+1}'
            )(xd)
        
        elif layer_cfg['type'] == 'd':
            # --------------------Dense block --------------------------------------
            # Flatten first if we are transitoning from Conv layers
            if i > 0 and model_shape[i-1]['type'] == 'c':
                xd = Flatten(name = f'Flatten{i+1}')(xd)

            # Order: Dense → BatchNorm → Dropout
            # BatchNorm before Dropout normalises activations first;
            # Dropout then stochastically zeroes some of those
            # normalised values, which is the more stable ordering.

            xd = Dense(
                units = layer_cfg['width'],
                activation = 'relu',
                kernel_initializer = 'he_normal',
                kernel_regularizer = regularizers.l2(DENSE_REG),
                name = f'Dense{i+1}'
            )(xd)

            xd = BatchNormalization(name = f'BN_dense{i+1}')(xd)
            xd = Dropout(rate = 0.4, name = f'Dropout{i+1}')(xd)

    # ------------------------------------------------------------------
    # Output head — binary (Bovine vs Human)
    # Kept as Dense(2) so the head can be replaced cleanly during
    # transfer learning without touching the feature extractor.
    # ------------------------------------------------------------------        

    output = Dense(
        units = 2,
        activation = 'softmax',
        kernel_initializer = 'he_normal',
        kernel_regularizer = regularizers.l2(DENSE_REG),
        name = 'output_head'
    )(xd)

    model = Model(inputs = input_vec, outputs = output)

    # Compile model
    model.compile(
        loss = 'categorical_crossentropy', 
        metrics = ['accuracy'], 
        optimizer = sgd
    )

    model.summary()
    return model

    
#%%

# =============================================================================
# Load data
# =============================================================================

blood_meal_df = pd.read_csv(
    os.path.join("..", "Data", "Blood_meal_lab.dat"), 
    delimiter= '\t'
    )

blood_meal_df['Cat3'] = blood_meal_df['Cat3'].str.replace('BF', 'Bovine')
blood_meal_df['Cat3'] = blood_meal_df['Cat3'].str.replace('HF', 'Human')

# Drop unused columns
blood_meal_df = blood_meal_df.drop(['Cat1', 'Cat2', 'Cat5'], axis=1)
blood_meal_df.rename(columns = {'Cat3':'blood_meal'}, inplace = True) 

print('Class counts', Counter(blood_meal_df['blood_meal']))

# features and labels 

# define X (matrix of features) and y (list of labels)
X_raw = np.asarray(blood_meal_df.iloc[:,1:]) 
y = np.asarray(blood_meal_df["blood_meal"])

# One-hot encode labels
host_list = [[host] for host in y]
hosts = MultiLabelBinarizer().fit_transform(host_list)
y_classes = list(np.unique(host_list))
print('Classes:', y_classes)


# Labels default - all classification
labels_default = [hosts]
classes_default =  [y_classes]
outputs_default = ['output_head']

# # Scale data
# scaler = StandardScaler().fit(X = X)
# X_trans = scaler.transform(X = X) # transform X

# # Serialize the scaler to a file using joblib

# joblib.dump(
#     scaler, 
#     os.path.join("..", "Results", "data_scaler_cnn.joblib")
#     )

#%%
# =============================================================================
# Augmentation
# =============================================================================
 
def augment_spectra(X, noise_std=0.003, scale_range=0.05, seed=None):
    """
    Augment spectral data by adding Gaussian noise and random amplitude
    scaling.  Apply to training data only — never to val or test.
 
    Parameters
    ----------
    X          : ndarray, shape (n_samples, n_wavelengths, 1)
    noise_std  : std of additive Gaussian noise
    scale_range: ± fractional amplitude perturbation
    seed       : optional int for reproducibility within a fold
 
    Returns
    -------
    Augmented copy of X with the same shape.
    """
    rng   = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=noise_std, size=X.shape)
    scale = rng.uniform(
        1.0 - scale_range,
        1.0 + scale_range,
        size=(X.shape[0], 1, 1)
    )
    return (X + noise) * scale
 
#%%
# =============================================================================
# Training function
# =============================================================================

def train_models(model_to_test, save_path, X_val, y_val):
    """
    Instantiate, train, and persist a model for one CV fold.
 
    Parameters
    ----------
    model_to_test : dict   — fold metadata and hyperparameters
    save_path     : str    — directory for outputs
    X_val         : array  — validation features  (N, W, 1)
    y_val         : array  — validation labels     (N, 2)
 
    Returns
    -------
    model   : trained keras.Model
    history : keras History object
    """

    model_shape = model_to_test["model_shape"][0]
    model_name = model_to_test["model_name"][0]
    input_layer_dim = model_to_test["input_layer_dim"][0]
    model_ver_num = model_to_test["model_ver_num"][0]
    fold = model_to_test["fold"][0]
    y_train = model_to_test["labels"][0]
    X_train = model_to_test["features"][0]

    model = create_models(model_shape, input_layer_dim)
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor = 'val_loss',
            patience = 15,                  # reduced from 30 -stops ealier
            restore_best_weights = True,
            verbose = 1,
            mode = 'auto'
        ),
        # Halves LR after 8 epochs of no val_loss improvement.
        # Allows the model to escape plateaus before early stopping
        # triggers, often recovering 1–3% generalisation accuracy.

        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1
        ),

        CSVLogger(
            save_path + model_name + "_" + str(model_ver_num) + '.csv',
            append = True,
            separator = ';'
        )
    ]
    
    history = model.fit(
        x = X_train, 
        y = y_train,
        batch_size = 32, 
        verbose = 1, 
        epochs = 500,
        validation_data = (X_val, y_val),
        callbacks = callbacks
    )

    model.save(
        save_path 
        + model_name + "_" 
        + str(model_ver_num) + "_"
        + str(fold) + "_" 
        + 'Model.keras'
    )
    
    graph_history(history, model_name, model_ver_num, fold, save_path)
            
    return model, history

#-----------------------------------------------------------------------------------
# Main training loop
#-----------------------------------------------------------------------------------

outdir = os.path.join("..", "Results")

savedir = (outdir + r"\CNN")            
build_folder(savedir, False)
savedir = (outdir + r"\CNN\l") 
# ------------------------------------------------------------------
# Architecture
# ------------------------------------------------------------------

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
    {'type':'d', 'width':50},
]

# # Name the model
model_name = 'CNN_base_6hr'
# label = labels_default
# features = X_trans 

# ------------------------------------------------------------------
# Cross-validation
# ------------------------------------------------------------------

num_folds = 5
kf = KFold(n_splits = num_folds, shuffle = True, random_state =RANDOM_SEED)
    
histories          = []
averaged_histories = []
fold               = 1

# start model training on standardized data

start_time = time()
save_predicted = []
save_true = []

for train_index, test_index in kf.split(X_raw):

    print(f'\n{"="*60}')
    print(f'  Fold {fold} / {num_folds}')
    print(f'{"="*60}')
 
    # ------------------------------------------------------------------
    # FIX 1 — Scale inside the fold (no leakage)
    # Fit scaler only on training rows; apply to val and test.

    X_trainset_raw, X_test_raw = X_raw[train_index], X_raw[test_index]

    scaler = StandardScaler().fit(X_trainset_raw)
    X_trainset = scaler.transform(X_trainset_raw)
    X_test     = scaler.transform(X_test_raw)
 
    # Save the fold-1 scaler as the canonical scaler for deployment
    if fold == 1:
        joblib.dump(
            scaler,
            os.path.join('..', 'Results', 'data_scaler_cnn.joblib')
        )
    
    y_trainset = list(map(lambda y:y[train_index], labels_default))
    y_test     = list(map(lambda y:y[test_index], labels_default))


    # Further divide training dataset into train and validation dataset 
    # 90 / 10 train / validation split
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainset,
        *y_trainset, 
        test_size = 0.1,
        random_state = RANDOM_SEED
    )

    # Reshape to (N, wavelengths, 1) for Conv1D
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # ------------------------------------------------------------------
    # FIX 2 — Data augmentation (training set only)
    # Doubles the training set with noise + amplitude perturbations.
    # Augmented samples get the same labels as their originals.
    # ------------------------------------------------------------------
    # X_train_aug = augment_spectra(
    #     X_train,
    #     noise_std=0.003,
    #     scale_range=0.05,
    #     seed=RANDOM_SEED + fold   # deterministic but fold-unique
    # )
    # X_train = np.concatenate([X_train, X_train_aug], axis=0)
    # y_train = np.concatenate([y_train, y_train],     axis=0)
 
    # Check the sizes of all newly created datasets
    print(f'  {X_train.shape}') #  (incl. augmented)
    print(f'  {X_val.shape}')
    print(f'  {X_test.shape}')


    input_layer_dim = X_train.shape[1] # (avoids accidental reference to global X)  #len(X[0])

    model_to_test = {
        "model_shape"     : [model_size], 
        "model_name"      : [model_name],
        "input_layer_dim" : [input_layer_dim], # size of input layer
        "model_ver_num"   : [0],
        "fold"            : [fold], # kf.split number on
        "labels"          : [y_train],
        "features"        : [X_train],
        "classes"         : [classes_default],
        "outputs"         : [outputs_default],
        "compile_loss"    : [{'output_head: categorical_crossentropy'}],
        "compile_metrics" : [{'output_head: accuracy'}]
    }

    model, history = train_models(model_to_test, savedir, X_val, y_val)
    histories.append(history)

    #-------------------------------------------------------------------
    # Evaluate on held-out test fold
    # ------------------------------------------------------------------

    y_predicted = model.predict(X_test)

    y_test = np.squeeze(np.asarray(y_test)) # remove any single dimension entries from the arrays
   
    print(f' y_predicted shape : {y_predicted.shape}')
    print(f' y_test shape      : {y_test.shape}')

    # save predicted and true value in each iteration

    for pred, tru in zip(y_predicted, y_test):
        save_predicted.append(pred)
        save_true.append(tru)

    hist = history.history
    averaged_histories.append(hist)

    # Plotting confusion matrix for each fold/iteration

    visualize(
        histories, savedir, model_name, 
        str(fold), classes_default, y_predicted, y_test
    )
    
    fold += 1

    # Clear the Keras session, otherwise it will keep adding new
    # models to the same TensorFlow graph each time we create
    # a model with a different set of hyper-parameters.

    K.clear_session()
    # Delete the Keras model with these hyper-parameters from memory.
    del model


#------------------------------------------------------------------
# Aggregate results
# -----------------------------------------------------------------
 
save_predicted = np.asarray(save_predicted)
save_true      = np.asarray(save_true)
 
print(f'\nAll-fold predicted shape : {save_predicted.shape}')
print(f'All-fold true shape      : {save_true.shape}')

# Averaged confusion matrix across all folds
visualize(
    1, savedir, model_name,
    "Averaged_training", classes_default,
    save_predicted, save_true
    )

end_time = time()
elapsed = end_time - start_time
print(f'\nRun time : {elapsed:.1f} s  |  '
      f'{elapsed/60:.2f} min  |  '
      f'{elapsed/3600:.3f} hr')

# %%

# =============================================================================
# Persist and plot training histories
# =============================================================================

combn_dictionar = combine_dictionaries(averaged_histories)

with open(
    os.path.join("..", "Results", "combined_history_dictionaries_base_model.txt"),
    'w'
    ) as outfile:
    json.dump(combn_dictionar, outfile)

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
