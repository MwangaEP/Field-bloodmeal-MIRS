# =============================================================================
# Blood Meal Source Classification (Human vs Bovine)
# MLP base model with K-fold cross-validation
# This script trains the base model that will be used for transfer learning.
# =============================================================================

#%%
# =============================================================================
# Import 
# =============================================================================

import os
import json
import joblib
from time import time

import numpy as np 
import pandas as pd
from collections import Counter 

import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import (train_test_split, KFold)
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow import keras
from keras import optimizers, initializers, regularizers
from keras.regularizers import *
from keras.models import Model, load_model
from keras.layers import (
    Input,
    Conv1D, 
    Dense, 
    Dropout, 
    MaxPooling1D,
    Flatten, 
    BatchNormalization
)
from keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau
from keras import backend as K

from MLP_functions import (
    build_folder, 
    log_data, 
    visualize, 
    graph_history, 
    graph_history_averaged,
    combine_dictionaries, 
    find_mean_from_combined_dicts
)

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

# =============================================================================
# 1. Model architecture
# =============================================================================
# Layer config reference:
#   Dense layer:        {'type': 'd', 'width': <int>}
#   Conv1D layer:       {'type': 'c', 'filter': <int>, 'kernel': <int>,
#                        'stride': <int>, 'pooling': <int>}
#
# Commented alternatives (uncomment to experiment):
# model_size = [
#     {'type': 'c', 'filter': 8, 'kernel': 2, 'stride': 1, 'pooling': 1},
#     {'type': 'c', 'filter': 8, 'kernel': 2, 'stride': 1, 'pooling': 1},
#     {'type': 'c', 'filter': 8, 'kernel': 2, 'stride': 1, 'pooling': 1},
#     {'type': 'd', 'width': 500},
#     {'type': 'd', 'width': 500},
#     {'type': 'd', 'width': 500},
# ]

MODEL_SIZE = [
    {"type": "d", "width": 500},
    {"type": "d", "width": 500},
    {"type": "d", "width": 500},
]
MODEL_NAME = "Baseline_MLP"

def create_models(model_shape, input_layer_dim):
    """Build and compile a Keras MLP (or CNN) model from a layer-spec list.
 
    Args:
        model_shape: list of layer-spec dicts (see config reference above).
        input_layer_dim: number of input features.
 
    Returns:
        Compiled Keras Model.
    """

    REG_CONST = 0.02
    
    # optimizer
    sgd = optimizers.SGD(
        learning_rate = 0.001, momentum = 0.9, nesterov = True, clipnorm = 1.0
    )
    
    # change the input shape to switch from 1D-CNN to MLP. By changing the input shape to 
    # (input_layer_dim, ) it will learn some combination of features with the learnable 
    # weights of the network

    input_vec = Input(name = 'input', shape = (input_layer_dim, )) 
    xd = None

    for i, layer_spec in enumerate(model_shape):
        layer_input = input_vec if i == 0 else xd

        if layer_spec['type'] == 'c':
            xd = Conv1D(
                name=f"Conv{i + 1}",
                filters=layer_spec['filter'], 
                kernel_size = layer_spec['kernel'], 
                strides = layer_spec['stride'],
                activation = 'relu',
                kernel_regularizer = regularizers.l2(REG_CONST), 
                kernel_initializer = 'he_normal',
            )(layer_input)
            xd = BatchNormalization(name=f"batchnorm_{i + 1}")(xd)
            xd = MaxPooling1D(
                name=f"maxpool_{i + 1}",
                pool_size=(layer_spec['pooling'])
            )(xd)
            
        elif layer_spec['type'] == 'd':

            # Flatten if transitioning from Conv → Dense
            if i > 0 and model_shape[i - 1]["type"] == "c":
                xd = tf.keras.layers.Flatten()(xd)
            else:
                xd = layer_input
 
            xd = tf.keras.layers.Dense(
                name=f"d{i + 1}",
                units=layer_spec["width"],
                activation="relu",
                kernel_regularizer=regularizers.l2(REG_CONST),
                kernel_initializer="he_normal",
            )(xd)

            xd = BatchNormalization(name=f"batchnorm_{i + 1}")(xd)
            xd = Dropout(name=f"dout{i + 1}", rate=0.5)(xd)

    # Output layer: 2 units (Bovine/Human) with softmax
    output = Dense(
        name = 'host_group', 
        units = 2, 
        activation = 'softmax', 
        kernel_regularizer = regularizers.l2(REG_CONST), 
        kernel_initializer = 'he_normal'
    )(xd)

    model = Model(inputs = input_vec, outputs = output)
    
    # Compile model
    model.compile(
        loss = 'categorical_crossentropy', 
        metrics = ['accuracy'], 
        optimizer=sgd
    )


    model.summary()
    return model


def train_model(
        model_shape, 
        model_name, 
        model_ver_num, 
        fold_num, 
        X_train, 
        y_train, 
        X_val, 
        y_val, 
        save_path
    ):
    """
    Create, fit, and save one fold's model.
 
    Validation data is passed explicitly to avoid reliance on outer-scope
    variables.
 
    Args:
        model_shape:    layer-spec list passed to create_model().
        model_name:     string prefix for saved file names.
        model_ver_num:  version integer appended to file names.
        fold_num:       current fold index (for file naming).
        X_train:        scaled training features.
        y_train:        one-hot training labels.
        X_val:          scaled validation features.
        y_val:          one-hot validation labels.
        save_path:      directory where model and CSV log are written.
 
    Returns:
        (model, history) tuple.
    """

    input_layer_dim = X_train.shape[1]
    model = create_models(model_shape, input_layer_dim)

    # Prepare model save path
    # csv_log_path = os.path.join(
    #     save_path, f"{model_name}_{model_ver_num}.csv"
    # )
    model_save_path = os.path.join(
        save_path, f"{model_name}_{model_ver_num}_{fold_num}_Model.keras"
    )

    # callbacks
    callbacks = [
        EarlyStopping(
            monitor = 'val_loss',
            patience = 100,                  
            restore_best_weights = True,
            verbose = 1,
            mode = 'auto'
        ),
        # Halves LR after 80 epochs of no val_loss improvement.
        # Allows the model to escape plateaus before early stopping
        # triggers, often recovering 1–3% generalisation accuracy.

        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.8,
            patience=80,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    history = model.fit(
        x = X_train, 
        y = y_train,
        batch_size = 256, 
        verbose = 1, 
        epochs = 3000,
        validation_data = (X_val, y_val),
        callbacks = callbacks
            )

    model.save(model_save_path)
    
    graph_history(history, model_name, model_ver_num, str(fold_num), save_path)
            
    return model, history

#%%
# =============================================================================
# 2. Load and prepare lab (training) data
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

print('Class distribution', Counter(blood_meal_df['blood_meal']))

# Select features and lebels
X_raw = np.asarray(blood_meal_df.iloc[:, 1:])
y_raw = np.asarray(blood_meal_df["blood_meal"])

# One-hot encode labels with explicit class ordering so index 0 = Bovine,
# index 1 = Human consistently across all folds and future datasets.
MLB_CLASSES = [["Bovine", "Human"]]
mlb = MultiLabelBinarizer(classes=MLB_CLASSES[0])
host_list = [[host] for host in y_raw]
y_onehot = mlb.fit_transform(host_list)

CLASS_NAMES = list(mlb.classes_)   # ['Bovine', 'Human']
print("Class mapping:", dict(zip(CLASS_NAMES, range(len(CLASS_NAMES)))))
print("One-hot shape:", y_onehot.shape)

#%%
# =============================================================================
# 3. Output directories
# =============================================================================

# outdir = os.path.join("..", "Results")

# savedir = (outdir + r"\MLP")            
# build_folder(savedir, False)
# savedir = (outdir + r"\MLP\l") 

outdir  = os.path.join("..", "Results")
savedir = os.path.join(outdir, "MLP")
build_folder(outdir, False)
build_folder(savedir, False)

#%%
# =============================================================================
# 4. K-fold cross-validation training
# =============================================================================

NUM_FOLDS        = 5
VALIDATION_SPLIT = 0.1
MODEL_VER_NUM    = 0
random_seed      = np.random.randint(0, 81470)
 
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=random_seed)


averaged_histories = []
save_predicted = []
save_true = []

start_time = time()

for fold_num, (train_index, test_index) in enumerate(
    kf.split(X_raw, y_onehot), start = 1):

    print(f"\n{'=' * 60}")
    print(f"Fold {fold_num} / {NUM_FOLDS}")
    print(f"{'=' * 60}")

    X_trainset_raw, X_test_raw = X_raw[train_index], X_raw[test_index]
    y_trainset, y_test = y_onehot[train_index], y_onehot[test_index]

    # --- Per-fold scaling (fit on train only, transform both splits) -------
    # This prevents test-fold statistics from leaking into the scaler and
    # ensures each fold is an independent experiment.

    fold_scaler = StandardScaler().fit(X_trainset_raw)
    X_trainset  = fold_scaler.transform(X_trainset_raw)
    X_test      = fold_scaler.transform(X_test_raw)

    # Save the fold-1 scaler as the canonical scaler for deployment
    if fold_num == 1:
        joblib.dump(
            fold_scaler,
            os.path.join("..", "Results", "MLP", "data_scaler.joblib"),
        )
        print("Saved fold-1 scaler → data_scaler.joblib")

    # --- Train / validation split within training fold --------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainset, y_trainset, 
        test_size = VALIDATION_SPLIT, 
        random_state = 42
    )

    # reshape to (n_samples, n_features) — required for Dense input
    X_train = X_train.reshape([X_train.shape[0], -1])
    X_val = X_val.reshape([X_val.shape[0], -1])
    X_test = X_test.reshape([X_test.shape[0], -1])

    print(f"X_train: {X_train.shape} | X_val: {X_val.shape} | X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape} | y_val:  {y_val.shape}")

     # --- Train ------------------------------------------------------------
    model, history = train_model(
        model_shape=MODEL_SIZE,
        model_name=MODEL_NAME,
        model_ver_num=MODEL_VER_NUM,
        fold_num=fold_num,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        save_path=savedir,
    )

    # --- Predict on held-out test fold ------------------------------------
    y_predicted = model.predict(X_test)

    y_test = np.squeeze(y_test)
 
    save_predicted.extend(y_predicted)
    save_true.extend(y_test)
 
    averaged_histories.append(history.history)

    # Per fold confusion matrix

    visualize(
        [history], savedir, MODEL_NAME, 
        str(fold_num), [CLASS_NAMES], y_predicted, y_test
    )
    
    # Clear the Keras session before freeing the model to avoid graph
    # accumulation across folds.

    K.clear_session()
    del model

# =============================================================================
# 5. Averaged confusion matrix across all folds
# =============================================================================
 
save_predicted = np.asarray(save_predicted)
save_true      = np.asarray(save_true)
 
print(f"\nAll-fold predicted shape: {save_predicted.shape}")
print(f"All-fold true shape:        {save_true.shape}")
 
visualize(
    1, savedir, MODEL_NAME,
    "Averaged_training", [CLASS_NAMES], save_predicted, save_true,
)
 
end_time = time()
elapsed  = end_time - start_time
print(f"\nTotal run time: {elapsed:.1f} s  "
      f"({elapsed / 60:.2f} min  /  {elapsed / 3600:.3f} h)")


#%%

# =============================================================================
# 6. Training history summary and plots
# =============================================================================

# combine all histories and save to file for future reference and plotting

combined_dictionar = combine_dictionaries(averaged_histories)

with open(
    os.path.join("..", "Results", "MLP", "combined_history_base_model.txt"),
'w') as outfile:
    json.dump(combined_dictionar, outfile)

history_mean = find_mean_from_combined_dicts(combined_dictionar)
graph_history_averaged(history_mean)

#%%