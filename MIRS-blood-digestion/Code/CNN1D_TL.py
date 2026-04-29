#%%
# ----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import os
import io
import joblib
from time import time

import numpy as np 
import pandas as pd

from collections import Counter 

# Custom utility functions
from MLP_functions import (
    build_folder, 
    log_data, 
    visualize, 
    graph_history, 
    graph_history_averaged,
    combine_dictionaries, 
    find_mean_from_combined_dicts
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    classification_report, 
    accuracy_score
)

import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten
from keras.layers import BatchNormalization
from keras.layers import Conv1D
from keras.optimizers import SGD
from keras.callbacks import CSVLogger
from keras.models import load_model

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

savedir = (outdir + r"\CNN_TL")            
build_folder(savedir, False)
savedir = (outdir+ r"\CNN_TL\l")

#%%
# Load the scaler used to scale the training data

scaler = joblib.load(
    os.path.join(
        "..", 
        "Results", 
        "data_scaler_cnn.joblib"
        )
    )

# Load the trained CNN model from disk
clf = load_model(
    os.path.join(
        "..", 
        "Results", 
        "CNN",
        "lCNN_base_6hr_0_2_Model.keras"
        )
    )

#%%
# =============================================================================
# Load data (blood meal by hours)
# =============================================================================

blood_hours_df = pd.read_csv(
    os.path.join(
        "..", 
        "Data", 
        "Bloodfed_hours.dat"
    ), 
    delimiter = '\t'
)

# Rename items in the column
blood_hours_df['Cat3'] = blood_hours_df['Cat3'].str.replace('CW', 'Bovine')
blood_hours_df['Cat3'] = blood_hours_df['Cat3'].str.replace('HN', 'Human')

# view the data
blood_hours_df.head()

#%%

# count the number of blood hours post feeding
print(f'Hours count: {Counter(blood_hours_df["Cat4"])}')

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

# target variable
y_48h = blood_48hours['Cat3']

# scale the data
X_48h_scl = scaler.transform(np.asarray(X_48h))

#%%

# Create a function to predict blood meal hours using the trained CNN model. 

def predict_bloodmeal_hours(X_data, y_data, model):
    """
    Predicts blood meal host source using a trained Keras model.

    Parameters:
    -----------
    X_data : array-like
        Scaled feature data (already transformed by the scaler).
    y_data : array-like
        True host labels (e.g., 'Bovine', 'Human').
    model : keras.Model
        Trained Keras model for prediction.

    Returns:
    --------
    y_predictions : np.ndarray
        Predicted class probabilities for each sample.
    y_val : np.ndarray
        One-hot encoded true labels.
    classes_default : list
        List containing the unique class names.
    """
    # Binarize labels using MultiLabelBinarizer for one-hot encoding
    host_list = [[host] for host in y_data]
    hosts_trans = MultiLabelBinarizer().fit_transform(host_list)
    y_classes = list(np.unique(host_list))
    print('Y classes:', y_classes)

    labels_default, classes_default = [hosts_trans], [y_classes]

    # Reshape input to (samples, timesteps, 1) as required by Conv1D
    X_data_2 = X_data.reshape(X_data.shape[0], X_data.shape[1], 1)

    # Remove redundant single-dimension entries from the label array
    y_val = np.squeeze(labels_default)

    # Generate predictions
    y_predictions = model.predict(X_data_2)

    # Evaluate model loss and accuracy on the provided data
    score = model.evaluate(X_data_2, y_val, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

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
# We will use 13% of the data for transfer learning from each dataset

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

    # -----------------------------------------------------------------------
    # Freeze the feature extractor (Conv blocks) — keep Dense blocks and
    # the output head trainable for fine-tuning.
    #
    # Name-based freezing is used because the base model assigns consistent
    # prefixes: 'Conv', 'BN_conv', 'MaxPool', 'SpatialDrop' for extractor
    # layers vs. 'Dense', 'BN_dense', 'Dropout', 'output_head' for the
    # classifier head.  Freezing by isinstance(BatchNormalization) would
    # incorrectly freeze BN_dense layers that should remain trainable.
    # -----------------------------------------------------------------------

    FROZEN_PREFIXES = ('Conv', 'BN_conv', 'MaxPool', 'SpatialDrop')

    for layer in model.layers:
        if any(layer.name.startswith(tag) for tag in FROZEN_PREFIXES):
            layer.trainable = False  # freeze feature extractor
        else:
            layer.trainable = True   # fine-tune Dense blocks + output head

    # Log trainability for inspection
    for layer in model.layers:
        print(f"  {layer.name:30s}  trainable={layer.trainable}")

    sgd_tl = SGD(
        learning_rate = 0.00001,
        momentum = 0.9,
        nesterov = True,
        clipnorm = 1.
    )

    # Use 'accuracy' (not 'acc') to match the base model compilation and
    # ensure history keys are consistent with plot_tl_accuracy_history.
    model.compile(
        loss = 'categorical_crossentropy',
        metrics = ['accuracy'],
        optimizer = sgd_tl
    )

    # Scale features using the pre-fitted scaler
    X = scaler.transform(X)
    X_val = scaler.transform(X_val)

    # One-hot encode training labels
    host_t = [[host] for host in y]
    hosts_t = MultiLabelBinarizer().fit_transform(host_t)
    labels_default_t = [hosts_t]

    # One-hot encode validation labels
    host_val = [[host] for host in y_val]
    hosts_val = MultiLabelBinarizer().fit_transform(host_val)
    labels_default_val = [hosts_val]

    # reshape data, and train the model
    X = X.reshape(X.shape[0], X.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

    history_transfer_lr = model.fit(
        x = X, 
        y = np.squeeze(labels_default_t),
        batch_size = 32, 
        verbose = 1, 
        epochs = 500,
        validation_data = (X_val, np.squeeze(labels_default_val)),
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor = 'val_loss', 
                patience = 30, 
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
 
    model.save(
        os.path.join(
            "..", 
            "Results", 
            "CNN_TL", 
            'CNN_TL_model.keras'
        )
    )

    return history_transfer_lr, model


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
# =============================================================================
# Predict blood meal host source at each digestion time point (6h, 12h, 24h, 48h)
# using the transfer learning model and save classification reports to disk
# =============================================================================

# Map each time point label to its corresponding held-out test DataFrame
hour_datasets = {
    "6h":  test_6h,
    "12h": test_12h,
    "24h": test_24h,
    "48h": test_48h,
}

# Stores (true, predicted) arrays per time point for later accuracy summary
predictions_store = {}

for label, test_df in hour_datasets.items():

    # Separate features and target from the test split
    X_test = np.asarray(test_df.drop('blood_meal', axis=1))
    y_test = np.asarray(test_df['blood_meal'])

    # Apply the same scaler used during base model training
    X_test_scl = scaler.transform(X_test)

    # Generate predictions using the transfer learning model
    pred, true, classes = predict_bloodmeal_hours(X_test_scl, y_test, transfer_model)

    # Store for accuracy summary later
    predictions_store[label] = (true, pred)

    # Plot and save confusion matrix
    visualize(
        1,
        savedir,
        "tl_model",
        f"{label}_bloodmeal",
        classes,
        pred,
        true
    )

    # Generate classification report (Bovine vs Human)
    cr = classification_report(
        np.argmax(true, axis=-1),
        np.argmax(pred, axis=-1),
        target_names=['Bovine', 'Human']
    )

    # Parse and save classification report as CSV
    cr_df = pd.read_fwf(io.StringIO(cr), header=0).iloc[1:]
    cr_df.to_csv(
        os.path.join("..", "Results", "CNN_TL", f"cr_report_TL_{label}_bloodmeal.csv")
    )
    print(f"Classification report saved for {label} time point.")

#%%
# =============================================================================
# Plot training history for the transfer learning model
# =============================================================================

def plot_tl_accuracy_history(history):
    """
    Plots and saves training vs. validation accuracy over epochs.

    Parameters:
    -----------
    history : keras.callbacks.History
        History object returned by model.fit().
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(history.history['accuracy'], label='accuracy')
    ax.plot(history.history['val_accuracy'], label='Val accuracy')

    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_yticks(np.arange(0.0, 1.05, step=0.2))
    ax.legend(loc='lower right')
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(
        os.path.join("..", "Results", "CNN_TL", "tl_accuracy_history.png"),
        dpi=500,
        bbox_inches="tight"
    )
    plt.show()


# Apply plot styling before calling the function
sns.set(
    context="paper",
    style="white",
    palette="deep",
    font_scale=1.5,
    color_codes=True,
    rc={"font.family": "Dejavu Sans"}
)

plot_tl_accuracy_history(history)


#%%
# =============================================================================
# Summarise per-timepoint accuracy and plot as a horizontal bar chart
# =============================================================================

# Build accuracy summary from predictions stored during the prediction loop
hour_labels = {"6h": "6 hours", "12h": "12 hours", "24h": "24 hours", "48h": "48 hours"}

accuracy_df = pd.DataFrame({
    "Blood meal hours": list(hour_labels.values()),
    "Accuracy": [
        np.round(
            accuracy_score(
                predictions_store[k][0].argmax(axis=-1),
                predictions_store[k][1].argmax(axis=-1)
            ), 2
        )
        for k in hour_labels
    ]
})

# Save accuracy summary to CSV
accuracy_df.to_csv(
    os.path.join("..", "Results", "CNN_TL", "tl_bloodmeal_hours_accuracies.csv")
)

# %%
# Plot per-timepoint accuracy as a horizontal bar chart
fig, ax = plt.subplots(figsize=(8, 4))

sns.barplot(
    x='Accuracy',
    y='Blood meal hours',
    data=accuracy_df,
    palette='colorblind',
    width=0.5,
    ax=ax
)

# Reference line marking the 0.70 accuracy threshold
ax.axvline(0.7, color='gray', linestyle='--', linewidth=2, label='0.70 threshold')

ax.set_xticks(np.arange(0.0, 1.1, step=0.2))
ax.set_ylabel('Hours', weight='bold')
ax.set_xlabel('Accuracy', weight='bold')
fig.tight_layout()

fig.savefig(
    os.path.join("..", "Results", "CNN_TL", "tl_bloodmeal_hours_accuracies.png"),
    dpi=500,
    bbox_inches="tight"
)
# plt.show()
# %%
