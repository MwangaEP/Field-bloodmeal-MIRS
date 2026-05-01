#%%
# =============================================================================
# Blood Meal Source Classification (Human vs Bovine)
# MLP Transfer Learning — fine-tune the base MLP on blood-meal hours data
# Base model was trained with Dense-only layers (no Conv) on lab spectra.
# =============================================================================

import os
import io
import json
import joblib
from time import time

import numpy as np 
import pandas as pd
from collections import Counter 

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, accuracy_score

import tensorflow as tf
from tensorflow import keras
from keras.optimizers import SGD
from keras.models import Model, load_model
from keras.regularizers import *
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import backend as K

from MLP_functions import build_folder, visualize

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
# 1. Output directories
# =============================================================================

outdir  = os.path.join("..", "Results")
savedir = os.path.join(outdir, "MLP_TL")
build_folder(outdir, False)
build_folder(savedir, False)

#%%
# =============================================================================
# 2. Load base model and its paired scaler
# =============================================================================

scaler = joblib.load(
    os.path.join("..", "Results", "MLP", "data_scaler.joblib")
)

clf = load_model(
    os.path.join("..", "Results", "MLP", "Baseline_MLP_0_5_Model.keras")
)

#%%
# =============================================================================
# 3. Load and prepare blood-meal hours data
# =============================================================================

blood_hours_df = pd.read_csv(
    os.path.join("..", "Data", "Bloodfed_hours.dat"), 
    delimiter = '\t'
)

# Rename host names 
blood_hours_df['Cat3'] = blood_hours_df['Cat3'].str.replace('CW', 'Bovine')
blood_hours_df['Cat3'] = blood_hours_df['Cat3'].str.replace('HN', 'Human')

print("Hours distribution:", Counter(blood_hours_df['Cat4']))

FEATURE_COLS_TO_DROP = ["Cat1", "Cat2", "Cat3", "Cat4", "StoTime"]
 
TIME_POINTS = {
    "6H":  "6 hours",
    "12H": "12 hours",
    "24H": "24 hours",
    "48H": "48 hours",
}

# Build a dict of raw (unscaled) X and y per time point — scaling happens
# inside the helper functions so there is one clear place of responsibility.
raw_data = {}
for key, label in TIME_POINTS.items():
    subset = blood_hours_df[blood_hours_df["Cat4"] == key].copy()
    raw_data[key] = {
        "X": np.asarray(subset.drop(FEATURE_COLS_TO_DROP, axis=1)),
        "y": np.asarray(subset["Cat3"]),
        "label": label,
    }

#%%
# =============================================================================
# 4. Helper: encode labels with consistent class ordering
# =============================================================================

# Explicit class ordering — must match the base model training encoding:
# index 0 = Bovine, index 1 = Human
CLASS_NAMES = ["Bovine", "Human"]

def encode_labels(y_raw):
    """One-hot encode string labels with a fixed class order.
 
    Using explicit classes= ensures index 0 = Bovine, index 1 = Human
    regardless of which classes happen to appear in the subset, preventing
    silent label-flip on small or imbalanced splits.
 
    Args:
        y_raw: 1-D array of string labels ('Bovine' / 'Human').
 
    Returns:
        hosts_onehot: (n_samples, 2) one-hot array.
    """
    host_list = [[h] for h in y_raw]
    mlb = MultiLabelBinarizer(classes=CLASS_NAMES)
    return mlb.fit_transform(host_list)
 

#%%

# =============================================================================
# 5. Helper: predict with the base or TL model
#    Expects pre-scaled X input.
# =============================================================================

def predict_bloodmeal_hours(X_scaled, y_raw, model, class_names=CLASS_NAMES):
    """Predict blood meal source with a trained Keras model.
 
    Args:
        X_scaled:  Pre-scaled feature array (n_samples, n_features).
        y_raw:     1-D array of string true labels.
        model:     Compiled Keras model.
        class_names: List of class names for one-hot encoding.
 
    Returns:
        y_pred:    Softmax probability array (n_samples, 2).
        y_true:    One-hot true label array (n_samples, 2).
        classes:   [CLASS_NAMES] — passed through for visualize().
    """

    y_true = encode_labels(y_raw)
    X      = X_scaled.reshape([X_scaled.shape[0], -1]) 

    score  = model.evaluate(X, y_true, verbose=0)
    print(f"  Loss: {score[0]:.4f} | Accuracy: {score[1]:.4f}")

    y_pred = model.predict(X)
    return y_pred, y_true, [CLASS_NAMES]

#%%
# =============================================================================
# 6. Base model evaluation on each time point (before fine-tuning)
# =============================================================================

print("\n--- Base model evaluation ---")
for key, data in raw_data.items():
    print(f"\nTime point: {data['label']}")
    X_scl = scaler.transform(data["X"])
    pred, true, classes = predict_bloodmeal_hours(X_scl, data["y"], clf)
    visualize(1, savedir, "base_model", f"{key}_bloodmeal", classes, pred, true)


#%%
# =============================================================================
# 7. Prepare transfer learning data
#    15 % of each time-point dataset → fine-tuning train set
#    85 % → held-out test set (following Mwanga et al., 2024)
# =============================================================================

def split_data(X, y, test_size = 0.85, random_state = 42):
    """
    Split into stratified train / test DataFrames.
 
    Args:
        X:            Feature array.
        y:            Label array.
        test_size:    Fraction reserved for testing.
        random_state: Reproducibility seed.
 
    Returns:
        train_df, test_df: DataFrames with features + 'blood_meal' column.
    """

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, 
        test_size = test_size, 
        random_state = random_state, 
        shuffle = True
    )
    # combine X and y into a single dataframe for both training and test sets
    train_df = pd.DataFrame(X_tr)
    train_df['blood_meal'] = y_tr

    test_df = pd.DataFrame(X_te)
    test_df['blood_meal'] = y_te
    
    return train_df, test_df


splits = {}

for key, data in raw_data.items():
    print(f"\nSplitting data for time point")
    train_df, test_df = split_data(data["X"], data["y"], test_size = 0.85)
    splits[key] = {"train": train_df, "test": test_df}

# Combine all time-point train / test splits into single DataFrames

train_trans_df = pd.concat(
    [splits[key]["train"] for key in TIME_POINTS],
    axis = 0,
).reset_index(drop = True)

test_trans_df = pd.concat(
    [splits[key]["test"] for key in TIME_POINTS],
    axis = 0,
).reset_index(drop = True)

X_train_t = np.asarray(train_trans_df.drop("blood_meal", axis=1))
y_train_t = np.asarray(train_trans_df["blood_meal"])
 
X_val_t = np.asarray(test_trans_df.drop("blood_meal", axis=1))
y_val_t = np.asarray(test_trans_df["blood_meal"])
    

#%%
# =============================================================================
# 8. Transfer learning — fine-tune the base model
# =============================================================================

def train_evaluate_tf(base_model, scaler, X_train, y_train, X_val, y_val):
    """
    Fine-tune a pre-trained Keras model on new data.
 
    Scaling is applied here with the provided scaler so callers pass raw
    (unscaled) arrays and there is one clear scaling responsibility.
 
    Args:
        base_model: Pre-trained Keras model to fine-tune.
        scaler:     Fitted StandardScaler from base model training.
        X_train:    Raw training features.
        y_train:    String training labels.
        X_val:      Raw validation features.
        y_val:      String validation labels.
 
    Returns:
        history:        Keras History object.
        tl_model:       Fine-tuned Keras Model.

    """

    # Wrap base model (keeps all weights as starting point)
    tl_model = Model(inputs = base_model.input, outputs = base_model.output)

    sgd_tl = SGD(
        learning_rate = 0.00001, 
        momentum = 0.9, 
        nesterov = True, 
        clipnorm = 1.0,
    )


    tl_model.compile(
        loss = 'categorical_crossentropy', 
        metrics = ['accuracy'], # canonical name; avoids 'acc' legacy ambiguity
        optimizer = sgd_tl
    )

    # Scale with the base model's scaler — transform only, no re-fitting
    X_train_scl = scaler.transform(X_train).reshape([X_train.shape[0], -1])
    X_val_scl = scaler.transform(X_val).reshape([X_val.shape[0], -1])

    y_train_enc = encode_labels(y_train)
    y_val_enc = encode_labels(y_val)

    history = tl_model.fit(
        x = X_train_scl, 
        y = y_train_enc,
        batch_size = 256, 
        verbose = 1, 
        epochs = 5000,
        validation_data = (X_val_scl, y_val_enc),
        callbacks = [
            EarlyStopping(
                monitor = 'val_loss', patience = 50, verbose = 1,
                restore_best_weights = True, mode = 'auto',
            ), 
            ReduceLROnPlateau(
                monitor = 'val_loss', factor = 0.1, patience = 10,
                verbose = 1,
                mode = 'auto'
            )
        ]
    )
 
    tl_model.save(
        os.path.join(savedir, 'MLP_TL_model.keras')
    )

    return history, tl_model

# Fine-tune model
start_time = time()
 
history, transfer_model = train_evaluate_tf(
    clf, scaler, X_train_t, y_train_t, X_val_t, y_val_t
)
 
elapsed = time() - start_time
print(f"\nRun time: {elapsed:.1f} s | ({elapsed / 60:.2f} min  |  {elapsed / 3600:.3f} h)")
 

#%%
# =============================================================================
# 9. Evaluate TL model on held-out test splits + save classification reports
# =============================================================================

accuracy_records = []

for key, d in TIME_POINTS.items():
    tag   = key.lower()          # e.g. '6h', '12h', etc
    label = d                    # e.g. '6 hours', '12 hours', etc

    test_df = splits[key]["test"]
    X_test = np.asarray(test_df.drop("blood_meal", axis=1))
    y_test = np.asarray(test_df["blood_meal"])

    # scale with base scaler (transform only, no re-fitting)
    X_test_scl = scaler.transform(X_test)

    # Predict
    pred, true, classes = predict_bloodmeal_hours(X_test_scl, y_test, transfer_model)

    visualize(1, savedir, "tl_model", f"{tag}_bloodmeal", classes, pred, true)

    # Classification report
    cr_str = classification_report(
        np.argmax(true, axis=-1),
        np.argmax(pred, axis=-1),
        target_names=CLASS_NAMES,
    )
    print(f"\nClassification report — {label}:\n{cr_str}")

    cr_df = pd.read_fwf(io.StringIO(cr_str), header=0).iloc[1:]
    cr_df.to_csv(
        os.path.join(savedir, f"cr_report_TL_{tag}_bloodmeal.csv"),
        index=False,
    )
 
    acc = accuracy_score(np.argmax(true, axis=-1), np.argmax(pred, axis=-1))
    accuracy_records.append((label, np.round(acc, 2)))


#%%

# =============================================================================
# 10. Accuracy summary — table and bar chart
# =============================================================================
 
accuracy_df = pd.DataFrame(accuracy_records, columns=["Blood meal hours", "Accuracy"])
accuracy_df.to_csv(
    os.path.join(savedir, "tl_bloodmeal_hours_accuracies.csv"),
    index=False,
)
 
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(
    x="Accuracy", y="Blood meal hours",
    data=accuracy_df,
    palette="colorblind",
    width=0.5,
    ax=ax,
)
ax.axvline(0.7, color="gray", linestyle="--", linewidth=2)
ax.set_xticks(np.arange(0.0, 1.1, 0.2))
ax.set_ylabel("Hours", fontweight="bold")
ax.set_xlabel("Accuracy", fontweight="bold")
plt.tight_layout()
fig.savefig(
    os.path.join(savedir, "tl_bloodmeal_hours_accuracies.png"),
    dpi=500, bbox_inches="tight",
)
plt.close(fig)

#%%
# =============================================================================
# 11. Training history plot
# =============================================================================
 
def plot_tl_history(history, save_path):
    """Plot and save accuracy curves for the TL training run.
 
    Args:
        history:    Keras History object from model.fit().
        save_path:  Directory to save the PNG.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history.history["accuracy"],     label="accuracy")
    ax.plot(history.history["val_accuracy"], label="val_accuracy")
    ax.set_xlabel("Epochs", fontweight="bold")
    ax.set_ylabel("Accuracy", fontweight="bold")
    ax.set_yticks(np.arange(0.0, 1.05, 0.2))
    ax.legend(loc="lower right")
    ax.grid(False)
    plt.tight_layout()
    fig.savefig(
        os.path.join(save_path, "tl_accuracy_history.png"),
        dpi=500, bbox_inches="tight",
    )
    plt.close(fig)
 
 
plot_tl_history(history, savedir)

# %%

