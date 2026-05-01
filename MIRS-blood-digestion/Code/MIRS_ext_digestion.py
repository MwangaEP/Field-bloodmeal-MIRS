#%%
# =============================================================================
# Blood Meal Source Classification (Human vs Bovine)
# SVC with repeated stratified K-fold cross-validation
# =============================================================================

import os
import io
import shap
import json
import joblib
from time import time

import numpy as np 
import pandas as pd
from collections import Counter 

import matplotlib.pyplot as plt # for making plots
import seaborn as sns

from sklearn.model_selection import (
    RandomizedSearchCV, 
    KFold
)

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    classification_report,  
    precision_recall_fscore_support, 
    roc_auc_score,
    roc_curve, 
)

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from my_functions import (
    plot_confusion_matrix, 
    visualize,
    evaluate_model,
    plot_algorithm_comparison
)

sns.set(
    context = "paper",
    style = "white",
    palette = "deep",
    font_scale = 1.5,
    color_codes = True,
    rc = ({"font.family": "Dejavu Sans"})
)

plt.rcParams["figure.figsize"] = [6,4]


#%%

# =============================================================================
# 1. Load and prepare lab (training) data
# =============================================================================

blood_meal_lab_df = pd.read_csv(
    os.path.join("..", "Data", "Blood_meal_lab.dat"), 
    delimiter= '\t'
)

blood_meal_lab_df['Cat3'] = blood_meal_lab_df['Cat3'].str.replace('BF', 'Bovine')
blood_meal_lab_df['Cat3'] = blood_meal_lab_df['Cat3'].str.replace('HF', 'Human')

# Drop unused columns
blood_meal_lab_df = blood_meal_lab_df.drop(['Cat1', 'Cat2', 'Cat5'], axis=1)
blood_meal_lab_df.rename(columns = {'Cat3':'blood_meal'}, inplace = True) 

print('Class distribution', Counter(blood_meal_lab_df['blood_meal']))
print(blood_meal_lab_df.head())

# Select features and labels 
X = blood_meal_lab_df.drop('blood_meal', axis = 1)
y = blood_meal_lab_df['blood_meal']

# Fit encoder and scaler ONCE on the full training data.
# These fitted objects are reused for all downstream predictions to prevent
# data leakage and ensure consistent label/feature encoding.
mlb = LabelEncoder()
y_encoded = mlb.fit_transform(np.asarray(y))

CLASS_NAMES = list(mlb.classes_)  # ['Bovine', 'Human']
print("Class mapping:", dict(zip(mlb.classes_, mlb.transform(mlb.classes_))))

scaler = StandardScaler()
X_scl = scaler.fit_transform(np.asarray(X)) # scalling only for evaluation

#%%

# =============================================================================
# 2. Algorithm comparison (sanity check before tuning)
# =============================================================================

seed = np.random.randint(0, 81470)
num_folds = 5 
scoring = 'accuracy' # evaluation metric

# specify cross-validation 

kf = KFold(n_splits = num_folds, shuffle = True, random_state = seed)

# make a list of models to test

models = [
    ('KNN', KNeighborsClassifier()),
    ('LR', LogisticRegression(max_iter = 2000, random_state = seed)),
    ('SVM', SVC(kernel = 'linear', gamma = 'auto', random_state = seed)),
    ('RF', RandomForestClassifier(n_estimators = 500, random_state = seed)),
    ('XGB', XGBClassifier(random_state = seed, n_estimators = 500)),
    ("MLP", MLPClassifier(
        random_state=seed, max_iter = 3500,
        solver = 'sgd', activation = 'logistic', alpha = 0.001
        )),
]

# algorithm comparison
results, names = evaluate_model(models, X_scl, y_encoded, kf, scoring)

fig = plot_algorithm_comparison(results, names)

# save the plot
fig.savefig(
    os.path.join("..", "Results", "ALL_WN", "model_comparison.png"),
    dpi = 500,
    bbox_inches = "tight"
)

plt.close(fig)

#%%

# =============================================================================
# 3. Repeated K-fold with randomised hyperparameter search (SVC)
# =============================================================================

num_rounds = 5 
scoring = 'accuracy' 

# Define the parameter grid
random_grid = {
    'C': [0.001, 0.01, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1, 10, 100, 1000],
    'class_weight': ['balanced', None]  
}

# prepare matrices of results
# Accumulators — defined here so re-running this cell in a Jupyter notebook
# always starts clean and never double-accumulates across kernel sessions.

kf_results                = pd.DataFrame() # model parameters and global accuracy score
svc_coef_df               = pd.DataFrame() # model coef for each iteration
kf_per_class_results      = [] # per class accuracy scores
save_predicted, save_true = [], [] # save predicted and true values for each loop
all_predicted_probs       = [] # save predicted probabilities for each loop
save_scores               = []   # P(Human) collected in the same loop iteration
                            # as save_true, guaranteeing lengths always match.

best_classifier    = None
best_scaler        = None          # scaler paired with the best classifier
best_score_overall = -1.0

start = time()

# Create the SVC model with linear kernel
base_svc = SVC(kernel='linear', probability=True)

X_raw = np.asarray(X)  # Use raw features for scaling within each fold

for round_idx in range (num_rounds):
    
    for fold_idx, (train_index, test_index) in enumerate(
        kf.split(X_raw, y_encoded)):

        X_train_set, X_val = X_raw[train_index], X_raw[test_index]
        y_train_set, y_val = y_encoded[train_index], y_encoded[test_index]

        print('The shape of X train set : {}'.format(X_train_set.shape))
        print('The shape of y train set  : {}'.format(y_train_set.shape))
        print('The shape of X val set : {}'.format(X_val.shape))
        print('The shape of y val set : {}'.format(y_val.shape))

        # --- Per-fold scaling (fit on train only, transform both splits) ---
        fold_scaler = StandardScaler().fit(X_train_set)
        X_train_set = fold_scaler.transform(X_train_set)
        X_val       = fold_scaler.transform(X_val)

        # ------ Randomised grid search ------

        rsCV = RandomizedSearchCV(
            estimator=base_svc,
            param_distributions=random_grid,
            n_iter=50, 
            scoring=scoring, 
            cv=kf, 
            refit=True, 
            n_jobs=-1,
            verbose=0,
        )
        
        rsCV_result = rsCV.fit(X_train_set, y_train_set)

        print(f"Round {round_idx + 1} | Fold {fold_idx + 1} | "
              f"Best CV: {rsCV.best_score_:.3f} | Params: {rsCV.best_params_}")

        # # print out results and give hyperparameter settings for best one
        # means = rsCV_result.cv_results_['mean_test_score']
        # stds = rsCV_result.cv_results_['std_test_score']
        # params = rsCV_result.cv_results_['params']
        # for mean, stdev, param in zip(means, stds, params):
        #     print("%.2f (%.2f) with: %r" % (mean, stdev, param))

        fold_classifier = SVC(kernel="linear", probability=True)
        fold_classifier.set_params(**rsCV_result.best_params_)
        fold_classifier.fit(X_train_set, y_train_set)

        # Track the globally best model and its paired scaler
        if rsCV.best_score_ > best_score_overall:
            best_score_overall = rsCV.best_score_
            best_classifier    = fold_classifier
            best_scaler        = fold_scaler

        # ------ Predictions ------
        y_pred          = fold_classifier.predict(X_val)
        predicted_probs = fold_classifier.predict_proba(X_val)
        all_predicted_probs.append(predicted_probs)

        save_predicted.extend(y_pred)
        save_true.extend(y_val)
        save_scores.extend(predicted_probs[:, 1])  # P(Human) — same length as y_val

        # ------ SVC coefficients ------
        coef_series = pd.Series(fold_classifier.coef_[0], index=X.columns)
        svc_coef_df = pd.concat(
            [svc_coef_df, coef_series.rename(
                f"r{round_idx}_f{fold_idx}"
            )],
            axis=1,
        )

        # predict test instances 

        y_pred = best_classifier.predict(X_val)
        classes = ['Bovine', 'Human']
        
        # Get predicted probabilities
        predicted_probs = best_classifier.predict_proba(X_val)
        all_predicted_probs.append(predicted_probs)  # Save the probabilities for the current fold

        # append feauture importances
        coef_table = pd.Series(best_classifier.coef_[0], X.columns)
        coef_table = pd.DataFrame(coef_table)
        svc_coef_df = pd.concat(
            [
                svc_coef_df, 
                coef_table
            ],
            axis = 1,
            ignore_index = True
        )

        # ------ Fold-level results ------

        local_cm     = confusion_matrix(y_val, y_pred)
        local_report = classification_report(y_val, y_pred)

        fold_row = pd.DataFrame(
            [
                ("Accuracy", accuracy_score(y_val, y_pred)), 
                ("TRAIN",str(train_index)),
                ("TEST",str(test_index)),
                ("CM", local_cm), 
                ("Classification report", local_report), 
                ("y_val", y_val),
            ]
        ).T
            
        fold_row.columns = fold_row.iloc[0]
        fold_row = fold_row[1:]
        kf_results = pd.concat(
            [ kf_results, fold_row], 
            axis = 0, 
            join = 'outer'
        ).reset_index(drop = True)

        # per class accuracy
        local_support = precision_recall_fscore_support(y_val, y_pred)[3]
        local_acc     = np.diag(local_cm)/local_support
        kf_per_class_results.append(local_acc)

elapsed = time() - start
print(f"\nTraining complete. Time elapsed: {elapsed / 60:.2f} min ({elapsed:.1f} s)")
print(f"Best overall CV accuracy: {best_score_overall:.3f}")

# --- Persist the best classifier and its paired scaler ---
joblib.dump(
    best_classifier,
    os.path.join("..", "Results", "ALL_WN", "best_svc_classifier.joblib"),
)
joblib.dump(
    best_scaler,
    os.path.join("..", "Results", "ALL_WN", "data_scaler.joblib"),
)
print("Saved best_svc_classifier.joblib and data_scaler.joblib to Results/ALL_WN")

#%%
# =============================================================================
# 4. Averaged confusion matrix and ROC curve (cross-validation performance)
# =============================================================================
visualize("baseline_cm", save_predicted, save_true)

y_true   = np.asarray(save_true)  # True labels
y_scores = np.asarray(save_scores)  # Get probabilities for the positive class (Human)

# Calculate ROC AUC
roc_auc = roc_auc_score(y_true, y_scores)
print(f"Cross-validation ROC AUC: {roc_auc:.2f}")

# Plot ROC Curve
fpr_cv, tpr_cv, _ = roc_curve(y_true, y_scores)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(fpr_cv, tpr_cv, color='red', label=f'ROC Curve (AUC = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line for chance level
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', weight='bold')
ax.set_ylabel('True Positive Rate', weight='bold')
ax.legend(loc='lower right')
ax.grid(False)
plt.tight_layout()

fig.savefig(
    os.path.join("..", "Results", "ALL_WN", "roc_curve.png"), 
    dpi=500, bbox_inches='tight',
)

#%%
# =============================================================================
# 5. SVC coefficient summary and plot
# =============================================================================

# summarizing SVC coefficients

sss_coef_2 = svc_coef_df
sss_coef_2.dropna(axis = 1, inplace = True)
sss_coef_2["coef mean"] = sss_coef_2.mean(axis=1)
sss_coef_2["coef sem"] = sss_coef_2.sem(axis=1)

n_features = 25 # number of top positive and negative features to plot

# sort the coefficients
sss_coef_2.sort_values(by = "coef mean", ascending = False, inplace = True)

# select the top 25 and bottom 25 coefficients
coef_data = sss_coef_2.drop(["coef sem", "coef mean"], axis = 1).T
coef_plot_data = coef_data.iloc[:,:].drop(
    coef_data.columns[n_features:-n_features], axis = 1
)

# Plotting
fig, ax = plt.subplots(figsize = (5,16))
sns.barplot(
        data = coef_plot_data, 
        orient = "h", 
        palette = "plasma", 
        capsize = .2,
        ax = ax
    )
plt.ylabel("Wavenumbers", weight = "bold")
plt.xlabel("Coefficients", weight = "bold")
ax.grid(False)
plt.tight_layout()

plt.savefig(os.path.join(
    "..", "Results", "ALL_WN", "svc_coef.png"), 
    dpi = 500, bbox_inches = "tight"
    )

# save wavenumbers for the top positive and negative coefficients

n_features_2 = 50
coef_top_data = coef_data.drop(
    coef_data.columns[n_features_2:-n_features_2], axis=1
)
with open(
    os.path.join("..", "Results", "ALL_WN", "top_svc_coefficients_wavenumbers.txt"), "w"
) as outfile:
    json.dump(coef_top_data.columns.tolist(), outfile)


#%%

# =============================================================================
# 6. SHAP analysis using the best model
# Load the persisted classifier and scaler so this section can be run
# independently without re-running the full training loop.
# =============================================================================

best_classifier = joblib.load(
    os.path.join("..", "Results", "ALL_WN", "best_svc_classifier.joblib")
)
best_scaler = joblib.load(
    os.path.join("..", "Results", "ALL_WN", "data_scaler.joblib")
)

# Scale the full training set with the best fold's scaler
X_shap_scaled = best_scaler.transform(X_raw)
X_shap_df = pd.DataFrame(X_shap_scaled, columns=X.columns)

# Use all features (no reduction) in the SHAP computation
X_shap_sample = X_shap_df[:200]  # Use first 200 samples (or any subset)

# Set up SHAP explainer with the full feature set
explainer = shap.KernelExplainer(
     best_classifier.predict_proba,
     X_shap_sample  # Pass all 200 samples with all features
)
shap_values = explainer.shap_values(X_shap_sample)

print(f"Shape of SHAP values: {shap_values.shape}")

# Class index 1 = Human (positive class)
shap_values_human = shap_values[:, :, 1]  # or shap_values[0] if you're interested in class 0

# Compute the mean absolute SHAP values for each feature across all samples (axis=0)
mean_shap_values = np.mean(np.abs(shap_values_human), axis=0)

shap_importance = pd.DataFrame({
    'feature': X_shap_df.columns,
    'mean_shap_value': mean_shap_values
    })
    
# Sort by mean SHAP values and select top 20 or 30 important features
top_shap_features = (
    shap_importance.sort_values(
        by='mean_shap_value', 
        ascending=False).head(30)
)

print(top_shap_features)

# Waterfall plot for the first sample
top_feature_names  = top_shap_features["feature"].astype(str).tolist()
top_feature_indices = [X_shap_df.columns.get_loc(f) for f in top_feature_names]
 
shap_values_top = shap_values_human[:, top_feature_indices]
sample_idx = 0
 
explanation = shap.Explanation(
    values=shap_values_top[sample_idx],
    base_values=explainer.expected_value[1],
    data=X_shap_sample.iloc[sample_idx, top_feature_indices],
    feature_names=top_feature_names,
)
shap.initjs()
shap.plots.waterfall(explanation, show=False)
plt.savefig(
    os.path.join("..", "Results", "ALL_WN", "shap_waterfall.png"),
    dpi=500, bbox_inches="tight",
)

plt.close()

#%%

# =============================================================================
# 7. Predict on time-point data (6, 12, 24, 48 hours post-feeding)
# Load the persisted classifier and scaler so this section can be run
# independently. The scaler is applied with transform() only — no re-fitting.
# =============================================================================

blood_hours_df = pd.read_csv(
       os.path.join("..", "Data", "Bloodfed_hours.dat"), 
       delimiter = '\t'
    )

# Rename items in the column
blood_hours_df['Cat3'] = blood_hours_df['Cat3'].str.replace('CW', 'Bovine')
blood_hours_df['Cat3'] = blood_hours_df['Cat3'].str.replace('HN', 'Human')

print("Hours distribution", Counter(blood_hours_df['Cat4']))

FEATURE_COLS_TO_DROP = ["Cat1", "Cat2", "Cat3", "Cat4", "StoTime"]
 
time_points = {
    "6H":  ("6 hours",  "6h"),
    "12H": ("12 hours", "12h"),
    "24H": ("24 hours", "24h"),
    "48H": ("48 hours", "48h"),
}

# # filter data with blood meal hours (To be used for model testing)
# blood_6hours = blood_hours_df[blood_hours_df['Cat4'] == '6H']
# blood_12hours = blood_hours_df[blood_hours_df['Cat4'] == '12H']
# blood_24hours = blood_hours_df[blood_hours_df['Cat4'] == '24H']
# blood_48hours = blood_hours_df[blood_hours_df['Cat4'] == '48H']

accuracy_records = []
probas_list, true_list, labels_list = [], [], []
all_cr_dfs = {}

for key, (label, tag) in time_points.items():

    subset_df = blood_hours_df[blood_hours_df['Cat4'] == key].copy()

    X_t = subset_df.drop(FEATURE_COLS_TO_DROP, axis=1)
    y_t = subset_df['Cat3']

    # Use transform to (not fit_transform) to apply the best fold's scaler.
    y_t_encoded = mlb.transform(np.asarray(y_t))
    X_t_scl     = best_scaler.transform(np.asarray(X_t))

    y_pred_t = best_classifier.predict(X_t_scl)
    y_proba_t = best_classifier.predict_proba(X_t_scl)
    acc_t = accuracy_score(y_t_encoded, y_pred_t)

    print(f"Accuracy {label}: {acc_t * 100:.2f}%")
    print(classification_report(y_t_encoded, y_pred_t))

    # plot confusion matrix
    visualize(f"{tag}_cm", y_pred_t, y_t_encoded)

    # Classification report → tidy DataFrame
    cr_str = classification_report(y_t_encoded, y_pred_t)
    cr_df = pd.read_fwf(io.StringIO(cr_str), header=0)
    cr_df.rename(columns={"Unnamed: 0": "class"}, inplace=True)
    cr_df.loc[0, "class"] = "Bovine"
    cr_df.loc[1, "class"] = "Human"
    cr_df.to_csv(
        os.path.join("..", "Results", "ALL_WN", f"cr_{tag}.csv"),
        index=False,
    )
    all_cr_dfs[tag] = cr_df
 
    accuracy_records.append((label, np.round(acc_t, 2)))
    probas_list.append(y_proba_t[:, 1])
    true_list.append(y_t_encoded)
    labels_list.append(label)

# %%

# =============================================================================
# 8. Accuracy bar chart and line plot across time points
# =============================================================================

accuracy_bhours_df = pd.DataFrame(accuracy_records, columns=["Hours", "Accuracy"])

# save to disk
accuracy_bhours_df.to_csv(
    os.path.join("..", "Results", "ALL_WN", "accuracy_bhours.csv"), 
    index = False
)

# Plot the results
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(
    x="Accuracy", y="Hours",
    data=accuracy_bhours_df,
    palette="colorblind",
    width=0.5,
    ax=ax,
)
ax.axvline(0.7, color="gray", linestyle="--", linewidth=2)
ax.set_xticks(np.arange(0.0, 1.1, 0.2))
ax.set_ylabel("Hours", fontweight="bold")
ax.set_xlabel("Accuracy", fontweight="bold")
ax.get_legend().remove() if ax.get_legend() else None
plt.tight_layout()
fig.savefig(
    os.path.join("..", "Results", "ALL_WN", "accuracy_bhours_bar.png"),
    dpi=500, bbox_inches="tight",
)
plt.close(fig)
 

fig, ax = plt.subplots(figsize=(6, 4))
sns.lineplot(
    x="Hours", y="Accuracy",
    data=accuracy_bhours_df,
    marker="o", color="red", ax=ax,
)
ax.axhline(0.7, color="gray", linestyle="--", linewidth=2)
ax.set_xlabel("Hours", fontweight="bold")
ax.set_ylabel("Accuracy", fontweight="bold")
ax.set_ylim(0.0, 1.0)
plt.tight_layout()
fig.savefig(
    os.path.join("..", "Results", "ALL_WN", "accuracy_bhours_line.png"),
    dpi=500, bbox_inches="tight",
)
plt.close(fig)

# %%
# =============================================================================
# 9. ROC curves for each time point
# =============================================================================
 
fig, ax = plt.subplots(figsize=(6, 5))
 
for y_proba, y_true, label in zip(probas_list, true_list, labels_list):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    ax.plot(fpr, tpr, label=f"{label} (AUC = {auc:.2f})")
 
ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel("False Positive Rate", fontweight="bold")
ax.set_ylabel("True Positive Rate", fontweight="bold")
ax.legend(loc="lower right")
plt.tight_layout()
fig.savefig(
    os.path.join("..", "Results", "ALL_WN", "roc_curves_bhours.png"),
    dpi=500, bbox_inches="tight",
)
plt.close(fig)

# %%
