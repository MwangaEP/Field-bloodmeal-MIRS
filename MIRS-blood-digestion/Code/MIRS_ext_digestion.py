#%%
import os
import io
import json
import ast
import itertools
import collections
from time import time
import sklearn
from tqdm import tqdm

from itertools import cycle
import pickle
import random as rn
import datetime

import numpy as np 
import pandas as pd

# from random import randint
from scipy.stats import uniform, randint
from collections import Counter 

from sklearn.model_selection import (
    ShuffleSplit, 
    train_test_split, 
    StratifiedKFold, 
    StratifiedShuffleSplit, 
    KFold
    ) 
from sklearn.model_selection import (
    RandomizedSearchCV, 
    GridSearchCV, 
    cross_val_score
    )
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer,LabelEncoder
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    classification_report, 
    max_error, 
    precision_recall_fscore_support, 
    roc_auc_score,
    roc_curve, 
    auc, 
    precision_score, 
    recall_score, 
    f1_score
    )

from imblearn.under_sampling import RandomUnderSampler

from sklearn import decomposition
from sklearn.pipeline import Pipeline

from sklearn.linear_model import (
    LogisticRegression, 
    LogisticRegressionCV
    )

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

import matplotlib.pyplot as plt # for making plots
import seaborn as sns
sns.set(
            context = "paper",
            style = "white",
            palette = "deep",
            font_scale = 1.5,
            color_codes = True,
            rc = ({"font.family": "Dejavu Sans"})
        )
# %matplotlib inline

plt.rcParams["figure.figsize"] = [6,4]

#%%

# This normalizes the confusion matrix and ensures neat plotting for all outputs.
# Function for plotting confusion matrcies

def plot_confusion_matrix(
                            cm, 
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
    
    plt.figure(figsize=(6,4))

    plt.imshow(
                cm, 
                interpolation = 'nearest', 
                vmin = .0, 
                vmax = 1.0,  
                cmap = cmap
            )
        
    # plt.title(title)
    plt.colorbar()
    classes_names = ['Bovine', 'Human']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes_names, rotation = xrotation)
    plt.yticks(tick_marks, classes_names, rotation = yrotation)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
                    j, 
                    i, 
                    format(cm[i, j], fmt), 
                    horizontalalignment = "center",
                    color = "white" if cm[i, j] > thresh else "black"
                )

    plt.tight_layout()
    plt.ylabel('True Host', weight = 'bold')
    plt.xlabel('Predicted Host', weight = 'bold')
    plt.savefig(
        os.path.join("..", "Results", figure_name + ".png"
        ), 
        dpi = 500, 
        bbox_inches = "tight"
        
    )

#%%

def visualize(figure_name, predicted, true):
    # Sort out predictions and true labels
    classes_pred = np.asarray(predicted)
    classes_true = np.asarray(true)
    print(classes_pred.shape)
    print(classes_true.shape)
    cnf_matrix = confusion_matrix(
                                    classes_true, 
                                    classes_pred, 
                                    # labels = classes
                                )
    plot_confusion_matrix(cnf_matrix)



#%%

# Load lab data

blood_meal_lab_df = pd.read_csv(
    os.path.join("..", "Data", "Blood_meal_lab.dat"), 
    delimiter= '\t'
)


blood_meal_lab_df['Cat3'] = blood_meal_lab_df['Cat3'].str.replace('BF', 'Bovine')
blood_meal_lab_df['Cat3'] = blood_meal_lab_df['Cat3'].str.replace('HF', 'Human')

# Drop unused columns
blood_meal_lab_df = blood_meal_lab_df.drop(
    [
        'Cat1', 
        'Cat2', 
        'Cat5'
    ], 
    axis=1
)

blood_meal_lab_df.rename(
    columns = {'Cat3':'blood_meal'}, 
    inplace = True
) 

print('Size of blood meal by count', Counter(blood_meal_lab_df['blood_meal']))
blood_meal_lab_df.head()

#%%

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

# define parameters

num_folds = 5 # split data into five folds
seed = np.random.randint(0, 81470) # random seed value
scoring = 'accuracy' # metric for model evaluation

# specify cross-validation strategy

kf = KFold(
            n_splits = num_folds,
            shuffle = True,
            random_state = seed
        )

# make a list of models to test

models = []
models.append(
                (
                    'KNN', KNeighborsClassifier()
                )
            )

models.append(
                (
                    'LR', LogisticRegression(
                                                multi_class = 'ovr',
                                                max_iter = 2000,
                                                random_state = seed
                                            )
                )
            )

models.append(
                (
                    'SVM', SVC(
                                kernel = 'linear',
                                gamma = 'auto',
                                random_state = seed
                            )
                )
            )

models.append(
                (
                    'RF', RandomForestClassifier(
                                                    n_estimators = 500,
                                                    random_state = seed
                                                )
                )
            )

models.append(
                ('XGB', XGBClassifier(
                                        random_state = seed,
                                        n_estimators = 500
                                    )
                )
            )

#%% 
# Prepare data for model training

# first, we need to count the number of samples in each class, checking for class imbalance
print(Counter(blood_meal_lab_df['blood_meal']))

# class are balanced, so we can proceed to encode the data
# define the features and target variable
X = blood_meal_lab_df.drop('blood_meal', axis = 1)
y = blood_meal_lab_df['blood_meal']

# encode the target variable
mlb = LabelEncoder()
y_encoded = mlb.fit_transform(np.asarray(y))

# scale the data
scaler = StandardScaler().fit(np.asarray(X))
X_scl = scaler.transform(np.asarray(X))

#%%
# Evaluate models to get the best performing model

results = []
names = []

for name, model in models:
    cv_results = cross_val_score(
        model,
        X_scl,
        y_encoded,
        cv=kf,
        scoring=scoring
    )
    results.append(cv_results)
    names.append(name)
    msg = f'Cross validation score for {name}: {cv_results.mean():.2%} ± {cv_results.std():.2%}'
    print(msg)

#%%

# Plot results for algorithm comparison

# transform the vectors into pandas dataframe
results_df = pd.DataFrame(
                            results,
                            columns = (0, 1, 2, 3, 4)
                        ).T # columns should correspond to the number of folds, k = 5

results_df.columns = names
results_df = pd.melt(results_df) # melt data frame into a long format.
results_df.rename(
                    columns = {'variable':'Model', 'value':'Accuracy'},
                    inplace = True
                )


# Plotting the algorithm selection

plt.figure(figsize = (5, 4))

sns.boxplot(
                x = 'Model',
                y = 'Accuracy',
                data = results_df,
                hue = 'Model',
                palette = 'colorblind'
            )
# sns.boxplot(x = names, y = results, width = .4)
sns.despine(offset = 5, trim = False)
plt.xticks(rotation = 90)
plt.yticks(np.arange(0.0, 1.0 + .1, step = 0.2))
plt.ylabel('Accuracy', weight = 'bold')
plt.xlabel(" ")
plt.legend().remove()
plt.tight_layout()

# save the plot
plt.savefig(
    os.path.join("..", "Results", "model_comparison.png"),
    dpi = 500,
    bbox_inches = "tight"
)


#%%

# big LOOP
# TUNNING THE SELECTED MODEL

num_rounds = 5 # increase this to 5 or 10 once code is bug-free
scoring = 'accuracy' # score model accuracy

# prepare matrices of results
kf_results = pd.DataFrame() # model parameters and global accuracy score
lr_coef_df = pd.DataFrame() # model coef for each iteration
kf_per_class_results = [] # per class accuracy scores
save_predicted, save_true = [], [] # save predicted and true values for each loop
all_predicted_probs = [] # save predicted probabilities for each loop

start = time()

# Define the parameter grid
random_grid = {
    'C': [0.001, 0.01, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1, 10, 100, 1000],
    'class_weight': ['balanced', None]  # To handle potential class imbalance
}

# Create the SVC model with linear kernel
classifier = SVC(kernel='linear', probability=True)

for round in range (num_rounds):
    SEED = np.random.randint(0, 8147)

    # cross validation and splitting of the validation set
    
    for train_index, test_index in kf.split(X_scl, y_encoded):
        X_train_set, X_val = X_scl[train_index], X_scl[test_index]
        y_train_set, y_val = y_encoded[train_index], y_encoded[test_index]

        print('The shape of X train set : {}'.format(X_train_set.shape))
        print('The shape of y train set  : {}'.format(y_train_set.shape))
        print('The shape of X val set : {}'.format(X_val.shape))
        print('The shape of y val set : {}'.format(y_val.shape))

        # RANDOMSED GRID SEARCH
        # Random search of parameters, using 5 fold cross validation, 
        # search across 100 different combinations, and use all available cores

        n_iter_search = 10
        rsCV = RandomizedSearchCV(
            verbose=1,
            estimator=classifier, 
            param_distributions=random_grid, 
            n_iter=n_iter_search, 
            scoring=scoring, 
            cv=kf, 
            refit=True, 
            n_jobs=-1
        )
        
        rsCV_result = rsCV.fit(X_train_set, y_train_set)

        # print out results and give hyperparameter settings for best one
        means = rsCV_result.cv_results_['mean_test_score']
        stds = rsCV_result.cv_results_['std_test_score']
        params = rsCV_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%.2f (%.2f) with: %r" % (mean, stdev, param))

        # print best parameter settings
        print("Best: %.2f using %s" % (rsCV_result.best_score_,
                                    rsCV_result.best_params_))

        # Insert the best parameters identified by randomized grid search into the base classifier
        best_classifier = classifier.set_params(**rsCV_result.best_params_)
        
        # fit the model
        best_classifier.fit(X_train_set, y_train_set)

        # predict test instances 

        y_pred = best_classifier.predict(X_val)
        classes = ['Bovine', 'Human']
        local_cm = confusion_matrix(y_val, y_pred)
        local_report = classification_report(y_val, y_pred)

        # Get predicted probabilities
        predicted_probs = best_classifier.predict_proba(X_val)
        all_predicted_probs.append(predicted_probs)  # Save the probabilities for the current fold


        # zip predictions for all rounds for plotting averaged confusion matrix
            
        for predicted, true in zip(y_pred, y_val):
            save_predicted.append(predicted)
            save_true.append(true)

        # # append feauture importances
        coef_table = pd.Series(best_classifier.coef_[0], X.columns)
        coef_table = pd.DataFrame(coef_table)
        lr_coef_df = pd.concat(
            [
                lr_coef_df, 
                coef_table
            ],
            axis = 1,
            ignore_index = True
        )

        # summarizing results
        local_kf_results = pd.DataFrame(
                                            [
                                                ("Accuracy", accuracy_score(y_val, y_pred)), 
                                                ("TRAIN",str(train_index)),
                                                ("TEST",str(test_index)),
                                                ("CM", local_cm), 
                                                ("Classification report", local_report), 
                                                ("y_val", y_val)
                                            ]
                                        ).T
            
        local_kf_results.columns = local_kf_results.iloc[0]
        local_kf_results = local_kf_results[1:]
        kf_results = pd.concat(
                                [
                                    kf_results, 
                                    local_kf_results
                                ], 
                                axis = 0, 
                                join = 'outer'
                            ).reset_index(drop = True)

        # per class accuracy
        local_support = precision_recall_fscore_support(y_val, y_pred)[3]
        local_acc = np.diag(local_cm)/local_support
        kf_per_class_results.append(local_acc)

elapsed = time() - start
print("Time elapsed: {0:.2f} minutes ({1:.1f} sec)".format(
    elapsed / 60, elapsed))

#%%
# Now you can use all_predicted_probs to plot the ROC curve
# Assuming '1' is the positive class for ROC curve
y_true = np.array(save_true)  # True labels
y_scores = np.vstack(all_predicted_probs)[:, 1]  # Get probabilities for the positive class (Human)

# Calculate ROC AUC
roc_auc = roc_auc_score(y_true, y_scores)
print(f"ROC AUC: {roc_auc:.2f}")

# Plotting ROC Curve

fpr, tpr, thresholds = roc_curve(y_true, y_scores)

plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, color='red', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line for chance level
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', weight='bold')
plt.ylabel('True Positive Rate', weight='bold')
plt.legend(loc='lower right')
plt.grid(False)
plt.tight_layout()

# Save the plot
plt.savefig(
    os.path.join("..", "Results", "roc_curve.png"), 
    dpi=500, 
    bbox_inches='tight'
)

#%%

# plot confusion averaged for the validation set

classes = np.unique(np.sort(y_val))
figure_name = 'baseline_cm'

visualize(
    figure_name, 
    save_predicted, 
    save_true
)

#%%
# summarizing logistic regression coefficients

sss_coef_2 = lr_coef_df
sss_coef_2.dropna(axis = 1, inplace = True)
sss_coef_2["coef mean"] = sss_coef_2.mean(axis=1)
sss_coef_2["coef sem"] = sss_coef_2.sem(axis=1)
# sss_coef_2.to_csv("coef_repeatedCV_coef.csv")

#%%

# plotting coefficients
n_features = 25

# sort the coefficients
sss_coef_2.sort_values(
    by = "coef mean", 
    ascending = False, 
    inplace = True
)

# select the top 25 and bottom 25 coefficients
coef_plot_data = sss_coef_2.drop(
    [
        "coef sem", 
        "coef mean"
    ], 
    axis = 1
).T

coef_plot_data = coef_plot_data.iloc[:,:].drop(
    coef_plot_data.columns[n_features:-n_features], 
    axis = 1
)


# Plotting 

plt.figure(figsize = (5,16))
sns.barplot(
                data = coef_plot_data, 
                orient = "h", 
                palette = "plasma", 
                capsize = .2
            )

plt.ylabel("Wavenumbers", weight = "bold")
plt.xlabel("Coefficients", weight = "bold")
plt.savefig(os.path.join("..", "Results", "lgr_coef.png"), 
            dpi = 500, 
            bbox_inches = "tight"
        )

#%%
# Making prediction on the blood meal hours data
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

y_6h = blood_6hours['Cat3']

# encode the target variable
y_6h_encoded = mlb.fit_transform(np.asarray(y_6h))

# scale the data
X_6h_scl = scaler.transform(np.asarray(X_6h))

# make predictions
y_pred_6h = best_classifier.predict(X_6h_scl)
y_proba_6h = best_classifier.predict_proba(X_6h_scl)
accuracy_6h = accuracy_score(y_6h_encoded, y_pred_6h)
print("Accuracy_6h: %.2f%%" % (accuracy_6h * 100.0))

#%%

# plot the confusion matrix for the 6 hours data
figure_name = '6h_cm'

visualize(
    figure_name, 
    y_pred_6h, 
    y_6h_encoded
)

# classification report
cr_6h = classification_report(y_6h_encoded, y_pred_6h)
print('Classification report : {}'.format(cr_6h))

# save classification report to disk as a csv

cr_6h_df = pd.read_fwf(
    io.StringIO(cr_6h), 
    header = 0  
)
                            
cr_6h_df = cr_6h_df[0:]

# rename the first column to 'class'
cr_6h_df.rename(
    columns = {'Unnamed: 0':'class'}, 
    inplace = True
)

# rename the first and second rows in class column to Bovine and Human
cr_6h_df.loc[0, 'class'] = 'Bovine'
cr_6h_df.loc[1, 'class'] = 'Human'

# save the classification report
cr_6h_df.to_csv(
    os.path.join("..", "Results", "cr_6h.csv"), 
    index = False
)

#%%
# Making prediction on the blood meal hours data
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

y_12h = blood_12hours['Cat3']

# encode the target variable
y_12h_encoded = mlb.fit_transform(np.asarray(y_12h))

# scale the data
X_12h_scl = scaler.transform(np.asarray(X_12h))

# make predictions
y_pred_12h = best_classifier.predict(X_12h_scl)
y_proba_12h = best_classifier.predict_proba(X_12h_scl)
accuracy_12h = accuracy_score(y_12h_encoded, y_pred_12h)
print("Accuracy_12h: %.2f%%" % (accuracy_12h * 100.0))

#%%

# plot the confusion matrix for the 12 hours data
figure_name = '12h_cm'

visualize(
    figure_name, 
    y_pred_12h, 
    y_12h_encoded
)

# classification report
cr_12h = classification_report(y_12h_encoded, y_pred_12h)
print('Classification report : {}'.format(cr_12h))

# save classification report to disk as a csv

cr_12h_df = pd.read_fwf(
    io.StringIO(cr_12h), 
    header = 0  
)
                            
cr_12h_df = cr_12h_df[0:]

# rename the first column to 'class'
cr_12h_df.rename(
    columns = {'Unnamed: 0':'class'}, 
    inplace = True
)

# rename the first and second rows in class column to Bovine and Human
cr_12h_df.loc[0, 'class'] = 'Bovine'
cr_12h_df.loc[1, 'class'] = 'Human'

# save the classification report
cr_12h_df.to_csv(
    os.path.join("..", "Results", "cr_12h.csv"), 
    index = False
)

# %%
# Making prediction on the blood meal hours data
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

y_24h = blood_24hours['Cat3']

# encode the target variable
y_24h_encoded = mlb.fit_transform(np.asarray(y_24h))

# scale the data
X_24h_scl = scaler.transform(np.asarray(X_24h))

# make predictions
y_pred_24h = best_classifier.predict(X_24h_scl)
y_proba_24h = best_classifier.predict_proba(X_24h_scl)
accuracy_24h = accuracy_score(y_24h_encoded, y_pred_24h)
print("Accuracy_24h: %.2f%%" % (accuracy_24h * 100.0))

#%%

# plot the confusion matrix for the 24 hours data
figure_name = '24h_cm'

visualize(
    figure_name, 
    y_pred_24h, 
    y_24h_encoded
)

# classification report
cr_24h = classification_report(y_24h_encoded, y_pred_24h)
print('Classification report : {}'.format(cr_24h))

# save classification report to disk as a csv

cr_24h_df = pd.read_fwf(
    io.StringIO(cr_24h), 
    header = 0  
)
                            
cr_24h_df = cr_24h_df[0:]

# rename the first column to 'class'
cr_24h_df.rename(
    columns = {'Unnamed: 0':'class'}, 
    inplace = True
)

# rename the first and second rows in class column to Bovine and Human
cr_24h_df.loc[0, 'class'] = 'Bovine'
cr_24h_df.loc[1, 'class'] = 'Human'

# save the classification report
cr_24h_df.to_csv(
    os.path.join("..", "Results", "cr_24h.csv"), 
    index = False
)

# %%
# Making prediction on the blood meal hours data
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

y_48h = blood_48hours['Cat3']

# encode the target variable
y_48h_encoded = mlb.fit_transform(np.asarray(y_48h))

# scale the data
X_48h_scl = scaler.transform(np.asarray(X_48h))

# make predictions
y_pred_48h = best_classifier.predict(X_48h_scl)
y_proba_48h = best_classifier.predict_proba(X_48h_scl)
accuracy_48h = accuracy_score(y_48h_encoded, y_pred_48h)
print("Accuracy_48h: %.2f%%" % (accuracy_48h * 100.0))

#%%

# plot the confusion matrix for the 48 hours data
figure_name = '48h_cm'

visualize(
    figure_name, 
    y_pred_48h, 
    y_48h_encoded
)

# classification report
cr_48h = classification_report(y_48h_encoded, y_pred_48h)
print('Classification report : {}'.format(cr_48h))

# save classification report to disk as a csv

cr_48h_df = pd.read_fwf(
    io.StringIO(cr_48h), 
    header = 0  
)
                            
cr_48h_df = cr_48h_df[0:]

# rename the first column to 'class'
cr_48h_df.rename(
    columns = {'Unnamed: 0':'class'}, 
    inplace = True
)

# rename the first and second rows in class column to Bovine and Human
cr_6h_df.loc[0, 'class'] = 'Bovine'
cr_6h_df.loc[1, 'class'] = 'Human'

# save the classification report
cr_6h_df.to_csv(
    os.path.join("..", "Results", "cr_6h.csv"), 
    index = False
)

# %%

# create a dataframe with all accuracies for prediction on blood meal hours data

accuracy_bhours_df = pd.DataFrame(
    [
        ("6 hours", np.round(accuracy_6h, 2)),
        ("12 hours", np.round(accuracy_12h, 2)),
        ("24 hours", np.round(accuracy_24h,2 )),
        ("48 hours", np.round(accuracy_48h, 2))
    ],
    columns=['Hours', 'Accuracy']
)

# save the results to csv
accuracy_bhours_df.to_csv(
    os.path.join("..", "Results", "accuracy_bhours.csv"), 
    index = False
)

# Plot the results
plt.figure(figsize = (8, 4))

sns.barplot(
    x = 'Accuracy',
    y = 'Hours',
    data = accuracy_bhours_df,
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
plt.legend().remove()
plt.tight_layout()

# save the plot
plt.savefig(
    os.path.join("..", "Results", "accuracy_bhours_bar.png"), 
    dpi = 500, 
    bbox_inches = "tight"
)

#%%
# Create the line plot

# Plot the results
plt.figure(figsize = (6, 4))

sns.lineplot(
    x='Hours', 
    y='Accuracy', 
    data=accuracy_bhours_df, 
    marker='o',  # Add markers for each point
    color='red'  # Set the line color to red
)

# Add a horizontal dotted line at y=0.7
plt.axhline(0.7, color='gray', linestyle='--', linewidth=2)

# Add labels and title
plt.xlabel('Hours', weight='bold')
plt.ylabel('Accuracy', weight='bold')
plt.ylim(0.0, 1.0)  # Set y-axis limits

# Customize the appearance
plt.tight_layout()  # Ensure everything fits well

# Save the plot
plt.savefig(
    os.path.join("..", "Results", "accuracy_bhours_line.png"), 
    dpi=500, 
    bbox_inches='tight'
)

# %%
# Get probabilities for the positive class (Human)
y_proba_6h_p = y_proba_6h[:, 1]  
y_proba_12h_p = y_proba_12h[:, 1]  
y_proba_24h_p = y_proba_24h[:, 1] 
y_proba_48h_p = y_proba_48h[:, 1]  

# Calculate ROC AUC for each time interval
# List of probabilities and corresponding true labels for each time interval
probas_list = [
    y_proba_6h_p, 
    y_proba_12h_p, 
    y_proba_24h_p, 
    y_proba_48h_p
]

true_list = [
    y_6h_encoded, 
    y_12h_encoded, 
    y_24h_encoded, 
    y_48h_encoded
]

labels_list = [
    '6 hours', 
    '12 hours', 
    '24 hours', 
    '48 hours'
]

# Create a figure for plotting
plt.figure(figsize=(6, 5))

# Loop over probabilities and true labels, and plot each ROC curve
for y_proba, y_true, label in zip(probas_list, true_list, labels_list):
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    # Plot the ROC curve
    plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')

# Plot the diagonal line for random classifier
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

# Customizations
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', weight='bold')
plt.ylabel('True Positive Rate', weight='bold')
# plt.title('ROC Curves for Different Time Intervals', weight='bold')
plt.legend(loc='lower right')
plt.tight_layout()

# Save the plot
plt.savefig(
    os.path.join("..", "Results", "roc_curves_bhours.png"), 
    dpi=500, 
    bbox_inches='tight'
)

# %%
