
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
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    classification_report, 
    max_error, 
    precision_recall_fscore_support, 
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
            font_scale = 2.0,
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
                            classes,
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
                vmin = .2, 
                vmax = 1.0,  
                cmap = cmap
            )
        
    # plt.title(title)
    plt.colorbar()
    classes = classes
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
                    horizontalalignment = "center",
                    color = "white" if cm[i, j] > thresh else "black"
                )

    plt.tight_layout()
    plt.ylabel('True Host', weight = 'bold')
    plt.xlabel('Predicted Host', weight = 'bold')
    plt.savefig(
        (
            "C:\Mannu\Projects\Mannu Phd\Final analysis\Results\Logistic regression\Confusion_Matrix_" 
            + figure_name 
            + "_" 
            + ".png"
        ), 
        dpi = 500, 
        bbox_inches = "tight"
        
    )

#%%

#  Define the base directory
base_directory = r"C:\Mannu\Projects\Mannu Phd\Final analysis\Results\Logistic regression"

# Create a function to generate paths within the base directory
def generate_path(*args):
    return os.path.join(base_directory, *args)

#%%

def visualize(figure_name, classes, predicted, true):
    # Sort out predictions and true labels
    # for label_predictions_arr, label_true_arr, classes, outputs in zip(predicted, true, classes, outputs):
    # print('visualize predicted classes', predicted)
    # print('visualize true classes', true)
    classes_pred = np.asarray(predicted)
    classes_true = np.asarray(true)
    print(classes_pred.shape)
    print(classes_true.shape)
    # classes = ['1-9', '10-17']
    cnf_matrix = confusion_matrix(
                                    classes_true, 
                                    classes_pred, 
                                    labels = classes
                                )
    plot_confusion_matrix(cnf_matrix, classes)


#%%

full_data_df = pd.read_csv(
                            "C:\Mannu\Projects\Sporozoite Spectra for An funestus s.s\Phd data\Analysis data\Biological_attr_2.dat", 
                            delimiter= '\t'
                        )

full_data_df.head()

#%%

# data shape
print(full_data_df.shape)

# Checking class distribution
Counter(full_data_df["Cat7"])

#%%

# Select abdomen data == funestus

blood_field_df = full_data_df.query(
                                        "Cat6 == 'BF' and Cat7 == 'AB' and Cat2 == 'AF'"
                                    )

print('The shape of field blood meal source data : {}'.format(blood_field_df.shape))

# Observe first few observations

blood_field_df.head()
Counter(blood_field_df["Cat2"])

# %%

# Import PCR results which contains the ID's of blood fed mosquitoes 
pcr_data_df = pd.read_csv(
                            "C:\Mannu\Projects\Mannu Phd\MIRS_blood_meal_PCR.csv"
                        )
print(pcr_data_df.head(5))

# Now select only human fed samples 

human_df = pcr_data_df.query("PCR_RESULTS == 'Human'")
# human_df = pcr_data_df.query('PCR_RESULTS in ["Human", "Human_cow", "Human_dog"]')
print(Counter(human_df["PCR_RESULTS"])) # Checking class distribution

# Now select only bovine fed samples 

bovine_df = pcr_data_df.query("PCR_RESULTS == 'Bovine'")
# bovine_df = pcr_data_df.query('PCR_RESULTS in ["Bovine", "Dog", "Cow_dog"]')
print(Counter(bovine_df["PCR_RESULTS"])) # Checking class distribution

# %%

# Select a vector of sample ID from PCR data and use it to index all the human blood samples
# from the blood meal field data

human_b_samples = human_df['SAMPLE_ID']
human_b_samples_df = blood_field_df.query("ID in @human_b_samples")

# create a new column in positive samples dataframe and name the samples as humans
human_b_samples_df['blood_meal'] = 'Human'
human_b_samples_df

#%%

# Select a vector of sample ID from PCR data and use it to index all the bovine blood samples
# from the blood meal field data

bovine_b_samples = bovine_df['SAMPLE_ID']
bovine_b_samples_df = blood_field_df.query("ID in @bovine_b_samples")

# create a new column in positive samples dataframe and name the samples as bovines
bovine_b_samples_df['blood_meal'] = 'Bovine'
bovine_b_samples_df

# %%

# Concatinating human and bovine blood-fed dataframes together

human_bov_bldfed_df = pd.concat(
                                    [
                                        human_b_samples_df, 
                                        bovine_b_samples_df
                                    ], 
                                    axis = 0, 
                                    join = 'outer'
                                )
# human_bov_bldfed_df.to_csv("E:\Sporozoite\human_bov_bldfed_df.csv")

# Drop unused columns
human_bov_bldfed_df = human_bov_bldfed_df.drop(
                                                [
                                                    'ID', 
                                                    'Cat2', 
                                                    'Cat3', 
                                                    'Cat4', 
                                                    'Cat5', 
                                                    'Cat6', 
                                                    'Cat7', 
                                                    'StoTime'
                                                ], 
                                                axis = 1
                                            )

human_bov_bldfed_df


# %%

# define X (matrix of features) and y (list of labels)

X = human_bov_bldfed_df.iloc[:,:-1] # select all columns except the last one
features = X 
y = human_bov_bldfed_df["blood_meal"]
print("Class distribution before resampling:", Counter(human_bov_bldfed_df["blood_meal"]))

print('shape of X : {}'.format(X.shape))
print('shape of y : {}'.format(y.shape))

# # make a list containing all column names
# col_names = features.columns.tolist()

# col_names = [int(x) for x in col_names]

# # get the closest wavenumbers
# print(list(map(lambda y:min(col_names, key = lambda x:abs(x - y)), [2400])))
# print(list(map(lambda y:min(col_names, key = lambda x:abs(x - y)), [1720])))

# # set the start and end column names as integers
# start_col = 2401 
# end_col = 1721 

# # get the column names between the start and end column
# cols_to_drop = [str(i) for i in range(start_col, end_col - 1, -2)]

# X_new = X.drop(cols_to_drop, axis = 1)

# rescalling the data (undersampling the over respresented class - Bovine fed class)

rus = RandomUnderSampler(random_state = 42)
X_res, y_res = rus.fit_resample(X, y)
print("Class distribution after resampling:", Counter(y_res))

# Get the indices of the samples that were not resampled
indices_not_resampled = np.setdiff1d(
                                        np.arange(len(X)), 
                                        rus.sample_indices_
                                    )

# Create a DataFrame with the remaining samples
remaining_samples = pd.DataFrame(
                                    X.iloc[indices_not_resampled], 
                                    columns = X.columns
                                )

remaining_samples['blood_meal'] = y.iloc[indices_not_resampled]

# Print the class distribution after resampling
print("Class distribution after resampling:", Counter(y_res))
print("Class distribution of remaining samples:", Counter(remaining_samples['blood_meal']))

# shift column 'Name' to first position
first_column = remaining_samples.pop('blood_meal')
  
# insert column using insert(position,column_name, first_column) function
remaining_samples.insert(0, 'blood_meal', first_column)

# split the data into train and out of sample/unseen dataset/test set

X_train, X_test, y_train, y_test = train_test_split(
                                                        X_res, 
                                                        y_res, 
                                                        test_size= .1, 
                                                        random_state = 42,
                                                        shuffle = True
                                                    )

print('The shape of X train index : {}'.format(X_train.shape))
print('The shape of y train index : {}'.format(y_train.shape))
print('The shape of X test index : {}'.format(X_test.shape))
print('The shape of y test index : {}'.format(y_test.shape))

# Concat resampled y_test and X_test into dataframe, only resampled samples
test_set = pd.concat(
                        [
                            y_test, 
                            X_test
                        ], 
                        axis = 1, 
                        join = 'outer'
                    ) 

# create a combine test_set with the remaining samples that were not undersampled
test_set_2 = pd.concat(
                        [
                            test_set, 
                            remaining_samples
                        ], 
                        axis = 0, 
                        join = 'inner'
                    ).reset_index(drop = True)

# save test set for calculating Blood index
# HBI_testset = pd.concat([y_test, X_test], axis = 1, join = 'outer')
# HBI_testset.to_csv("C:\Mannu\Projects\Mannu Phd\Final analysis\HBI_testset.csv")


#%%

# Standardise inputs using standard scaler

X = np.asarray(X_train)
y = np.asarray(y_train)
print('y labels : {}'.format(np.unique(y)))

# Data standardisation 

scaler = StandardScaler().fit(X = X)
X_transformed  = scaler.transform(X = X)

# %%

# Data splitting and defining models

num_folds = 5 # Spliting the training set into 6 parts
validation_size = 0.1 # defining the size of the validation set
seed = 42 # you can choose any integer, this ensures reproducibility of the tests
scoring = 'accuracy' # score model accuracy

random_seed  = np.random.randint(0, 81478)
kf = KFold(
            n_splits = num_folds, 
            shuffle = True, 
            random_state = random_seed
        )

models = [] # telling python to create sub names models
models.append(
                (
                    "KNN", KNeighborsClassifier()
                )
            )
models.append(
                (
                    "LR", LogisticRegression(
                                                multi_class = 'ovr', 
                                                random_state = seed, 
                                                max_iter = 3500
                                            )
                )
            )
models.append(
                (
                    "SVM", SVC(
                                random_state = seed, 
                                kernel = 'linear', 
                                gamma = 'auto'
                            )
                )
            )
models.append(
                (
                    "XGB", XGBClassifier(
                                            random_state = seed, 
                                            n_estimators = 1000
                                        )
                )
            )
models.append(
                (
                    "RF", RandomForestClassifier(
                                                    random_state = seed, 
                                                    n_estimators = 1000
                                                )
                )
            )

# models.append(("MLP", MLPClassifier(random_state=seed, max_iter = 3500,
#                                     solver = 'sgd',
#                                     activation = 'logistic', alpha = 0.001)))


#%%

# comparative evaluation of different classifiers

results = []
names = []

for name, model in models:
    cv_results = cross_val_score(
                                    model, 
                                    X_transformed, 
                                    y, 
                                    cv = kf, 
                                    scoring = scoring
                                )
    results.append(cv_results)
    names.append(name)
    msg = "Cross val score for {0}: {1:.2%} ± {2:.2%}".format(
                                                                name, 
                                                                cv_results.mean(), 
                                                                cv_results.std()
                                                            )   
    print(msg)


#%%

# plotting the results of the classifier

# create a dataframe for plotting using seaborn

result_df = pd.DataFrame(
                            results, 
                            columns = (0, 1, 2, 3, 4)
                        ).T # columns should correspond to the number of folds, k = 5

# rename columns to have number of components

result_df.columns = names #['KNN', 'LR', 'SVM', 'XGB', 'RF']
result_df = pd.melt(result_df) # melt data frame into a long format. 
result_df.rename(
                    columns = {'variable':'Model', 'value':'Accuracy'}, 
                    inplace = True
                )

# Import MLP results for MLP-DL predictions (k=5)
mlp_acc = pd.read_csv(generate_path('MLP_accuracy.csv'))

# join data (traditional ML accuracy and deep learning accuracy)
results_df  = pd.concat(
                            [
                                result_df, 
                                mlp_acc
                            ], 
                            axis = 0, 
                            join = 'outer'
                        ).reset_index(drop = True)

sns.boxplot(
            x = results_df['Model'], 
            y = results_df['Accuracy']
        )

sns.despine(offset = 10, trim = True)
# plt.title("Algorithm comparison", weight="bold")
plt.xticks(rotation = 90)
plt.yticks(np.arange(0.2, 1.0 + .05, step = 0.2))
plt.ylabel('Accuracy', weight = 'bold')
plt.xlabel(" ");
plt.savefig(
            (generate_path("_algo_selection.png")), 
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

start = time()

# Specify model

classifier = LogisticRegression(
                                    multi_class = 'ovr', 
                                    random_state = random_seed, 
                                    max_iter = 3500
                                )

# Optimizing hyper-parameters for logistic regression

solvers = ['newton-cg', 'lbfgs', 'liblinear']
c_values = [0.01, 0.1, 1, 10]     # 100, 50, 20,  10, 5]
    
# Create the random grid

random_grid = {
                'solver': solvers,
                'C': c_values
            }

for round in range (num_rounds):
    SEED = np.random.randint(0, 8147)

    # cross validation and splitting of the validation set
    
    for train_index, test_index in kf.split(X_transformed, y):
        X_train_set, X_val = X_transformed[train_index], X_transformed[test_index]
        y_train_set, y_val = y[train_index], y[test_index]

        print('The shape of X train set : {}'.format(X_train_set.shape))
        print('The shape of y train set  : {}'.format(y_train_set.shape))
        print('The shape of X val set : {}'.format(X_val.shape))
        print('The shape of y val set : {}'.format(y_val.shape))

        # generate models using all combinations of settings

        # RANDOMSED GRID SEARCH
        # Random search of parameters, using 5 fold cross validation, 
        # search across 100 different combinations, and use all available cores

        n_iter_search = 100
        gridCV = GridSearchCV(
                                verbose = 1,
                                estimator = classifier, 
                                param_grid = random_grid, 
                                scoring = scoring, 
                                cv = kf, 
                                refit = True, 
                                n_jobs = -1
                            )
            
        gridCV_result = gridCV.fit(X_train_set, y_train_set)

        # print out results and give hyperparameter settings for best one
        means = gridCV_result.cv_results_['mean_test_score']
        stds = gridCV_result.cv_results_['std_test_score']
        params = gridCV_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%.2f (%.2f) with: %r" % (mean, stdev, param))

        # print best parameter settings
        print("Best: %.2f using %s" % (
                                        gridCV_result.best_score_,
                                        gridCV_result.best_params_
                                        )
                                    )

        # Insert the best parameters identified by randomized grid search into the base classifier
        best_classifier = classifier.set_params(
                                                **gridCV_result.best_params_, 
                                                n_jobs = -1
                                            )
        
        best_classifier.fit(X_train_set, y_train_set)

        # predict test instances 

        y_pred = best_classifier.predict(X_val)
        # y_test = np.delete(y_res, train_index, axis=0)
        local_cm = confusion_matrix(y_val, y_pred)
        local_report = classification_report(y_val, y_pred)

        # zip predictions for all rounds for plotting averaged confusion matrix
            
        for predicted, true in zip(y_pred, y_val):
            save_predicted.append(predicted)
            save_true.append(true)

        # # append feauture importances
        coef_table = pd.Series(best_classifier.coef_[0], features.columns)
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
                                                ("y_test", y_test)
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

# plot confusion averaged for the validation set

classes = np.unique(np.sort(y))
figure_name = 'baseline_model_2'

visualize(
            figure_name, 
            classes, 
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

# plotting coefficients
n_features = 25
# coef = pd.read_csv("coef_repeatedCV_coef.csv")
# coef_2 = coef.rename(columns = {'Unnamed: 0': 'Wavenumbers'})
# coef_3 = coef_2.reset_index().set_index('Wavenumbers')

sss_coef_2.sort_values(
                        by = "coef mean", 
                        ascending = False, 
                        inplace = True
                    )

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
# coef_plot_data_late = coef_plot_data

# Plotting 

plt.figure(figsize = (6,16))

sns.barplot(
                data = coef_plot_data, 
                orient = "h", 
                palette = "plasma", 
                capsize = .2
            )

plt.ylabel("Wavenumbers", weight = "bold")
plt.xlabel("Coefficients", weight = "bold")
plt.savefig(
                generate_path("lgr_coef.png"), 
                dpi = 300, 
                bbox_inches="tight"
            )

#%%

# save the trained model to disk for future use

with open(
    generate_path('classifier_2.pkl'), 
    'wb'
    ) as fid:
     pickle.dump(best_classifier, fid)

# %%

# Results

kf_results.to_csv(
                    generate_path("crf_kfCV_record.csv"), 
                    index = False
                )
kf_results = pd.read_csv(generate_path("crf_kfCV_record.csv"))

# Accuracy distribution

crf_acc_distrib = kf_results["Accuracy"]
crf_acc_distrib.columns=["Accuracy"]

crf_acc_distrib.to_csv(
                        generate_path("crf_acc_distrib.csv"), 
                        header = True, 
                        index = False
                    )

crf_acc_distrib = pd.read_csv(generate_path("crf_acc_distrib.csv"))
crf_acc_distrib = np.round(crf_acc_distrib, 2)
print(crf_acc_distrib)


#%%

# plotting accuracy distribution

plt.figure(figsize=(2.25,3))
sns.distplot(
                crf_acc_distrib, 
                kde = False, 
                bins = 12
            )
# plt.savefig("lgr_acc_distrib.png", bbox_inches="tight")

#%%

# class distribution
 
class_names = np.unique(np.sort(y))
crf_per_class_acc_distrib = pd.DataFrame(kf_per_class_results, columns = class_names)
crf_per_class_acc_distrib.dropna().to_csv(generate_path("crf_per_class_acc_distrib.csv"))

crf_per_class_acc_distrib = pd.read_csv(
                                            generate_path("crf_per_class_acc_distrib.csv"), 
                                            index_col = 0
                                        )

crf_per_class_acc_distrib = np.round(crf_per_class_acc_distrib, 2)
crf_per_class_acc_distrib_describe = crf_per_class_acc_distrib.describe()
crf_per_class_acc_distrib_describe.to_csv(generate_path("crf_per_class_acc_distrib.csv"))


#%%

# plotting class distribution

lgr_per_class_acc_distrib = pd.melt(crf_per_class_acc_distrib, var_name = "status new")


plt.figure(figsize = (6,4))

sns.violinplot(
                x = "status new", 
                y = "value", 
                cut = 1, 
                data = lgr_per_class_acc_distrib
            )

sns.despine(left = True)
plt.xticks(rotation = 0, ha = "right")
plt.xticks()
plt.yticks(np.arange(0.2, 1.0 + .05, step = 0.2))
plt.xlabel(" ")
plt.ylabel("Prediction accuracy", weight = "bold")

plt.savefig(
                generate_path("per_class_acc_distrib_full_wn.png"), 
                dpi = 500, 
                bbox_inches = "tight"
            )


#%%

# Predict test data with resampled/balanced dataset

# scale data with standard scaler 

trans_X_test = scaler.transform(X = np.asarray(X_test))

# Predict blood meal
y_test_pred = best_classifier.predict(trans_X_test)

accuracy_val = accuracy_score(y_test, y_test_pred)
print("Accuracy: %.2f%%" % (accuracy_val * 100.0))

#%%

# Plotting confusion matrix for test set 

classes = np.unique(np.sort(y_test))
figure_name = 'test_prediction_2'

visualize(
            figure_name, 
            classes,  
            y_test_pred, 
            y_test
        )

#%%

# Summarising precision, f_score, and recall for the test set

cr_full_wn_val = classification_report(y_test, y_test_pred)
print('Classification report : {}'.format(cr_full_wn_val))

# save classification report to disk as a csv

cr_full_wn_val = pd.read_fwf(
                            io.StringIO(cr_full_wn_val), 
                            header = 0  
                            )
                            
cr_full_wn_val = cr_full_wn_val.iloc[0:]
cr_full_wn_val.to_csv(generate_path("classification_report_2.csv"))

#%%

def my_logistic_report(X_test_, y_test_, threshold=0.5): # you could make it even more general!
    
    y_hat = best_classifier.predict(X_test_)#[:,1]
    print(y_hat)
    # y_hat = np.where(probs>=threshold,1,0)
    
    cm = confusion_matrix(y_test_, y_hat)
    accuracy = accuracy_score(y_test_, y_hat)
    precision = precision_score(y_test_, y_hat, pos_label = 'Bovine')
    recall = recall_score(y_test_, y_hat, pos_label = 'Bovine')
    f1score = f1_score(y_test_, y_hat, pos_label = 'Bovine')
    cm_labeld = pd.DataFrame(
                                cm, 
                                index = ['Actual : Bovine ','Actual : Human'], 
                                columns = ['Predict : Bovine','Predict :Human']
                            )
    
    print("-----------------------------------------")
    print('Accuracy  = {}'.format(accuracy))
    print('Precision = {}'.format(precision))
    print('Recall    = {}'.format(recall))
    print('f1_score  = {}'.format(f1score))
    print("-----------------------------------------")
    return cm_labeld

cm_labeld_ = my_logistic_report(trans_X_test, y_test)
# cm_labeld_.to_csv("C:\Mannu\Projects\Mannu Phd\Final analysis\Results\Logistic regression\cm_labeld_2.csv")


#%%

# Prepare the remaining samples, use the imbalanced dataset for prediction
# Select abdomen data == gambiae (AG)

blood_field_df_ar = full_data_df.query(
                                        "Cat6 == 'BF' and Cat7 == 'AB' and Cat2 == 'AG'"
                                    )

print('The shape of field blood meal source data : {}'.format(blood_field_df_ar.shape))

# Observe first few observations

blood_field_df_ar.head()
Counter(blood_field_df_ar["Cat2"])

#%%
# Select a vector of sample ID from PCR data and use it to index all the human blood samples
# from the blood meal field data

human_b_samples_df_ar = blood_field_df_ar.query("ID in @human_b_samples")

# create a new column in positive samples dataframe and name the samples as human
human_b_samples_df_ar['blood_meal'] = 'Human'
human_b_samples_df_ar

#%%

# Select a vector of sample ID from PCR data and use it to index all the bovine blood samples
# from the blood meal field data

bovine_b_samples_df_ar = blood_field_df_ar.query("ID in @bovine_b_samples")

# create a new column in positive samples dataframe and name the samples as Bovine
bovine_b_samples_df_ar['blood_meal'] = 'Bovine'
bovine_b_samples_df_ar

# %%

# Concatinating human and bovine blood-fed dataframes together

human_bov_bldfed_df_ar = pd.concat(
                                    [
                                        human_b_samples_df_ar, 
                                        bovine_b_samples_df_ar
                                    ], 
                                    axis = 0, 
                                    join = 'outer'
                                )
# human_bov_bldfed_df.to_csv("E:\Sporozoite\human_bov_bldfed_df.csv")

# Drop unused columns
human_bov_bldfed_df_ar = human_bov_bldfed_df_ar.drop(
                                                        [
                                                            'ID', 
                                                            'Cat2', 
                                                            'Cat3', 
                                                            'Cat4', 
                                                            'Cat5', 
                                                            'Cat6', 
                                                            'Cat7', 
                                                            'StoTime'
                                                        ], 
                                                        axis = 1
                                                    )

# insert column using insert(position,column_name, first_column) function
human_bov_bldfed_df_ar.insert(
                                0, 
                                'blood_meal', 
                                human_bov_bldfed_df_ar.pop('blood_meal')
                            )

# Combine An. funestus test dataset with An. arabiensis test dataset
#  
test_set_3 = pd.concat(
                        [
                            test_set_2, 
                            human_bov_bldfed_df_ar
                        ], 
                        axis = 0, 
                        join = 'outer'
                    ).reset_index(drop = True)


# %%

# Predict test set with all samples, imbalanced dataset

# prepare X_test_2 matrix and y_test_2 vector of labels 
X_test_2 = np.asarray(test_set_3.iloc[:, 1:])
y_test_2 = np.asarray(test_set_3['blood_meal'])
print("Class distribution of remaining samples:",  Counter(y_test_2))

#%%
# Scale data using standard scaler
trans_X_test_2 = scaler.transform(X = X_test_2)

# load model from disk

with open(generate_path('classifier_2.pkl'), 'rb') as fid:
    loaded_model = pickle.load(fid)

# Predict blood meal
y_test_pred_2 = loaded_model.predict(trans_X_test_2)

accuracy_test_2 = accuracy_score(y_test_2, y_test_pred_2)
print("Accuracy_2: %.2f%%" % (accuracy_test_2 * 100.0))

# %%

# Plotting confusion matrix for test set without undersampling 

classes_2 = np.unique(np.sort(y_test_2))
figure_name = 'test_prediction_no_under_sampling_2'
visualize(
            figure_name, 
            classes_2,  
            y_test_pred_2, 
            y_test_2
        )

# %%
# Summarising precision, f_score, and recall for the test set

cr_full_wn_test_2 = classification_report(y_test_2, y_test_pred_2)
print('Classification report : {}'.format(cr_full_wn_test_2))

# save classification report to disk as a csv

cr_full_wn_test_2 = pd.read_fwf(
                                    io.StringIO(cr_full_wn_test_2), 
                                    header = 0  
                                )
                            
cr_full_wn_test_2 = cr_full_wn_test_2.iloc[0:]
cr_full_wn_test_2.to_csv(generate_path("classification_report_no_rus_2.csv"))


#%%
# # Plot predicted class distribution for amplified samples

# unique_labels, counts = np.unique(y_test_pred_2, return_counts = True)

# plt.pie(
#             counts, 
#             labels = unique_labels, 
#             autopct = '%1.1f%%'
#         )

# plt.title('Predicted Class Distribution')
# plt.savefig(
#                 ("C:\Mannu\Projects\Mannu Phd\Final analysis\Results\Logistic regression\predicted_class_distr_ampl.png"), 
#                 dpi = 500, 
#                 bbox_inches = "tight"
#             )

# # %%

# # Predict PCR unapplified samples 

# temp_df = full_data_df.query(
#                                 "Cat6 == 'BF' and Cat7 == 'AB'"
#                                 )

# # Now select only un amplified samples 

# unamplified_df = pcr_data_df.query("PCR_RESULTS == 'N'")
# print(Counter(unamplified_df["PCR_RESULTS"])) # Checking class distribution

# # Select a vector of sample ID from PCR data and use it to index all the unamplified blood samples
# # from the blood meal field data

# unampl_b_samples = unamplified_df['SAMPLE_ID']
# unampl_b_samples_df = temp_df.query("ID in @unampl_b_samples")

# # create a new column in positive samples dataframe and name the samples as positives
# unampl_b_samples_df['blood_meal'] = 'Unamplified'
# unampl_b_samples_df


# # %%

# # Prepare X_test
# # Predict test set with all samples, not undersampled

# # prepare X_test_2 matrix and y_test_2 vector of labels 
# X_test_3 = np.asarray(unampl_b_samples_df.iloc[:, 8:-1])
# # y_test_2 = np.asarray(test_set_3['blood_meal'])

# # Transform data 
# trans_X_test_3 = scaler.transform(X = X_test_3)

# # load model from disk

# with open('C:\Mannu\Projects\Mannu Phd\Final analysis\Results\Logistic regression\classifier_2.pkl', 'rb') as fid:
#     loaded_model = pickle.load(fid)

# # Predict PCR unamplified blood meal
# y_test_pred_3 = loaded_model.predict(trans_X_test_3)

# # %%

# # Plot predicted class distribution
# unique_labels, counts = np.unique(y_test_pred_3, return_counts = True)

# plt.pie(
#             counts, 
#             labels = unique_labels, 
#             autopct = '%1.1f%%'
#         )

# plt.title('Predicted Class Distribution')
# plt.savefig(
#     ("C:\Mannu\Projects\Mannu Phd\Final analysis\Results\Logistic regression\predicted_class_distr.png"), 
#     dpi = 500, 
#     bbox_inches = "tight"
#     )

# %%

# # Save data for comparison
# # imbalanced

# temp_2 = test_set_3[
#                         [
#                             'Cat2', 
#                             'Cat3', 
#                             'Cat4', 
#                             'Cat5', 
#                             'Cat9', 
#                             'blood_meal'
#                         ]
#                     ].reset_index(drop = True)

# # temp_3 = pd.DataFrame(temp_1.groupby(['Cat2', 'Cat3', 'Cat4', 'Cat5', 'blood_meal']).size().reset_index())

# temp_2.rename(columns = {
#                             'Cat2':'Species', 
#                             'Cat3':'HH ID', 
#                             'blood_meal':'host_blood',
#                             'Cat4':'Trap', 
#                             'Cat5':'Position', 
#                             'Cat9': 'Date'
#                         }, inplace = True
#                     ) 

# temp_2['Trap'] = temp_2['Trap'].str.replace('RST', 'RBK')
# temp_2['Species'] = temp_2['Species'].str.replace('AG', 'arabiensis')
# temp_2['Species'] = temp_2['Species'].str.replace('AF', 'funestus')
# temp_2['Position'] = temp_2['Position'].str.replace('IN', 'Indoor')
# temp_2['Position'] = temp_2['Position'].str.replace('OUT', 'Outdoor')

# temp_2['Date'] = pd.to_datetime(
#                                     temp_2['Date'].astype(str), 
#                                     format = '%d%m%y', 
#                                     errors='coerce'
#                                 )

# temp_2.to_csv('C:\Mannu\Projects\Mannu Phd\Final analysis\Results\Logistic regression\imbalanced_test_set_3.csv', index = False)

%%

# # unamplified

# temp_3 = unampl_b_samples_df[
#                                 [
#                                     'Cat2', 
#                                     'Cat3', 
#                                     'Cat4', 
#                                     'Cat5', 
#                                     'Cat9', 
#                                     'blood_meal'
#                                 ]
#                             ].reset_index(drop = True)

# # temp_3 = pd.DataFrame(temp_1.groupby(['Cat2', 'Cat3', 'Cat4', 'Cat5', 'blood_meal']).size().reset_index())

# temp_3.rename(columns = {
#                             'Cat2':'Species', 
#                             'Cat3':'HH ID', 
#                             'blood_meal':'host_blood',
#                             'Cat4':'Trap', 
#                             'Cat5':'Position', 
#                             'Cat9': 'Date'
#                         }, 
#                         inplace = True
#                     ) 

# temp_3['Trap'] = temp_3['Trap'].str.replace('RST', 'RBK')
# temp_3['Species'] = temp_3['Species'].str.replace('AG', 'arabiensis')
# temp_3['Species'] = temp_3['Species'].str.replace('AF', 'funestus')
# temp_3['Position'] = temp_3['Position'].str.replace('IN', 'Indoor')
# temp_3['Position'] = temp_3['Position'].str.replace('OUT', 'Outdoor')

# temp_3['Date'] = pd.to_datetime(
#                                     temp_3['Date'].astype(str), 
#                                     format = '%d%m%y', 
#                                     errors = 'coerce'
#                                 )

# temp_3.to_csv(r'C:\Mannu\Projects\Mannu Phd\Final analysis\Results\Logistic regression\unamplified_test_set_3.csv', index = False)

