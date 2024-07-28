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
                                find_mean_from_combined_dicts,
                                human_blood_index
                            )

# from Human_blood_index import human_blood_index

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
                                f1_score, 
                                recall_score, 
                                precision_score
                            )

from imblearn.under_sampling import RandomUnderSampler
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

outdir = "C:\Mannu\Projects\Mannu Phd\Transfer_learning MLP"
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

savedir = (outdir+"\_transfer_learning")            
build_folder(savedir, False)
savedir = (outdir+"\_transfer_learning\l")            
           

#%%

# Load full field data

full_data_df = pd.read_csv(
                            "C:\Mannu\Projects\Sporozoite Spectra for An funestus s.s\Phd data\Analysis data\Biological_attr.dat", 
                            delimiter= '\t'
                        )
print(full_data_df.head())

# data shape
print(full_data_df.shape)

# Checking class distribution
Counter(full_data_df["Cat7"])

#%%

# Select only abdomen bloodfed data

blood_field_df = full_data_df.query("Cat6 == 'BF' and Cat7 == 'AB'")
print('The shape of field blood meal source data : {}'.format(blood_field_df.shape))

# Observe first few observations
blood_field_df.head()

# %%

# Import PCR results which contains the ID's of blood fed mosquitoes 
pcr_data_df = pd.read_csv("C:\Mannu\Projects\Mannu Phd\MIRS_blood_meal_PCR.csv")
print(pcr_data_df.head(5))

# Now select only human fed samples 

human_df = pcr_data_df.query("PCR_RESULTS == 'Human'")
print(Counter(human_df["PCR_RESULTS"])) # Checking class distribution

# Now select only bovine fed samples 

bovine_df = pcr_data_df.query("PCR_RESULTS == 'Bovine'")
print(Counter(bovine_df["PCR_RESULTS"])) # Checking class distribution

# %%

# Select a vector of sample ID from PCR data and use it to index all the human blood samples
# from the blood meal field data

human_b_samples = human_df['SAMPLE_ID']
human_b_samples_df = blood_field_df.query("ID in @human_b_samples")

# create a new column in positive samples dataframe and name the samples as human
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

# Concatinating human and bovine bloodfed dataframes together

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
# human_bov_bldfed_df = human_bov_bldfed_df.drop(['ID', 'Cat2', 'Cat3', 'Cat4', 'Cat5', 'Cat6', 'Cat7', 'StoTime'], axis=1)
first_column = human_bov_bldfed_df.pop('blood_meal')
  
# insert column using insert(position,column_name,
# first_column) function
human_bov_bldfed_df.insert(
                            0, 
                            'blood_meal', 
                            first_column
                        )
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

X_new = human_bov_bldfed_df.iloc[:,1:] # select all columns except the last one
y_new = human_bov_bldfed_df["blood_meal"]

# scaler
#  Define the base directory
base_directory = r"C:\Mannu\Projects\Mannu Phd\Transfer_learning MLP\_transfer_learning"

# Create a function to generate paths within the base directory
def generate_path(*args):
    return os.path.join(base_directory, *args)

scaler = joblib.load(generate_path('data_scaler.joblib'))
temp = scaler.transform(np.asarray(X_new))
X_new_2 = pd.DataFrame(temp, columns = X_new.columns)

# random under sampler
rus = RandomUnderSampler(random_state = 4)
X_res, y_res = rus.fit_resample(X_new_2, y_new)
print("Class distribution after resampling:", Counter(y_res))

# Get the indices of the samples that were not resampled
indices_not_resampled = np.setdiff1d(
                                        np.arange(len(X_new)), 
                                        rus.sample_indices_
                                    )

# Create a DataFrame with the remaining samples
remaining_samples = pd.DataFrame(
                                    X_new.iloc[indices_not_resampled], 
                                    columns = X_new.columns
                                )

remaining_samples['blood_meal'] = y_new.iloc[indices_not_resampled]

# Print the class distribution after resampling
print("Class distribution of remaining samples:", Counter(remaining_samples['blood_meal']))

# shift column 'Name' to first position
first_column = remaining_samples.pop('blood_meal')
  
# insert column using insert(position,column_name, first_column) function
remaining_samples.insert(
                            0, 
                            'blood_meal', 
                            first_column
                        )
remaining_samples

# Transform matrix of features and label into numpy arrays
X_res = np.asarray(X_res)
y_res = np.asarray(y_res)

# %%

# Apply transfer learning to the pre-trained model 
# load a pre-trained deep learning model saved to disk

model = tf.keras.models.load_model(generate_path("lBaseline_CNN_0_3_Model.tf"))
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

# Define the list of test sizes you want to use
test_sizes = [0.98, 0.97, 0.95, 0.94, 0.92, 0.90, 0.85, 0.80, 0.75, 0.70]

# Dictionary to store the results
results = {}

# For loop to iterate over different test sizes and store the results

start_time = time()

num_rounds = 10
test_results = [] # List to store results for each round

for test_size in test_sizes:
    for round in range (num_rounds):
        random_s = np.random.randint(0, 8147)

        X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
                                                                                X_res, 
                                                                                y_res, 
                                                                                test_size = test_size, 
                                                                                random_state = random_s, 
                                                                                shuffle = True
                                                                            )

        # save x_test_new for later 
        X_test_new_2 = X_test_new

        print('The shape of X train index : {}'.format(X_train_new.shape))
        print('The shape of y train index : {}'.format(y_train_new.shape))
        print('The shape of X test index : {}'.format(X_test_new.shape))
        print('The shape of y test index : {}'.format(y_test_new.shape))

        # prepare Y_values for the transfer learning - training

        host_list_trans = [
                            [host_trans] for host_trans in y_train_new
                        ]
        hosts_trans = MultiLabelBinarizer().fit_transform(host_list_trans)
        y_classes_trans = list(np.unique(host_list_trans))
        print('Y classes trans', y_classes_trans)
        print('y classes trans binarized', hosts_trans)

        labels_default_trans, classes_default_trans = [hosts_trans], [y_classes_trans]

        # prepare Y_values for the transfer learning - validation

        host_list_val = [[host_val] for host_val in y_test_new]
        hosts_val = MultiLabelBinarizer().fit_transform(host_list_val)
        y_classes_val = list(np.unique(host_list_val))
        print('Y classes trans', y_classes_val)
        print('y classes trans binarized', hosts_val)

        labels_default_val, classes_default_val = [hosts_val], [y_classes_val]

        # reshape data, and train the model
        X_train_new = X_train_new.reshape([X_train_new.shape[0], -1])
        X_test_new = X_test_new.reshape([X_test_new.shape[0], -1])

        history_transfer_lr = transfer_lr_model.fit(x = X_train_new, 
                                    y = np.squeeze(labels_default_trans),
                                    batch_size = 256, 
                                    verbose = 1, 
                                    epochs = 3000,
                                    validation_data = (X_test_new, labels_default_val),
                                    callbacks = [
                                        tf.keras.callbacks.EarlyStopping(
                                            monitor = 'val_loss', 
                                            patience = 50, 
                                            verbose = 1, 
                                            mode = 'auto'
                                            ), 
                                        CSVLogger(
                                            savedir 
                                            + 'transfer_logger.csv', 
                                            append = True, 
                                            separator = ';')
                                            ]
                                        )

        # transfer_lr_model.save("C:\Mannu\Projects\Mannu Phd\Transfer_learning MLP\_transfer_learning\Transfer_lr_model%.tf")
        
        # Evaluate model
        # Make predictions using a model trained with transfer learning
        # change the dimension of y_test to array

        # y_test_v = np.asarray(labels_default_val)
        y_test_v = np.squeeze(labels_default_val) # remove any single dimension entries from the arrays

        # generates output predictions based on the X_input passed

        predictions = transfer_lr_model.predict(X_test_new)

        y_pred_classes = np.argmax(predictions, axis = -1)
        accuracy_tl = accuracy_score(
                                        np.argmax(y_test_v, axis= -1), 
                                        y_pred_classes
                                    )

        # Calculate confusion matrix

        cm = confusion_matrix(
                                np.argmax(y_test_v, axis= -1), 
                                y_pred_classes
                            )

        cm_df = pd.DataFrame(
                                cm, 
                                index = ['Actual : Bovine ','Actual : Human'], 
                                columns = ['Predict : Bovine','Predict :Human']
                            )

        total_predicted_tl = sum(cm_df['Predict : Bovine']) + sum(cm_df['Predict :Human'])
        number_human_pred_tl = cm_df['Predict :Human'][1] + cm_df['Predict :Human'][0]
        actual_tl_human = cm_df.iloc[1, 0] + cm_df.iloc[1, 1]

        # calculate HBI
        # for PCR
        HBI_pcr_tl = human_blood_index(actual_tl_human, total_predicted_tl)
        print('HBI estimated by PCR_tl', np.round(HBI_pcr_tl, 2))

        # for MLP predictions
        HBI_tl = human_blood_index(number_human_pred_tl, total_predicted_tl)
        print('HBI estimated by tl', np.round(HBI_tl, 2))

        # Store the result for each round
        test_results.append(
            {
                'Test Size': test_size, 
                'Round': round + 1, 
                'num_samples': X_train_new.shape[0], 
                'accuracy': accuracy_tl,
                'PCR': HBI_pcr_tl,
                'MLP-TL': HBI_tl

            }
        )

# # Print the results
# for test_size, test_results in results.items():
#     print(f"Test Size: {test_size}")
#     for round, result in enumerate(test_results):
#         print(f"Round: {round + 1}, Num Samples: {result['num_samples']}, Accuracy: {result['accuracy']}")

# Convert test_results to DataFrame
df_results = pd.DataFrame(test_results)

# # Print the DataFrame
# print(df_results)
df_results.to_csv(generate_path("_transfer_samples_20pb_final_last.csv"))

# save model to disk
transfer_lr_model.save(generate_path("Transfer_lr_model.tf"))

end_time = time()
print('Run time : {} s'.format(end_time-start_time))
print('Run time : {} m'.format((end_time-start_time)/60))
print('Run time : {} h'.format((end_time-start_time)/3600))


#%%

# Plot the line plot for accuracy with increased number of samples for transfer learning

ax = sns.lineplot(
                data = df_results, 
                x = 'num_samples', 
                y = 'accuracy', 
                marker = 'o'
                )
plt.yticks(np.arange(0.70, 0.90 + 0.01 , step = 0.02))
plt.xticks(np.arange(0, 100 + 10, step = 20))
plt.xlabel('Number of Samples', weight = 'bold')
plt.ylabel('Accuracy', weight = 'bold')
plt.savefig(
                generate_path('_transfer_samples_20pb_final_all_samples.png'), 
                dpi = 500, 
                bbox_inches = "tight"
            )
# plt.show()

#%%

# Plot line plot for HBI, for both PCR and MLP-TL estimated, in a single plot

ax = sns.lineplot(
                    data = df_results, 
                    x = 'num_samples', 
                    y = 'PCR', 
                    marker = 'o',
                    label = 'PCR'
                )
sns.lineplot(
                data = df_results, 
                x = 'num_samples', 
                y = 'MLP-TL', 
                marker = 'o',
                label = 'MLP-TL'
            )
plt.yticks(np.arange(0.0, 1.0 + .1, step = 0.2))
plt.xticks(np.arange(0, 100 + 10, step = 20))
plt.xlabel('Number of Samples', weight = 'bold')
plt.ylabel('HBI', weight = 'bold')
plt.legend()
plt.savefig(
                generate_path('HBI_final_all_samples.png'), 
                dpi = 500, 
                bbox_inches = "tight"
            )
# plt.show()

#%%

# Make predictions using a model trained with transfer learning
# change the dimension of y_test to array

# y_validation = np.asarray(labels_default_val)
y_validation = np.squeeze(labels_default_val) # remove any single dimension entries from the arrays

# generates output predictions based on the X_input passed

predictions = transfer_lr_model.predict(X_test_new)

# computes the loss based on the X_input you passed, along with any other metrics requested in the metrics param 
# when model was compiled
 
score = transfer_lr_model.evaluate(X_test_new, y_validation, verbose = 1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Calculating precision, recall and f-1 scores metrics for the predicted samples 

cr_pca = classification_report(
                                np.argmax(y_validation, axis=-1), 
                                np.argmax(predictions, axis=-1)
                            )
print(cr_pca)

# save classification report to disk 
cr = pd.read_fwf(io.StringIO(cr_pca), header = 0)
cr = cr.iloc[0:]
cr.to_csv(generate_path('cr_report_TL_20pb_all_samples.csv'))

#%%

# Plot the confusion matrix for predcited samples 

visualize(
            1, 
            savedir, 
            model_name, 
            "Transfer_learning_20pb_final_temp",
            classes_default_val,
            predictions, 
            y_validation
        )

# %%

cm_labeld_transfer = confusion_matrix(
                                        np.argmax(y_validation, axis = -1), 
                                        np.argmax(predictions, axis = -1)
                                    )

cm_2 = pd.DataFrame(
                        cm_labeld_transfer, 
                        index = ['Actual : Bovine ','Actual : Human'], 
                        columns = ['Predict : Bovine','Predict :Human']
                    )
# cm_2.to_csv("C:\Mannu\Projects\Mannu Phd\Final analysis\Results\Logistic regression\cm_labeld_transfer_20pb.csv")

# %%

# Save in true and predicted values to text file

combined_prediction_array = np.vstack(
                                        (
                                            np.argmax(y_validation, axis = -1),
                                            np.argmax(predictions, axis = -1)
                                        )
                                    )

column_names = ['y_test', 'predictions']

prediction_dict = {
    column: combined_prediction_array[i, :].tolist() for i, column in enumerate(column_names)
    }

with open(
    generate_path('predictions_final.txt'), 
    'w'
    ) as outfile:
     json.dump(
         prediction_dict, 
         outfile
         )

#%%

# Use transfer learning to make predictions on the imbalanced dataset

X_test_new_df = pd.DataFrame(X_test_new_2, columns = X_new.columns)
y_test_new_df = pd.DataFrame(y_test_new)

test_samples_df = pd.concat(
                                [
                                    y_test_new_df,
                                    X_test_new_df
                                ],
                                axis = 1,
                                join = 'outer'
                            )                                           

test_samples_df.rename(columns = {0:'blood_meal'}, inplace = True)

# concationate test data with samples that were not undersampled 

all_test_samples_df = pd.concat(
                                    [
                                        remaining_samples,
                                        test_samples_df
                                    ],
                                    axis = 0,
                                    join = 'inner'
                                    ).reset_index(drop = True)

# select matrix of features and vector of labels
all_test_X = np.asarray(all_test_samples_df.iloc[:, 1:])
all_test_y = np.asarray(all_test_samples_df['blood_meal'])

# Binarize labels for test with Non-undersampled samples  
host_list_all = [[host_all] for host_all in all_test_y]
hosts_all_test = MultiLabelBinarizer().fit_transform(host_list_all)
y_classes_all_test = list(np.unique(host_list_all))
print('Y classes trans', y_classes_all_test)
print('y classes trans binarized', hosts_all_test)

labels_default_all, classes_default_all = [hosts_all_test], [y_classes_all_test]

# Reshape features  
all_test_X = all_test_X.reshape([all_test_X.shape[0], -1])

# Make predictions using a model trained with transfer learning
# change the dimension of y_test to array

# y_all_sp = np.asarray(labels_default_al)
y_all_sp = np.squeeze(labels_default_all) # remove any single dimension entries from the arrays

# generates predictions

predictions_all_sp = transfer_lr_model.predict(all_test_X)

# computes the loss based on the X_input you passed, along with any other metrics requested in the metrics param 
# when model was compiled

score_all_sp = transfer_lr_model.evaluate(all_test_X, y_all_sp, verbose = 1)
print('Test loss:', score_all_sp[0])
print('Test accuracy:', score_all_sp[1])

# Calculating precision, recall and f-1 scores metrics for the predicted samples 

cr_all_sp = classification_report(np.argmax(y_all_sp, axis = -1), np.argmax(predictions_all_sp, axis = -1))
print(cr_all_sp)

# save classification report to disk 
cr_sp = pd.read_fwf(io.StringIO(cr_all_sp), header = 0)
cr_sp = cr_sp.iloc[0:]
cr_sp.to_csv(generate_path('cr_TL_all_sp.csv'))

#%%

# Plot the confusion matrix for predcited samples 

visualize(
            2, 
            savedir, 
            model_name, 
            "Transfer_learning_20pb_final_all_samples",
            classes_default_val,
            predictions_all_sp, 
            y_all_sp
        )

# %%
