
#%%

import io
import json
import numpy as np 
import pandas as pd
from collections import Counter

from My_functions_DL import human_blood_index

import seaborn as sns
import matplotlib.pyplot as plt


#%% 

# Function to calculate Human bloodmeal index (HBI)

# def human_blood_index(beta, alpha):

#     '''
#     The formular to calculate blood index

#     beta: for PCR; Number of mosquitoes blood-fed on human/
#           for ML; Number of mosquitoes predicted as human blood-fed
#     alpha: for PCR; Total number of mosquitoes for ML testset (human & bovine)/
#            for ML; Total number of predicted blood-fed mosquitoes
#     '''
#     bi = beta/alpha

#     return bi

#%%

# calculate the human blood index MIRS prediction using field model

lr_pred_df = pd.read_csv('C:\Mannu\Projects\Mannu Phd\Final analysis\Results\Logistic regression\cm_labeld_2.csv')
print(lr_pred_df)

# Access the number of predicted and actual samples from the dataframe

total_predicted = sum(lr_pred_df['Predict : Bovine']) + sum(lr_pred_df['Predict :Human'])
number_human_pred = lr_pred_df['Predict :Human'][1] + lr_pred_df['Predict :Human'][0]
actual_lr_human = lr_pred_df.iloc[1, 1] + lr_pred_df.iloc[1, 2]

# calculate HBI
# for PCR
HBI_pcr_lr = human_blood_index(actual_lr_human, total_predicted)
print('HBI estimated by PCR', np.round(HBI_pcr_lr, 2))

# for LR predictions
HBI_lr = human_blood_index(number_human_pred, total_predicted)
print('HBI estimated by LR', np.round(HBI_lr, 2))

#%%

# calculate the human blood index MIRS prediction using transfered MLP model

tl_pred_df = pd.read_csv('C:\Mannu\Projects\Mannu Phd\Final analysis\Results\Logistic regression\cm_labeld_transfer_20pb.csv')
print(tl_pred_df)

# Access the number of predicted and actual samples from the dataframe

total_predicted_tl = sum(tl_pred_df['Predict : Bovine']) + sum(tl_pred_df['Predict :Human'])
number_human_pred_tl = tl_pred_df['Predict :Human'][1] + tl_pred_df['Predict :Human'][0]
actual_tl_human = tl_pred_df.iloc[1, 1] + tl_pred_df.iloc[1, 2]

# calculate HBI
# for PCR
HBI_pcr_tl = human_blood_index(actual_tl_human, total_predicted_tl)
print('HBI estimated by PCR_tl', np.round(HBI_pcr_tl, 2))

# for MLP predictions
HBI_tl = human_blood_index(number_human_pred_tl, total_predicted_tl)
print('HBI estimated by tl', np.round(HBI_tl, 2))

#%%

# Plot HBI's

data = {

    'Method': ['PCR', 'LR', 'PCR', 'TL'],
    'HBI': [
        np.round(HBI_pcr_lr, 2), 
        np.round(HBI_lr, 2), 
        np.round(HBI_pcr_tl, 2), 
        np.round(HBI_tl, 2)
        ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Set up the figure and axes
sns.set(

    context = "paper",
    style = "white",
    palette = "deep",
    font_scale = 2.0,
    color_codes = True,
    rc = ({"font.family": "Dejavu Sans"})
    
    )

plt.figure(figsize=(6, 4))

# Define the x-axis positions for each bar group
x_positions = [0, 1.0, 3.5, 4.5]

# Define custom colors for each method
colors = {

    'PCR': sns.color_palette("colorblind")[0], 
    'LR': sns.color_palette("colorblind")[1], 
    'TL': sns.color_palette("colorblind")[2]
    
    }

# Iterate through the DataFrame rows and create the custom bar plot
for index, row in df.iterrows():
    plt.bar(

        x_positions[index],
        row['HBI'],
        width = 0.5,
        color = colors[row['Method']],
        label = row['Method']
    
    )

# Set custom x-axis labels and title
x_labels = ['PCR', 'LR', 'PCR', 'TL']
plt.xticks(x_positions, x_labels)
plt.yticks(np.arange(0.0, 1.0 + .05, 0.2))
# plt.xlabel('Method')
plt.ylabel('HBI', weight = 'bold')
# plt.title('Comparison of HBI by Method')

# Show the legend
# plt.legend()

# Show the plot
plt.tight_layout()
plt.savefig(
    
    'C:\Mannu\Projects\Mannu Phd\Final analysis\Results\hbi.png', 
    dpi = 500, bbox_inches = "tight"
    
    )
# plt.show()

#%%

# import pandas as pd
# import scipy.stats as stats
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Create a pandas DataFrame with the data
# data = pd.read_csv('C:/Mannu/Projects/Mannu Phd/GLM_metadata/fishers_data.csv')
# data.loc[data['Species'] == 'arabiensis', 'Species'] = 'An. arabiensis'
# data.loc[data['Species'] == 'funestus', 'Species'] = 'An. funestus'
# # pd.DataFrame({
# #     'Species': ['arabiensis', 'arabiensis', 'arabiensis', 'arabiensis', 'arabiensis',
# #                 'arabiensis', 'arabiensis', 'arabiensis', 'arabiensis', 'arabiensis'],
# #     'Host': ['Human', 'Human', 'Bovine', 'Bovine', 'Bovine', 'Bovine',
# #              'Bovine', 'Bovine', 'Bovine', 'Bovine']
# # })

# # Create a contingency table from the DataFrame
# contingency_table = pd.crosstab(data['Species'], data['Host'])

# # Perform Fisher's exact test
# odds_ratio, p_value = stats.fisher_exact(contingency_table.iloc[:2, :2])

# # Print the odds ratio and p-value
# print("Odds Ratio:", odds_ratio)
# print("P-value:", p_value)

# # Calculate percentage values for each category
# percentage_table = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100

# # Plotting the stacked bar graph
# sns.set(context = "paper",
#         style = "white",
#         palette = "deep",
#         font_scale = 2.0,
#         color_codes = True,
#         rc = ({"font.family": "Dejavu Sans"}))
# plt.figure(figsize = (10, 8))
        
# # Plotting the stacked bar graph
# ax = percentage_table.plot(kind='bar', stacked=True)

# # Set the x-axis labels
# ax.set_xticklabels(percentage_table.index, rotation=0)

# plt.xlabel(' ')
# plt.ylabel('Percentage', weight='bold')

# # Modify the legend labels to include percentages for each species
# legend_labels = []
# for col in percentage_table.columns:
#     species_percentages = ", ".join([f"{idx} {val:.1f}%" for idx, val in percentage_table[col].items()])
#     legend_labels.append(f"{col}: {species_percentages}")

# # Set the modified legend labels
# ax.legend(labels=legend_labels, title='Host', bbox_to_anchor=(1.10, 1.05), loc='upper left')

# plt.show()

# %%
# import pandas as pd
# import scipy.stats as stats
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Create a pandas DataFrame with the data
# data = pd.read_csv('C:/Mannu/Projects/Mannu Phd/GLM_metadata/fishers_data.csv')
# data.loc[data['Species'] == 'arabiensis', 'Species'] = 'An. arabiensis'
# data.loc[data['Species'] == 'funestus', 'Species'] = 'An. funestus'

# # fast method
# filter_list = ['Human', 'Bovine', 'Dog']
# data = data[data.Host.isin(filter_list)]

# # Create a contingency table from the DataFrame
# contingency_table = pd.crosstab(data['Species'], data['Host'])

# # Perform chi-square test
# chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)

# # Print the chi-square test statistics and p-value
# print("Chi-square:", chi2)
# print("P-value:", p_value)

# # Calculate percentage values for each category
# percentage_table = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100

# # Plotting the stacked bar graph
# sns.set(context="paper", style="white", palette="deep", font_scale=2.0, color_codes=True, rc=({"font.family": "Dejavu Sans"}))
# plt.figure(figsize=(10, 8))

# # Plotting the stacked bar graph
# ax = percentage_table.plot(kind='bar', stacked=True)

# # Set the x-axis labels
# ax.set_xticklabels(percentage_table.index, rotation=0)

# plt.xlabel(' ')
# plt.title(print('chi square', np.round(chi2, 2)))
# plt.ylabel('Percentage', weight='bold')

# # Modify the legend labels to include percentages for each species
# legend_labels = []
# for col in percentage_table.columns:
#     species_percentages = ", ".join([f"{idx} {val:.1f}%" for idx, val in percentage_table[col].items()])
#     legend_labels.append(f"{col}: {species_percentages}")

# # Set the modified legend labels
# ax.legend(labels=legend_labels, title='Host', bbox_to_anchor=(1.10, 1.05), loc='upper left')

# # plt.show()
# plt.savefig('C:\Mannu\Projects\Mannu Phd\Final analysis\Results\chi_square_3_host.png', dpi = 500, bbox_inches="tight")



# # %%


# %%

# load important wavenumbers stored in the disk

with open(
    "C:\Mannu\Projects\Mannu Phd\Transfer_learning MLP\_transfer_learning\predictions.txt"
    ) as json_file:
    predictions = json.load(json_file)

predictions_df = pd.DataFrame(predictions)
y_test = np.asarray(predictions_df['y_test'])
y_pred = np.asarray(predictions_df['predictions'])

classes = ['Bovine', 'Human']

confusion_matrix = confusion_matrix(y_test, y_pred)

# %%

# Create a DataFrame from the confusion matrix for easy plotting with Seaborn
df_cm = pd.DataFrame(confusion_matrix, index=classes, columns=classes)

# Create a heatmap using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot = True, fmt = 'd', cmap = 'Blues', xticklabels = classes, yticklabels = classes)

# Add labels, title, and axis ticks
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title('Confusion Matrix', fontsize=14)

# Show the plot
plt.show()

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predictions')
plt.axhline(y=0.5, color='red', linestyle='--', label='Threshold')
plt.xlabel('True Labels')
plt.ylabel('Predicted Probabilities')
plt.title('Scatter Plot of Classification Predictions')
plt.legend()
plt.show()

# %%
