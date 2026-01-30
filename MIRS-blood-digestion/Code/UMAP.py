#%%
import os
import io
import json
import joblib
import umap
from time import time

from itertools import cycle
import random as rn

import numpy as np 
import pandas as pd

from random import randint
from collections import Counter 

from sklearn import manifold

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
    silhouette_score
)

from sklearn.metrics import accuracy_score

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
# Making prediction on the blood meal hours data
# Load hours blood meal data

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

blood_hours_df = blood_hours_df.drop(
    [
        'Cat1', 
        'Cat2',
        'StoTime'
    ], 
    axis = 1
)

# Check counts
print('Size of blood hours by count', Counter(blood_hours_df['Cat4']))
print('Size of blood hours by blood meal type', Counter(blood_hours_df['Cat3']))

# view the data
blood_hours_df.head()

#%%
# Separate features and labels
X = blood_hours_df.drop(['Cat3', 'Cat4'], axis = 1)
y_host = blood_hours_df['Cat3']
y_hour = blood_hours_df['Cat4']

# Scale x features
X_scaled = StandardScaler().fit_transform(np.asarray(X))

#%%
# Fit UMPA once on all samples so emmbedings are comparable across hours

num_neighbors = 32
min_dist = 0.7
num_metrics = 'chebyshev'
dimensions = 2

umap_model = umap.UMAP(
    n_neighbors = num_neighbors,
    min_dist=min_dist,
    n_components = dimensions,
    metric = num_metrics,
    random_state = 42
)

X_umap = umap_model.fit_transform(X_scaled)


#%%

hours = ['6H', '12H', '24H', '48H']
titles = ['6 Hours', '12 Hours', '24 Hours', '48 Hours']
colors = {'Bovine': '#1f77b4', 'Human': '#ff7f0e'}

fig, axes = plt.subplots(2, 2, 
                         figsize=(10, 8), 
                         sharex=True, 
                         sharey=True)

axes = axes.flatten()

for ax, hour, title in zip(axes, hours, titles):

    idx = y_hour == hour

    for host in np.unique(y_host):
        m_idx = idx & (y_host == host)
        ax.scatter(
            X_umap[m_idx, 0],
            X_umap[m_idx, 1],
            label = host,
            alpha = 0.7,
            # s = 18
        )

    ax.set_title(title)
    ax.set_xlabel('UMAP-1')
    ax.set_ylabel('UMAP-2')

# Single legend for the whole figure
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %%

# Fine tune UMAP

start_time = time()

neighbors_grid = [5, 10, 15, 20, 30]
min_dist_grid = [0.0, 0.05, 0.1, 0.2, 0.5]
metrics_grid = ['euclidean', 'cosine', 'correlation']

# Grid search over UMAP parameters

results = []

for metric in metrics_grid:
    for n in neighbors_grid:
        for d in min_dist_grid:
            
            umap_model = umap.UMAP(
                n_neighbors=n,
                min_dist=d,
                n_components=2,
                metric=metric,
                random_state=42
            )
            
            embedding = umap_model.fit_transform(X_scaled)

            score = silhouette_score(
                embedding,
                y_host   # Bovine vs Human
            )

            results.append({
                'metric': metric,
                'n_neighbors': n,
                'min_dist': d,
                'silhouette': score
            })

end_time = time()
print('Run time : {} s'.format(end_time-start_time))
print('Run time : {} m'.format((end_time-start_time)/60))
print('Run time : {} h'.format((end_time-start_time)/3600))

#%%
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by = 'silhouette', 
    ascending = False
)

results_df.head(10)

# %%
# Re-fit UMPA once on all samples so emmbedings are comparable across hours

num_neighbors = 15
min_dist = 0.50
num_metrics = 'correlation'
dimensions = 2

umap_model = umap.UMAP(
    n_neighbors = num_neighbors,
    min_dist=min_dist,
    n_components = dimensions,
    metric = num_metrics,
    random_state = 42
)

X_umap = umap_model.fit_transform(X_scaled)


#%%
# Plotting UMAP embeddings

sns.set(
    context = "paper",
    style = "white",
    palette = "deep",
    font_scale = 1.5,
    color_codes = True,
    rc = ({"font.family": "Dejavu Sans"})
)

fig, axes = plt.subplots(2, 2, 
                         figsize=(8, 7), 
                         sharex=True, 
                         sharey=True)

axes = axes.flatten()

for ax, hour, title in zip(axes, hours, titles):

    idx = y_hour == hour

    for host in np.unique(y_host):
        m_idx = idx & (y_host == host)
        ax.scatter(
            X_umap[m_idx, 0],
            X_umap[m_idx, 1],
            label = host,
            alpha = 0.7,
            s = 18
        )

    ax.set_title(title)
    ax.set_xlabel('UMAP-1')
    ax.set_ylabel('UMAP-2')

# Single legend for the whole figure
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig(
    os.path.join("..",
                 "Results", 
                 "UMAP.png"
                ), 
    dpi = 500, 
    bbox_inches = "tight"
)

# plt.show()
# %%
