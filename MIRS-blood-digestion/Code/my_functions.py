import os
import itertools
from time import time

import numpy as np 
import pandas as pd

from sklearn.model_selection import (
    RandomizedSearchCV, 
    cross_val_score
    )
from sklearn.metrics import confusion_matrix

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
                            figure_name,
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
    tick_marks = np.arange(len(classes_names))
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

    # plt.close() # close the plot to prevent memory overload
        

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
    plot_confusion_matrix(cnf_matrix, figure_name)


#%%
# Evaluate models to get the best performing model

def evaluate_model(models, X, y, cv, scoring):

    """Evaluate a model using cross-validation."""

    results = []
    names = []

    for name, model in models:
        cv_results = cross_val_score(
            model,
            X,
            y,
            cv = cv,
            scoring = scoring
        )
        results.append(cv_results)
        names.append(name)
        msg = f'Cross validation score for {name}: {cv_results.mean():.2%} ± {cv_results.std():.2%}'
        print(msg)

    return results, names


#%%

# Plotting function for algorithm comparison

def plot_algorithm_comparison(results, names):
    """
    Plots a boxplot comparing algorithm performance based on accuracy.

    Parameters:
    - results: List of lists or 2D array-like, where each sublist contains accuracy scores for a model.
    - names: List of model names corresponding to the columns in `results`.

    Returns:
    - fig: The matplotlib Figure object containing the plot.
    """
    # Transform the results into a pandas DataFrame
    results_df = pd.DataFrame(results).T  # Transpose to match the expected format
    results_df.columns = names

    # Melt the DataFrame into a long format
    results_df = pd.melt(results_df, var_name='Model', value_name='Accuracy')

    # Create the plot
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.boxplot(
        x='Model',
        y='Accuracy',
        data=results_df,
        hue='Model',
        palette='colorblind',
        ax=ax
    )
    sns.despine(offset=5, trim=False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_yticks(np.arange(0.0, 1.1, step=0.2))
    ax.set_ylabel('Accuracy', weight='bold')
    ax.set_xlabel("")
    
    # Check if legend exists before removing it
    if ax.legend_:
        ax.legend_.remove()
    
    fig.tight_layout()
    fig.tight_layout()

    return fig