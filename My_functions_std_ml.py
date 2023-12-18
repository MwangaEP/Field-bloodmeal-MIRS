# Functions for ML models to predict blood_meal sources

import os
import joblib
import itertools

import numpy as np 

from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

from sklearn import decomposition
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

import matplotlib.pyplot as plt # for making plots
import seaborn as sns
sns.set(context = "paper",
        style = "whitegrid",
        palette = "deep",
        font_scale = 2.0,
        color_codes = True,
        rc = None)

# %matplotlib inline
plt.rcParams["figure.figsize"] = [6,4]

#%%

# This normalizes the confusion matrix and ensures neat plotting for all outputs.
# Function for plotting confusion matrcies

def plot_confusion_matrix(cm, classes, file_dr,
                          normalize = True,
                          title = 'Confusion matrix',
                          xrotation = 0,
                          yrotation = 0,
                          cmap = plt.cm.Blues,
                          printout = False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
        if printout:
            print("Normalized confusion matrix")
    else:
        if printout:
            print('Confusion matrix')

    if printout:
        print(cm)
    
    plt.figure(figsize=(6,4))

    plt.imshow(cm, interpolation = 'nearest', vmin = .2, vmax= 1.0,  cmap=cmap)
    plt.title(' ')
    plt.colorbar()
    classes = classes
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = xrotation)
    plt.yticks(tick_marks, classes, rotation = yrotation)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', weight = 'bold')
    plt.xlabel('Predicted label', weight = 'bold')
    plt.savefig(file_dr, dpi = 500, bbox_inches = "tight")

#%%

def visualize(classes, true, predicted, file_dr):
    # Sort out predictions and true labels
    # for label_predictions_arr, label_true_arr, classes, outputs in zip(predicted, true, classes, outputs):
    # print('visualize predicted classes', predicted)
    # print('visualize true classes', true)
    classes_pred = np.asarray(predicted)
    classes_true = np.asarray(true)
    print(classes_pred.shape)
    print(classes_true.shape)
    # classes = ['1-9', '10-17']
    cnf_matrix = confusion_matrix(classes_true, classes_pred, labels = classes)
    plot_confusion_matrix(cnf_matrix, classes, file_dr)


#%%

def plot_cumulative_variance(eigenvalues, n_components = 0, figure_size = None, title = None, save_filename = None):
    """
    Plots the eigenvalues as bars and their cumulative sum to visualize
    the percent variance in the data explained by each principal component
    individually and by each principal component cumulatively.

    **Example:**

    .. code:: python

        from PCAfold import PCA, plot_cumulative_variance
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,5)

        # Perform PCA and obtain eigenvalues:
        pca_X = PCA(X)
        eigenvalues = pca_X.L

        # Plot the cumulative variance from eigenvalues:
        plt = plot_cumulative_variance(eigenvalues,
                                       n_components=0,
                                       title='PCA on X',
                                       save_filename='PCA-X.pdf')
        plt.close()

    :param eigenvalues:
        a 0D vector of eigenvalues to analyze. It can be supplied as an attribute of the
        ``PCA`` class: ``PCA.L``.
    :param n_components: (optional)
        how many principal components you want to visualize (default is all).
    :param figure_size: (optional)
        tuple specifying figure size.
    :param title: (optional)
        ``str`` specifying plot title. If set to ``None`` title will not be
        plotted.
    :param save_filename: (optional)
        ``str`` specifying plot save location/filename. If set to ``None``
        plot will not be saved. You can also set a desired file extension,
        for instance ``.pdf``. If the file extension is not specified, the default
        is ``.png``.

    :return:
        - **plt** - ``matplotlib.pyplot`` plot handle.
    """

    if title is not None:
        if not isinstance(title, str):
            raise ValueError("Parameter `title` has to be of type `str`.")

    if save_filename is not None:
        if not isinstance(save_filename, str):
            raise ValueError("Parameter `save_filename` has to be of type `str`.")

    bar_color = '#191b27'
    line_color = '#ff2f18'

    (n_eigenvalues, ) = np.shape(eigenvalues)

    if n_components == 0:
        n_retained = n_eigenvalues
    else:
        n_retained = n_components

    x_range = np.arange(1, n_retained+1)

    if figure_size is None:
        fig, ax1 = plt.subplots(figsize = (n_retained, 4))
    else:
        fig, ax1 = plt.subplots(figsize = figure_size)

    ax1.bar(x_range, eigenvalues[0:n_retained], color = bar_color, edgecolor = bar_color, align = 'center', zorder = 2, label = 'Eigenvalue')
    ax1.set_ylabel('Eigenvalue', weight = 'bold')
    ax1.set_ylim(0, 1.05)
    # ax1.grid(zorder=0)
    ax1.set_xlabel('Principal component', weight = 'bold')

    ax2 = ax1.twinx()
    ax2.plot(x_range, np.cumsum(eigenvalues[0:n_retained])*100, 'o-', color = line_color, zorder = 2, label = 'Cumulative')
    ax2.set_ylabel('Variance explained [%]', color = line_color, weight = 'bold')
    ax2.set_ylim(0, 105)
    ax2.tick_params('y', colors = line_color)

    plt.xlim(0, n_retained + 1)
    plt.xticks(x_range)

    if title != None:
        plt.title(title)

    if save_filename != None:
        plt.savefig(save_filename, dpi = 500, bbox_inches = 'tight')

    return plt

#%%

# Define function to select number of components based accuracy 

def get_models(n_components_range):

	models = dict()

	for i in n_components_range:

		steps = [('pca', decomposition.PCA(n_components = i)), 

                 ('m', XGBClassifier(gamma = 0.1, learning_rate = 0.1,max_depth = 8,
                                     min_child_weight = 3, n_estimators = 1000, 
                                     colsample_bytree=0.5))]

		models[str(i)] = Pipeline(steps = steps)

	return models

# evaluate a given model using cross-validation

def evaluate_model(model, X, y, random_seed, num_folds):

	cv = KFold(n_splits = num_folds, shuffle = True, random_state = random_seed)
	scores = cross_val_score(model, X, y, scoring = 'accuracy', cv = cv)

	return scores

#%%
