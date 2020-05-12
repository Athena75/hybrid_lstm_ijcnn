import sys
import os

SRC_PATH = '../src'
VISUALIZATIONS_PATH = '../results/visualizations'

sys.path.insert(0, os.path.join(SRC_PATH, 'Modeling'))

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
from sklearn.manifold import TSNE


def _tsne(x):
    # reduction de dimension en 2D
    print("computing components ")
    print("for more informations : https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html")
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(x)
    return X_tsne


def _get_fig_filename(title):
    """get the asolute path of the figure to save
    """
    now = datetime.now().strftime("%Y-%m-%d:%H-%M-%S")
    return os.path.join(VISUALIZATIONS_PATH, title + ' ' + now + '.png')


def tsne_plot(x, y, save=False, title=''):
    """ Get 2D visualization 
    
    Parameters
    ----------
    x : samples(without labels)
    y : label column
    save : boolean to save or not the figure
        if True the plot will be saved in png format in VISUALIZATIONS_PATH
    title : string that will be set as a title on the resulted plot 
    
    """
    X_tsne = _tsne(x)
    # set fig dimensions
    plt.figure(figsize=(12, 8))
    plt.title(title)
    # plot normal data
    # x_axis = 1st feature of X_tsne
    # y_axis = 2nd feature of X_tsne

    # plot normal data points
    plt.scatter(X_tsne[np.where(y == 0), 0], X_tsne[np.where(y == 0), 1], marker='o', color='g', linewidth='1',
                alpha=0.8, label='normal')
    # add abnormal data point
    plt.scatter(X_tsne[np.where(y == 1), 0], X_tsne[np.where(y == 1), 1], marker='*', color='r', linewidth='1',
                alpha=0.8, label='abnormal')
    # add legend
    plt.legend(loc=0)
    # save_fig
    if save:
        plot_path = _get_fig_filename(title)
        print("saving tsne {} plot ;) \n****** ==> see : {}".format(title, plot_path))
        plt.savefig(plot_path)
    else:
        plt.show()
