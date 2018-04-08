import itertools

import numpy as np

from matplotlib import cm

def plot_confusion_matrix(ax, matrix, classes, cmap=None, title=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    ax.imshow(matrix, interpolation='nearest', cmap=cmap or cm.Purples)
    if title:
        ax.title(title)
    ax.grid(False)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    thresh = matrix.max() / 2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        ax.text(j, i, '{:.2f} %'.format(matrix[i, j]), horizontalalignment="center", color="white" if matrix[i, j] > thresh else "black")

    ax.set_ylabel('Real')
    ax.set_xlabel('Predicted')