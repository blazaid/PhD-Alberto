import random

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import fuzzle.mfs

class Dataset:
    def __init__(self, path=None, values=None, outputs_num=None):
        values = values or []
        if path:
            with open(path, 'r') as f:
                for line in f.readlines():
                    values.append([float(v) for v in line.split(',') if v])
        self.values = np.array(values)
        self.i = None
        self.o = None
        self.outputs_num = outputs_num or 1        
    
    def split(self, ratio, shuffle=False):
        values = self.values.copy()
        if shuffle:
            np.random.shuffle(values)
        idx = int(len(values) * ratio)
        return (
            self.__class__(
                values=values.tolist()[:idx],
                outputs_num=self.outputs_num
            ),
            self.__class__(
                values=values.tolist()[idx:],
                outputs_num=self.outputs_num
            ),
        )
    
    @property
    def inputs(self):
        if self.i is None:
            self.i = self.values[:,:-self.outputs_num].astype(np.float32)
        return self.i

    @property
    def outputs(self):
        if self.o is None:
            self.o = np.reshape(
                self.values[:,-self.outputs_num:],
                (-1, self.outputs_num)
            ).astype(np.float32)
        return self.o


def plot_mfs(ax, X, Y, stage='first', **kwargs):
    """
    
    :param stage: 'first', 'train' or 'final'.
    """
    stage_styles = {
        'first': {'linewidth': 3, 'alpha': 0.4},
        'train': {'linewidth': 0.2, 'c': '0.6'},
        'final': {'linewidth': 2},
    }
    for i, column in enumerate(Y.T):
        # Select the style of the line according to the parameter stage
        mf_params = stage_styles[stage].copy()
        # Select the color index from the style cmap
        if 'c' not in mf_params:
            mf_params['c'] = 'C{}'.format(i % 10)
        # Get the label for this membership function (if any). If not, set the
        # default value.
        if 'labels' in kwargs and len(kwargs['labels']) > i:
            mf_params['label'] = kwargs['labels'][i]
        # Plot the membership function
        if 'singletons' in kwargs and kwargs['singletons'][i] is not None:
            mf_params['colors'] = mf_params.pop('c')
            ax.vlines(kwargs['singletons'][i], 0, 1, **mf_params);
        else:
            ax.plot(X[:,0], column, **mf_params);
    ax.plot(X, [0 for _ in X], linewidth=2, c='0.0');

    
def plot_lvar(lvar, res=1000):
    mf_names = [fs for fs in lvar]
    singletons = [
        lvar[fs].a if isinstance(lvar[fs], fuzzle.mfs.SingletonMF) else None
        for fs in mf_names
    ]

    X = np.linspace(*lvar.domain, res)
    X = np.concatenate(([-X[1]], X, [X[-1] + X[1] - X[0]]))
    Y = []
    
    for x in X:
        Y.append([lvar[mf_name](x) for mf_name in mf_names])
    X = np.reshape(X, (-1, 1))
    Y = np.array(Y)

    fig, ax = plt.subplots(1, 1)
    fig.suptitle(lvar.name)
    plot_mfs(ax, X, Y, stage='final', labels=mf_names, singletons=singletons)
    
    
original_plot_params = {'linewidth': 10, 'alpha': 0.4, 'c': 'b'}
original_train_params = {'linewidth': 0.2, 'alpha': 0.9, 'c': 'r'}
first_result_params = {'linewidth': 10, 'alpha': 0.4, 'c': 'r'}
final_result_params = {'linewidth': 2, 'c': 'b'}