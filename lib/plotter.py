"""
Module that contains functions related to plotting.

loss: Plots the loss and learning rate history.

pred_vs_true: Plots scatterplot of predicted vs true values.

plot_spec: Plots the predicted spectrum, vs the true spectrum if supplied.

"""

# -*- coding: utf-8 -*-

import sys, os

#import keras
#from keras import backend as K
#from keras.models import Sequential, Model
#from keras.layers import Convolution1D, Dense, MaxPooling1D, Flatten, Input, Lambda, Wrapper, merge, concatenate
#from keras.engine import InputSpec
#from keras.layers.core import Dense, Dropout, Activation, Layer, Lambda, Flatten
#from keras.regularizers import l2
#from keras.optimizers import RMSprop, Adadelta, adam
#from keras.layers.advanced_activations import LeakyReLU
#from keras import initializers
#import tensorflow as tf

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn import metrics

import stats as S

# to allow plots with many points
mpl.rcParams['agg.path.chunksize'] = 10000


def loss(nn, plotdir, fname='history_train_val_loss'):
    """
    Plots the loss.

    Inputs
    ------
    n    : int.    Number model in the ensemble.
    nn   : object. NN to plot the loss for.
    fname: string. Path/to/file for the plot to be saved.
                   Extension must be .png or .pdf.
                   No extension defaults to .png.

    Outputs
    -------
    `fname`: image file w/ plot of the loss.
    """
    tr_loss  = nn.historyNN.history['loss']
    val_loss = nn.historyNN.history['val_loss']
    min_loss = min(np.nanmin( tr_loss), 
                   np.nanmin(val_loss))

    plt.figure()
    plt.title('Model Loss History')
    plt.plot(tr_loss,  label='train')
    plt.plot(val_loss, label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(plotdir+fname, bbox_inches='tight')
    plt.ylim(min_loss, min_loss+np.abs(min_loss*0.5))
    plt.savefig(plotdir+fname+'_zoom', bbox_inches='tight')
    plt.close()

    clr_lr   = nn.historyCLR['lr']
    clr_loss = nn.historyCLR['loss']
    plt.figure()
    plt.title('CLR History')
    if np.min(clr_loss) > 0:
        plt.loglog(clr_lr, clr_loss)
    else:
        plt.semilogx(clr_lr, clr_loss)
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")
    plt.savefig(plotdir+fname.replace('train_val_loss', 'clr_loss'), 
                bbox_inches='tight')
    plt.ylim(np.nanmin(val_loss), 
             np.nanmin(val_loss)+np.abs(np.nanmin(val_loss)*0.5))
    plt.savefig(plotdir+fname.replace('train_val_loss', 'clr_loss_zoom'), 
                bbox_inches='tight')
    plt.close()


def pred_vs_true(fpred_mean, fy_test_un, 
                 names, plotdir):
    """
    Plots mean predictions for all cases vs. the true values.

    Inputs
    ------
    pred_mean: array. Mean denormalized test predictions.
    y_test_un: array. Raw test data before normalization.
    names    : list.  Parameter names, for plotting.
    plotdir  : str.   Directory to save plots.

    Outputs
    -------
    Plot for each parameter showing the mean predicted values vs. true values.
    """
    # Load the data
    y_test_un = np.load(fy_test_un)
    pred_mean = np.load(fpred_mean)
    R2  = []

    # Plot all predictions vs true values for all parameters
    for p in range(y_test_un.shape[-1]):
        minx = np.min(y_test_un[:, p])
        maxx = np.max(y_test_un[:, p])
        miny = np.min(pred_mean[:, p])
        maxy = np.max(pred_mean[:, p])
        yyy  = [min(minx, miny), max(maxx, maxy)]
        plt.plot(yyy,yyy,'r')
        plt.title(names[p])
        plt.ylabel('Predictions')
        plt.xlabel('True')
        #plt.xlim([minx,maxx])
        #plt.ylim([miny,maxy])
        plt.xlim(yyy)
        plt.ylim(yyy)
        plt.scatter(y_test_un[:, p],
                    pred_mean[:, p],
                    s=10, alpha=0.6)
        # MSE: 
        MSE = np.mean((y_test_un[:, p] -
                       pred_mean[:, p])**2)
        sig = np.std ((y_test_un[:, p] -
                       pred_mean[:, p])**2)
        plt.title(names[p] + '\nMSE: %.3f,  2 Standard Deviations %.3f' %
                  (MSE, sig))
        R2.append(metrics.r2_score(y_test_un[:, p],
                                   pred_mean[:, p]))
        plt.savefig(plotdir + \
                    names[p].replace('mathrm', '').replace('\\', '').
                             replace(' ', '_').replace('$', '').
                             replace('_',  '').replace('^', '').
                             replace('{',  '').replace('}', '') + \
                    '_test_predictions', bbox_inches='tight')
        plt.close()

    return np.asarray(R2)


def plot_spec(fname, predspec, truespec=None, 
              xvals=None, xlabel=None, ylabel=None, olog=False):
    """
    Plots the predicted spectrum, vs the known spectrum if supplied.
    """
    # Set defaults
    if xvals is None:
        xvals = np.arange(len(predspec))
    if xlabel is None:
        xlabel  = 'Parameter #'
    if ylabel is None:
        ylabel  = 'Predicted Value'
    if olog:
        predspec = 10**predspec
        truespec = 10**truespec

    fig1   = plt.figure(1)
    frame1 = fig1.add_axes((.1, .3, .8, .6))
    plt.plot(xvals, predspec, label='Predicted', lw=0.5)
    plt.ylabel(u''+ylabel, fontsize=12)
    #frame1.yaxis.set_label_coords(-0.2, 0.5)
    if truespec is not None:
        plt.plot(xvals, truespec, label='True', ls='--', lw=0.5)
        plt.legend(loc='best')
        frame1.set_xticklabels([])
        frame2 = fig1.add_axes((.1, .1, .8, .2))
        plt.plot(xvals, 100 * (predspec - truespec) / truespec, lw=0.5)
        yticks = frame2.yaxis.get_major_ticks()
        yticks[-1].label1.set_visible(False)
        plt.ylabel('Residuals (%)', fontsize=12)
        #frame2.yaxis.set_label_coords(-0.2, 0.5)
    plt.xlabel(u''+xlabel, fontsize=12)
    plt.savefig(fname, bbox_inches='tight', dpi=600)
    plt.close()

