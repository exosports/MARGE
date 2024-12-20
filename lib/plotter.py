"""
Module that contains functions related to plotting.

resume_save: Handles plot filenames when there are existing plots.

loss: Plots the loss and learning rate history.

pred_vs_true: Plots scatterplot of predicted vs. true values.

plot_spec: Plots the predicted spectrum, vs. the true spectrum if supplied.

plot: Simple plot of y vs. x values.

"""

import sys, os, platform
import numpy as np
import matplotlib as mpl
if platform.system() == 'Darwin':
    # Mac fix: use a different backend
    mpl.use("TkAgg")
import matplotlib.pyplot as plt
import scipy.signal      as ss
from sklearn import metrics

import stats as S

# to allow plots with many points
mpl.rcParams['agg.path.chunksize'] = 10000


def resume_save(fname):
    """
    Handles setting file name in case training is resumed.

    Inputs
    ------
    fname: Base save name

    Outputs
    -------
    Plot file of `fname` or an incremented version of it
        Example: fname = 'path/to/someplot.png'
                 1st call to resume_save(): saves path/to/someplot.png
                 2nd call to resume_save(): saves path/to/someplot_res1.png
                 3rd call to resume_save(): saves path/to/someplot_res2.png

    """
    if not os.path.exists(fname):
        plt.savefig(fname, bbox_inches='tight')
    else:
        res    = 1
        fsplit = fname.rsplit('.', 1)
        resfoo = fsplit[0] + '_res1'
        if len(fsplit) == 2:
            # Add file extension
            resfoo = resfoo + '.' + fsplit[1]
        while os.path.exists(resfoo):
            resfoo = resfoo.replace('_res'+str(res), '_res'+str(res+1))
            res   += 1
        plt.savefig(resfoo, bbox_inches='tight')


def loss(nn, plotdir, fname='history_train_val_loss.png', resume=False):
    """
    Plots the loss.

    Inputs
    ------
    n    : int.    Number model in the ensemble.
    nn   : object. NN to plot the loss for.
    fname: string. Path/to/file for the plot to be saved.
                   Extension must be .png or .pdf.

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
    # Training loss is at 1/2 the epoch,
    # while val loss is at the end of the epoch
    plt.plot(np.arange(len(tr_loss ))+0.5, tr_loss,  label='train')
    plt.plot(np.arange(len(val_loss))+1.0, val_loss, label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    resume_save(plotdir+fname)
    plt.ylim(min_loss, min_loss+np.abs(min_loss*0.5))
    resume_save(plotdir+fname.replace('loss', 'loss_zoom'))
    plt.close()

    if nn.historyCLR is not None:
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
        pname = plotdir+fname.replace('train_val_loss', 'clr_loss')
        if resume:
            resume_save(pname)
        else:
            plt.savefig(pname, bbox_inches='tight')
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
              xvals=None, xlabel=None, ylabel=None, smoothing=0):
    """
    Plots the predicted spectrum, vs the known spectrum if supplied.

    Inputs
    ------
    fname   : string. Path/to/plot to be saved.
    predspec: array.  NN output prediction.
    truespec: array.  (optional) True output.
    xvals   : array.  (optional) X values associated with NN prediction.
    xlabel  : string. (optional) X-axis label.
    ylabel  : string. (optional) Y-axis label.

    Outputs
    -------
    Plot of the supplied data, saved as `fname`.

    """
    # Set defaults
    if xvals is None:
        xvals = np.arange(len(predspec))
    if xlabel is None:
        xlabel  = 'Parameter #'
    if ylabel is None:
        ylabel  = 'Predicted Value'
    if truespec is None:
        trans = 1.0
    else:
        trans = 0.5

    fig1   = plt.figure(1)
    frame1 = fig1.add_axes((.1, .3, .8, .6))
    plt.plot(xvals, predspec, label='Predicted', lw=0.5, alpha=trans, c='b')
    plt.ylabel(u''+ylabel, fontsize=12)
    if truespec is not None:
        plt.plot(xvals, truespec, label='True', lw=0.5, alpha=trans, c='r')
        if smoothing:
            predsmooth = ss.savgol_filter(predspec, smoothing, 3)
            plt.plot(xvals, predsmooth, c='navy',
                     label='Predicted, smoothed', ls='--', lw=0.5, alpha=trans)
            truesmooth = ss.savgol_filter(truespec, smoothing, 3)
            plt.plot(xvals, truesmooth, c='maroon',
                     label='True, smoothed', ls='--', lw=0.5, alpha=trans)
        lgd = plt.legend(loc='best', prop={'size': 9})
        for lgdobj in lgd.legendHandles:
            lgdobj.set_linewidth(2.0)
        frame1.set_xticklabels([])
        frame2 = fig1.add_axes((.1, .1, .8, .2))
        resid       = 100 * (predspec   - truespec  ) / truespec
        plt.scatter(xvals, resid, s=0.4,
                    alpha=0.7, label='Full resolution', c='b')
        if smoothing:
            residsmooth = 100 * (predsmooth - truesmooth) / truesmooth
            plt.scatter(xvals, residsmooth, s=0.4,
                        alpha=0.7, label='Smoothed', c='r')
            lgd = plt.legend(loc='best', prop={'size': 8})
            for lgdobj in lgd.legendHandles:
                lgdobj.set_sizes([16])
        plt.hlines(0, np.amin(xvals), np.amax(xvals))
        yticks = frame2.yaxis.get_major_ticks()
        yticks[-1].label1.set_visible(False)
        plt.ylabel('Residuals (%)', fontsize=12)
        ylims = plt.ylim()

    plt.xlabel(u''+xlabel, fontsize=12)

    if truespec is not None:
        frame3 = fig1.add_axes((0.9, .1, .1, .2))
        plt.hist(resid[np.abs(resid)!=np.inf], bins=60, density=True,
                 orientation="horizontal")
        plt.xlabel('PDF', fontsize=12)
        plt.ylim(*ylims)
        plt.yticks(visible=False)
        plt.setp(frame3.get_xticklabels()[0], visible=False)

    plt.savefig(fname, bbox_inches='tight', dpi=600)
    plt.close()


def plot(fname, xvals, yvals, xlabel, ylabel):
    """
    Plots y vs. x values.

    Inputs
    ------
    fname : string. Path/to/plot to save.
    xvals : array.  X-axis values.  If None, uses simple range.
    yvals : array.  Y-axis values.
    xlabel: string. X-axis label.
    ylabel: string. Y-axis label.

    Outputs
    -------
    Plot of the supplied data, saved as `fname`.
    """
    if xvals is None:
        xvals = np.arange(len(yvals), dtype=int)
    plt.plot(xvals, yvals)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(fname, bbox_inches='tight', dpi=600)
    plt.close()
