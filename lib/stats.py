"""
Module that contains functions related to statstics.

get_stats: Uses Welford's method to calculate the mean and standard deviation
           of the entire dataset, without loading all data in memory at once.

save_stats_files: Calls get_stats(), saves the results, and checks them for
                  issues.

rmse: Calculates the RMSE for some predictions vs true values.

r2: Calcualtes the coefficient of determination (R^2) for some predictions
    vs true values.

"""

import sys, os
import glob
import numpy as np
import scipy.interpolate as si

import tensorflow.keras.backend as K

import utils as U


def get_stats(datafiles, ilog, olog, ishape, oshape, statsaxes, perc=10, num_per_file=None, verb=False):
    """
    Uses Welford's method to calculate the mean and standard deviation (via the
    variance) of the entire dataset, without loading all data in memory at once.

    Inputs
    ------
    datafiles: list, strings. [path/to/datafile1.npz, path/to/datafile2.npz, ... ]
                              Must contain two Numpy arrays in the NPZ file,
                              'x' and 'y', corresponding to the inputs and
                              outputs, respectively.
    ilog: bool. Determines whether to take the log10 of the input data/features.
    olog: bool. Determines whether to take the log10 of the output/target data.
    ishape: tuple. Shape of a single case from the input  data.
    oshape: tuple. Shape of a single case from the output data.
    statsaxes: string. Determines which axes to compute stats over.
                       Options: all - all axes except last axis
                              batch - only 0th axis
    perc: int.  Determines the frequency of output of updates.
                Updates will be printed every `perc` %, if verb>0.
    num_per_file: int. Number of cases per file.
                       Not used by MARGE, but available if a user wants to apply
                       this to some dataset where data files are expected to
                       contain `num_per_file` cases per data file.
                       Prints warning if a file does not match the expected
                       value (requires verb > 0)
    verb: bool or int. Flag that determines whether to print additional outputs.

    Outputs
    -------
    xmean  : arr, float. Mean  values of the dataset inputs.
    xstdev : arr, float. Stdev values of the dataset inputs.
                         Computed as sqrt of sample variance.
    xmin   : arr, float. Minima of the dataset inputs.
    xmax   : arr, float. Maxima of the dataset inputs.
    ymean  : arr, float. Mean  values of the dataset outputs.
    ystdev : arr, float. Stdev values of the dataset outputs.
    ymin   : arr, float. Minima of the dataset outputs.
    ymax   : arr, float. Maxima of the dataset outputs.

    Notes
    -----
    https://www.johndcook.com/blog/standard_deviation/
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

    """
    if statsaxes == 'all':
        istatshape = ishape[-1]
        ostatshape = oshape[-1]
    elif stataxes == 'batch':
        istatshape = ishape
        ostatshape = oshape

    up    = 0  # Counter for checking percent done
    ncx   = 0                          # Number of cases seen thus far
    ncy   = 0
    xmean = np.zeros(istatshape)           # Running mean
    xM2   = np.zeros(istatshape)           # Running variance
    xmin  = np.ones (istatshape) *  np.inf # Min for each element
    xmax  = np.ones (istatshape) * -np.inf # Max for each element
    ymean = np.zeros(ostatshape)           # Running mean
    yM2   = np.zeros(ostatshape)           # Running variance
    ymin  = np.ones (ostatshape) *  np.inf # Min for each element
    ymax  = np.ones (ostatshape) * -np.inf # Max for each element

    for i in range(len(datafiles)):
        data = np.load(datafiles[i])
        x = data['x']
        if np.any(np.isnan(x)):
            raise ValueError("Data file " + datafiles[i] + " has nans in the inputs.")
        if np.any(np.isinf(x)):
            raise ValueError("Data file " + datafiles[i] + " has infs in the inputs.")
        y = data['y']
        if np.any(np.isnan(y)):
            raise ValueError("Data file " + datafiles[i] + " has nans in the outputs.")
        if np.any(np.isinf(y)):
            raise ValueError("Data file " + datafiles[i] + " has infs in the outputs.")
        # Ensure proper shape
        if x.shape[0] != y.shape[0]:
            # This file is a single input/output case,
            # but isn't shaped to reflect this
            raise ValueError("The inputs and outputs of this file do not "+\
                             "have the same number of cases:\n"           +\
                             datafiles[i])
        else:
            # This file has multiple data cases, ensure proper shape
            if x.shape[1:] != ishape:
                raise ValueError("This file has improperly shaped input data:\n"+\
                                 datafiles[i]+ "\nExpected shape: " +\
                                 str(ishape) + "\nReceived shape: " + str(x.shape))
            if y.shape[1:] != oshape:
                raise ValueError("This file has improperly shaped output data:\n"+\
                                 datafiles[i]+ "\nExpected shape: " +\
                                 str(oshape) + "\nReceived shape: " + str(y.shape))
        # Take logs
        np.seterr(all='raise')
        if ilog:
          try:
            x[..., ilog] = np.log10(x[..., ilog])
          except:
            print("Problem foo:", datafiles[i])
            sys.exit()
        if olog:
          try:
            y[..., olog] = np.log10(y[..., olog])
          except:
            print("Problem foo:", datafiles[i])
            sys.exit()
        # Update min/max
        if statsaxes == 'all':
            # Reshape to 2D for the calculations
            x = x.reshape(-1, istatshape)
            y = y.reshape(-1, ostatshape)
        xmin = np.amin(np.vstack([xmin[None, :], x]), axis=0)
        xmax = np.amax(np.vstack([xmax[None, :], x]), axis=0)
        ymin = np.amin(np.vstack([ymin[None, :], y]), axis=0)
        ymax = np.amax(np.vstack([ymax[None, :], y]), axis=0)
        # Consider each data vector in this file
        for j in range(x.shape[0]):
            if np.all(x[j] == 0):
                if verb:
                    print("This file has an input data vector of only zeros:", datafiles[i])
                    print("Index:", j)
                    print("Ignoring this file")
                continue
            ncx    += 1
            xdelta  = x[j] - xmean
            xmean   = xmean + xdelta / ncx
            xM2     = xM2   + xdelta * (x[j] - xmean)
        for j in range(y.shape[0]):
            if np.all(y[j] == 0):
                if verb:
                    print("This file has an output data vector of only zeros:", datafiles[i])
                    print("Index:", j)
                    print("Ignoring this file")
                continue
            ncy    += 1
            ydelta  = y[j] - ymean
            ymean   = ymean + ydelta / ncy
            yM2     = yM2   + ydelta * (y[j] - ymean)
            #M2     = M2   + (nc-1) * delta**2 / nc    # alt way to calc this
        # Print status updates
        if verb and (100*i // len(datafiles)) >= (up+1)*perc:
            up += ((100*i // len(datafiles)) - up*perc) // perc
            print(len(str(up*perc))*'-' + "----------")
            print(str(up*perc)+"% complete")
            print("mean:", xmean, ymean)
            print("var :", xM2/(nc-1), yM2/(nc-1))
            print(len(str(up*perc))*'-' + "----------")

    print('-----------------------------------------')
    print('Completed mean/stdev/min/max calculations')
    print('-----------------------------------------')
    xvariance = xM2 / (ncx - 1)
    yvariance = yM2 / (ncy - 1)
    xstd = xvariance**0.5
    ystd = yvariance**0.5

    # Can't have 0 for stdev -- use 1 to not modify values
    xstd[xstd==0] = 1.
    ystd[ystd==0] = 1.

    return xmean, xstd, xmin, xmax, \
           ymean, ystd, ymin, ymax


def save_stats_files(foos, ilog, olog, ishape, oshape,
                     fxmean, fxstd, fxmin, fxmax,
                     fymean, fystd, fymin, fymax, statsaxes):
    """
    Saves out mean, standard deviation, minimum, and maximum files for the
    data set.

    Inputs
    ------
    foos: list, strings. Files to compute stats for.
    ilog: bool. Determines whether to take the log10 of the input data/features.
    olog: bool. Determines whether to take the log10 of the output/target data.
    ishape: tuple. Shape of a single case from the input  data.
    oshape: tuple. Shape of a single case from the output data.
    fxmean: string. Filepath to save the mean of the input data.
    fxstd : string. Filepath to save the standard deviation of the input data.
    fxmin : string. Filepath to save the minima of the input data.
    fxmax : string. Filepath to save the maxima of the input data.
    fymean: string. Filepath to save the mean of the output data.
    fystd : string. Filepath to save the standard deviation of the output data.
    fymin : string. Filepath to save the minima of the output data.
    fymax : string. Filepath to save the maxima of the output data.
    statsaxes: string. Determines which axes to compute stats over.
                       Options: all - all axes except last axis
                              batch - only 0th axis

    Outputs
    -------
    xmean  : arr, float. Mean  values of the dataset inputs.
    xstdev : arr, float. Stdev values of the dataset inputs.
                         Computed as sqrt of sample variance.
    xmin   : arr, float. Minima of the dataset inputs.
    xmax   : arr, float. Maxima of the dataset inputs.
    ymean  : arr, float. Mean  values of the dataset outputs.
    ystdev : arr, float. Stdev values of the dataset outputs.
    ymin   : arr, float. Minima of the dataset outputs.
    ymax   : arr, float. Maxima of the dataset outputs.
    """
    x_mean, x_std, x_min, x_max, \
    y_mean, y_std, y_min, y_max  = get_stats(foos, ilog, olog, ishape, oshape, statsaxes)
    np.save(fxmean, x_mean)
    np.save(fymean, y_mean)
    np.save(fxstd,  x_std)
    np.save(fystd,  y_std)
    np.save(fxmin,  x_min)
    np.save(fxmax,  x_max)
    np.save(fymin,  y_min)
    np.save(fymax,  y_max)

    # This shouldn't be possible, since nans are checked earlier
    if np.any(np.isnan(x_mean)):
        raise ValueError("The calculated input mean has nans.")
    if np.any(np.isnan(y_mean)):
        raise ValueError("The calculated output mean has nans.")
    if np.any(np.isnan(x_std)):
        raise ValueError("The calculated input stdev has nans.")
    if np.any(np.isnan(y_std)):
        raise ValueError("The calculated output stdev has nans.")

    # Make sure none of these statistics are inf
    if np.any(np.isinf(x_mean)):
        raise ValueError("The calculated input mean has infs.")
    if np.any(np.isinf(y_mean)):
        raise ValueError("The calculated output mean has infs.")
    if np.any(np.isinf(x_std)):
        raise ValueError("The calculated input stdev has infs.")
    if np.any(np.isinf(y_std)):
        raise ValueError("The calculated output stdev has infs.")
    # Make sure no zeros in the stdevs
    if np.any(x_std == 0):
        raise ValueError("The calculated input stdev has zeros.")
    if np.any(y_std == 0):
        raise ValueError("The calculated output stdev has zeros.")

    return x_mean, x_std, x_min, x_max, y_mean, y_std, y_min, y_max


def rmse_r2(fpred, ftrue, y_mean,
            y_std=None, y_min=None, y_max=None, scalelims=None,
            olog=False, y_mean_delog=None,
            x_vals=None, filters=None, filtconv=1.0):
    """
    Calculates the root mean squared error (RMSE) and R-squared for a data set.
    Predicted data must be saved in .NPY file(s), while the true data must be
    save in .NPZ file(s), with inputs under 'x' and outputs under 'y'.
    `fpred` and `ftrue` must exactly correspond in number of files and
    ordering of cases.

    Default behavior: compute RMSE/R2 for the raw predictions.

    If y_std, y_min, y_max, scalelims, and olog are specified,
    it will compute RMSE/R2 for both the raw and denormalized predictions.

    If filters and x_vals are specified, then the RMSE/R2 will be computed on
    the integrated filter bandpasses, rather than for each output parameter.

    Inputs
    ------
    fpred: list, strings. .NPY files with predicted values to compute RMSE.
    ftrue: list, strings. .NPZ files with true outputs to compute RMSE.
    y_mean: array. Training set mean value of each output parameter.
    y_std : array. Training set standard deviation of each output parameter.
    y_min : array. Training set minimum of each output parameter.
    y_max : array. Training set maximum of each output parameter.
    scalelims: tuple/list/array. Min/max that the normalized data was scaled to.
    olog  : bool.  Determines if the denormalized values are the log10 of the
                   true output parameters.
    y_mean_delog: array. Mean of the training set, not log-scaled.
    x_vals : array.         X values corresponding to the Y values.
    filters: list, strings. Filter bandpasses to integrate over.
                            Must be 2-column file: wavelength then transmission.
    filtconv: float.        Conversion factor for filter's wavelengths to
                            microns.

    Outputs
    -------
    rmse_norm  : array. RMSE for each parameter,   normalized data.
    rmse_denorm: array. RMSE for each parameter, denormalized data.
    r2_norm    : array. R2   for each parameter,   normalized data.
    r2_denorm  : array. R2   for each parameter, denormalized data.

    Notes
    -----
    Data will only be denormalized if y_std, y_min, y_max, scalelims, and olog
    are all not None.
    """
    if len(fpred) != len(ftrue):
        raise ValueError("There are different numbers of predicted and true files.\n"+\
                         "Ensure that each set of files follows the same "    +\
                         "exact structure and order.\nSee NNModel.Yeval() in NN.py.")

    # Integrate over filter bandpasses?
    if filters is not None and x_vals is not None:
        integ = True
        # Load filters
        nfilters = len(filters)
        filttran = []
        ifilt = np.zeros((nfilters, 2), dtype=int)
        meanwn = []
        for i in range(nfilters):
            datfilt = np.loadtxt(filters[i])
            # Convert filter wavelenths to microns, then convert um -> cm-1
            finterp = si.interp1d(10000. / (filtconv * datfilt[:,0]),
                                  datfilt[:,1],
                                  bounds_error=False, fill_value=0)
            # Interpolate and normalize
            tranfilt = finterp(x_vals)
            tranfilt = tranfilt / np.trapz(tranfilt, x_vals)
            meanwn.append(np.sum(x_vals*tranfilt)/sum(tranfilt))
            # Find non-zero indices for faster integration
            nonzero = np.where(tranfilt!=0)
            ifilt[i, 0] = max(nonzero[0][ 0] - 1, 0)
            ifilt[i, 1] = min(nonzero[0][-1] + 1, len(x_vals)-1)
            filttran.append(tranfilt[ifilt[i,0]:ifilt[i,1]]) # Store filter
    else:
        integ = False

    if not olog:
        y_mean_delog = y_mean
    else:
        if y_mean_delog is None:
            raise TypeError("If the outputs have been log-scaled, you must " + \
                            "supply the mean of the non-log-scaled training "+ \
                            "set outputs, `y_mean_delog`.")

    if all(v is not None for v in [y_std, y_min, y_max, scalelims]) or olog:
        denorm = True
    else:
        denorm = False

    # Variables for computing RMSE & R2
    n    = 0 # number of cases seen
    mss  = 0 # Model sum of squares
    tss  = 0 # True  sum of squares
    rss  = 0 # Residual sum of squares -- squared error
    if denorm:
        mss_denorm  = 0
        tss_denorm  = 0
        rss_denorm  = 0
    # By definition, R2 = 1 - rss / tss
    # If mss + rss = tss then R2 = mss / tss

    # Compute RMSE & R2
    for j in range(len(fpred)):
        # Load batch
        true  = np.load(ftrue[j])
        pred  = np.load(fpred[j])[..., :true.shape[-1]]
        # Add contributions to RMSE/R2
        if integ:
            pred_res, true_res, y_mean_res = integ_spec(pred, true, y_mean, x_vals, filttran, ifilt)
            contribs = squared_diffs(pred_res, true_res, y_mean_res)

        else:
            contribs = squared_diffs(pred, true, y_mean)
        n    += pred.shape[0]
        mss  += contribs[0]
        tss  += contribs[1]
        rss  += contribs[2]

        if denorm:
            # Calculate this for the denormalized values
            pred = U.denormalize(U.descale(pred,
                                           y_min, y_max, scalelims),
                                 y_mean, y_std)
            true = U.denormalize(U.descale(true,
                                           y_min, y_max, scalelims),
                                 y_mean, y_std)
            if olog:
                pred[...,olog] = 10**pred[...,olog]
                true[...,olog] = 10**true[...,olog]
            if integ:
                pred_res, true_res, y_mean_res = integ_spec(pred, true,
                                                            y_mean_delog,
                                                            x_vals,
                                                            filttran, ifilt)
                contribs = squared_diffs(pred_res, true_res, y_mean_res)
            else:
                contribs = squared_diffs(pred, true, y_mean_delog)
            mss_denorm  += contribs[0]
            tss_denorm  += contribs[1]
            rss_denorm  += contribs[2]
        print("  Batch "+str(j+1)+"/"+str(len(fpred)), end='\r')
    print('')
    rmse = (rss / n)**0.5
    r2   = 1 - rss / tss

    if denorm:
        rmse_denorm = (rss_denorm / n)**0.5
        r2_denorm   = 1 - rss_denorm / tss_denorm
    else:
        rmse_denorm = -1
        r2_denorm   = -1

    return rmse, rmse_denorm, r2, r2_denorm


### Helper functions for rmse_r2()
def squared_diffs(pred, true, y_mean):
    """
    Computes squared differences for RMSE/R2 calculations.
    """
    mss   = np.sum((pred - y_mean)**2, axis=0)
    tss   = np.sum((true - y_mean)**2, axis=0)
    rss   = np.sum((true - pred  )**2, axis=0)
    return mss, tss, rss

def integ_spec(pred, true, y_mean, x_vals, filttran, ifilt):
    """
    Integrates and predicted and true spectra according to filter
    bandpasses.
    """
    nfilters = len(filttran)
    pred_res = np.zeros((pred.shape[0], nfilters))
    true_res = np.zeros((true.shape[0], nfilters))
    y_mean_res = np.zeros(nfilters)
    for i in range(nfilters):
        pred_spec      = pred[:,ifilt[i,0]:ifilt[i,1]]
        true_spec      = true[:,ifilt[i,0]:ifilt[i,1]]
        xval_spec      = x_vals[ifilt[i,0]:ifilt[i,1]]
        y_mean_spec    = y_mean[ifilt[i,0]:ifilt[i,1]]
        pred_res[:, i] = np.trapz(pred_spec * filttran[i], xval_spec,
                                  axis=-1)
        true_res[:, i] = np.trapz(true_spec * filttran[i], xval_spec,
                                  axis=-1)
        y_mean_res[i]  = np.trapz(y_mean_spec * filttran[i], xval_spec)
    return pred_res, true_res, y_mean_res
