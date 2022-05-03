"""
Module that contains functions related to statstics.

mean_stdev: Uses Welford's method to calculate the mean and standard deviation 
            of the entire dataset, without loading all data in memory at once.

rmse: Calculates the RMSE for some predictions vs true values.

r2: Calcualtes the coefficient of determination (R^2) for some predictions 
    vs true values.

"""

import sys, os
import glob
import numpy as np
import scipy.interpolate as si

from keras import backend as K

import utils as U


def mean_stdev(datafiles, inD, ilog, olog, perc=10, num_per_file=None, verb=False):
    """
    Uses Welford's method to calculate the mean and standard deviation (via the 
    variance) of the entire dataset, without loading all data in memory at once.

    Inputs
    ------
    datafiles: list, strings. [path/to/datafile1.npy, path/to/datafile2.npy, ... ]
    inD : int.  Dimension of the input data.
    ilog: bool. Determines whether to take the log10 of the input data/features.
    olog: bool. Determines whether to take the log10 of the output/target data.
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
    mean  : arr, float. Mean  values of the dataset.
    stdev : arr, float. Stdev values of the dataset. 
                        Computed as sqrt of sample variance.
    datmin: arr, float. Minima of the dataset.
    datmax: arr, float. Maxima of the dataset.

    Notes
    -----
    https://www.johndcook.com/blog/standard_deviation/
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

    """
    up     = 0  # Counter for checking percent done
    n_ele  = np.load(datafiles[0]).shape[-1] # Data vector size
    nc     = 0                         # Number of cases seen thus far
    mean   = np.zeros(n_ele)           # Running mean
    M2     = np.zeros(n_ele)           # Running variance
    datmin = np.ones (n_ele) *  np.inf # Min for each element
    datmax = np.ones (n_ele) * -np.inf # Max for each element

    for i in range(len(datafiles)):
        data = np.load(datafiles[i])
        # Files must be 2D
        if len(data.shape) == 1:
            print("*WARNING*: Broken file!!!")
            print(datafiles[i])
            print("Shape:", data.shape)
            continue
        elif len(data.shape) == 2:
            if num_per_file is not None:
                # Check that it's the right shape
                if verb and data.shape[0] != num_per_file:
                    print("Warning: Incomplete file!", datafiles[i], data.shape)
            # Take logs
            np.seterr(all='raise')
            if ilog:
                data[:, :inD][:, ilog] = np.log10(data[:, :inD][:, ilog])
            if olog:
                data[:, inD:][:, olog] = np.log10(data[:, inD:][:, olog])
            # Update min/max
            datmin = np.amin(np.vstack([datmin, data]), axis=0)
            datmax = np.amax(np.vstack([datmax, data]), axis=0)
            # Consider each data vector in this file
            for j in range(data.shape[0]):
                if np.all(data[j] == 0):
                    if verb:
                        print("We gotta lotta zeros here:", datafiles[i], j)
                    continue
                nc    += 1
                delta  = data[j] - mean
                mean   = mean + delta / nc
                M2     = M2   + delta * (data[j] - mean)
                #M2     = M2   + (nc-1) * delta**2 / nc    # alt way to calc this
        else:
            raise ValueError("File", datafiles[i], "has shape", data.shape,
                             ".\nThis data format is not understood.", 
                             "\nPlease reformat the data,",
                             "or code this ino stats.py.")
        # Print status updates
        if verb and (100*i // len(datafiles)) >= (up+1)*perc:
            up += ((100*i // len(datafiles)) - up*perc) // perc
            print(len(str(up*perc))*'-' + "----------")
            print(str(up*perc)+"% complete")
            print("mean:", mean)
            print("var :", M2/(nc-1))
            print(len(str(up*perc))*'-' + "----------")

    print('-----------------------------------------')
    print('Completed mean/stdev/min/max calculations')
    print('-----------------------------------------')
    variance = M2 / (nc - 1)

    return mean, variance**0.5, datmin, datmax


def rmse_r2(fpred, ftrue, y_mean, 
            y_std=None, y_min=None, y_max=None, scalelims=None, 
            olog=False, y_mean_delog=None, 
            filters=None, x_vals=None, filt2um=1.0):
    """
    Calculates the root mean squared error (RMSE) and R-squared for a data set.
    Data must be saved in .NPY file(s), and `fpred` and `ftrue` must exactly 
    correspond.

    Default behavior: compute RMSE/R2 for the raw predictions.

    If y_std, y_min, y_max, scalelims, and olog are specified, 
    it will compute RMSE/R2 for both the raw and denormalized predictions.

    If filters and x_vals are specified, then the RMSE/R2 will be computed on  
    the integrated filter bandpasses, rather than for each output parameter.

    Inputs
    ------
    fpred: list, strings. .NPY files to compute RMSE, predicted values.
    ftrue: list, strings. .NPY files to compute RMSE, true      values.
    y_mean: array. Training set mean value of each output parameter.
    y_std : array. Training set standard deviation of each output parameter.
    y_min : array. Training set minimum of each output parameter.
    y_max : array. Training set maximum of each output parameter.
    scalelims: tuple/list/array. Min/max that the normalized data was scaled to.
    olog  : bool.  Determines if the denormalized values are the log10 of the 
                   true output parameters.
    y_mean_delog: array. Mean of the training set, not log-scaled.
    filters: list, strings. Filter bandpasses to integrate over.
                            Must be 2-column file: wavelength then transmission.
    x_vals : array.         X values corresponding to the Y values.
    filt2um: float.         Conversion factor for filter's wavelengths to 
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
        raise Exception("The prediction/true file structures do not match.\n" +\
                        "Ensure that each set of files follows the same "     +\
                        "exact structure and order.\nSee NNModel.Yeval().")

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
            finterp = si.interp1d(10000. / (filt2um * datfilt[:,0]), 
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
            raise ValueError("Must give the non-log-scaled training set mean.")

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

    ### Helper functions to avoid repeating code
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
        Integrates and predicted and true spectrum according to filter 
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

    # Compute RMSE & R2
    for j in range(len(fpred)):
        # Load batch
        pred  = np.load(fpred[j])
        true  = np.load(ftrue[j])
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
                pred[:,olog] = 10**pred[:,olog]
                true[:,olog] = 10**true[:,olog]
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
    

