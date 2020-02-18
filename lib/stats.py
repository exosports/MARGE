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
            if ilog:
                data[:, :inD] = np.log10(data[:, :inD])
            if olog:
                data[:, inD:] = np.log10(data[:, inD:])
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


def rmse(nn, batch_size, num_batches, D, 
         y_mean, y_std, y_min, y_max, scalelims, preddir, 
         mode, denorm=False):
    """
    Calculates the RMSE for the specified trained models.

    Inputs
    ------
    nn          : object. NN model.
    batch_size  : int.    Size of batches for the model.
    num_batches : int.    Number of batches in the dataset.
    D           : int.    Dimensionality of the output.
    y_mean      : arr, float.  Mean of the output/target parameters.
    y_std       : arr, float.  Standard deviation of the output parameters.
    y_min       : arr, float.  Minima of the output parameters.
    y_max       : arr, float.  Maxima of the output parameters.
    scalelims   : list, float. [min, max] of range the data has been scaled to.
    preddir     : string. Path/to/directory to save the predictions.
    mode        : string. 'val' or 'test' to compute the RMSE for the 
                          validation or test data, respectively.
    denorm      : bool.   Determines if to calculate the denormalized
                          RMSE (True) or the normalized RMSE (False).

    Outputs
    -------
    rmse: array.  RMSE for each parameter.

    """
    if mode == 'val':
        fpred = preddir+'valid/'+mode+'pred'
        ftrue = preddir+'valid/'+mode+'true'
        X     = nn.Xval
        Y     = nn.Yval
    elif mode == 'test':
        fpred = preddir+'test/'+mode+'pred'
        ftrue = preddir+'test/'+mode+'true'
        X     = nn.Xte
        Y     = nn.Yte
    else:
        raise ValueError("Invalid specification for `mode` parameter of " + \
                         "rmse().\nAllowed options: 'val' or 'test'"      + \
                         "\nPlease correct this and try again.")

    # Variables for calculating RMSE
    n    = 0
    sqer = 0
    # Check for predictions
    predfoo = sorted(glob.glob(fpred+'*'))
    if len(predfoo) != num_batches:
        print("Predicting on input data...")
        preds = np.zeros((batch_size, D))
        for j in range(num_batches):
            fname = fpred+str(j).zfill(len(str(num_batches)))+'.npy'
            x_batch = K.eval(X)
            preds   = nn.model.predict(x_batch)
            np.save(fname, preds)
            print("  Batch "+str(j+1)+"/"+str(num_batches), end='\r')
        print('')
        predfoo = sorted(glob.glob(fpred+'*'))
    # Get the true values
    truefoo = sorted(glob.glob(ftrue+'*'))
    print("\nComputing squared error...")
    for i in range(num_batches):
        fname = ftrue+str(i).zfill(len(str(num_batches)))+'.npy'
        if len(truefoo) != num_batches:
            y_batch = K.eval(Y)
        else:
            y_batch = np.load(truefoo[i])
        # Denormalize the predictions and the normalized true values
        preds = np.load(predfoo[i])
        if denorm:
            preds   = U.denormalize(U.descale(preds, 
                                              y_min, y_max, scalelims),
                                    y_mean, y_std)
            y_batch = U.denormalize(U.descale(y_batch, 
                                              y_min, y_max, scalelims), 
                                    y_mean, y_std)

        # Add the squared difference to the running mean
        n    += y_batch.shape[0]
        sqer += np.sum((preds - y_batch)**2, axis=0)
        if len(truefoo) != num_batches:
            np.save(fname, y_batch)
        print("  Batch "+str(i+1)+"/"+str(num_batches), end='\r')
    # Calculate the RMSE and store it
    rmse = (sqer/n)**0.5
    print("")

    return rmse


def r2(nn, batch_size, num_batches, D, 
       y_mean, y_std, y_min, y_max, scalelims, preddir, 
       mode, denorm=False):
    """
    Computes the coefficient of determination (R2) between predictions and true
    values.

    Inputs
    ------
    nn          : object. NN model.
    batch_size  : int.    Size of batches for the model.
    num_batches : int.    Number of batches in the dataset.
    D           : int.    Dimensionality of the output.
    y_mean      : arr, float.  Mean of the output/target parameters.
    y_std       : arr, float.  Standard deviation of the output parameters.
    y_min       : arr, float.  Minima of the output parameters.
    y_max       : arr, float.  Maxima of the output parameters.
    scalelims   : list, float. [min, max] of range the data has been scaled to.
    preddir     : string. Path/to/directory to save the predictions.
    mode        : string. 'val' or 'test' to compute the RMSE for the 
                          validation or test data, respectively.
    denorm      : bool.   Determines if to calculate the denormalized
                          R2 (True) or the normalized R2 (False).

    Outputs
    -------
    R2_score: arr, float. R2 value for each output.

    """
    if mode == 'val':
        fpred = preddir+'valid/'+mode+'pred'
        ftrue = preddir+'valid/'+mode+'true'
        X     = nn.Xval
        Y     = nn.Yval
    elif mode == 'test':
        fpred = preddir+'test/'+mode+'pred'
        ftrue = preddir+'test/'+mode+'true'
        X     = nn.Xte
        Y     = nn.Yte
    else:
        raise ValueError("Invalid specification for `mode` parameter of " + \
                         "r2().\nAllowed options: 'val' or 'test'"        + \
                         "\nPlease correct this and try again.")

    mss = np.zeros(D) # Model sum of squares
    tss = np.zeros(D) # True  sum of squares
    rss = np.zeros(D) # Residual sum of squares
    # By definition, R2 = 1 - rss / tss
    # If mss + rss = tss then R2 = mss / tss

    print('\nCalculating R squared...')
    # Iterate through the batches
    for i in range(num_batches):
        # Load data
        suffix = str(i).zfill(len(str(num_batches)))+'.npy'
        pred   = np.load(fpred+suffix)
        true   = np.load(ftrue+suffix)

        if denorm:
            pred = U.denormalize(U.descale(pred, 
                                           y_min, y_max, scalelims),
                                 y_mean, y_std)
            true = U.denormalize(U.descale(true, 
                                           y_min, y_max, scalelims), 
                                 y_mean, y_std)
        else:
            y_mean = U.scale(U.normalize(y_mean, y_mean, y_std), 
                             y_min, y_max, scalelims)

        # Add the squared differences
        mss += np.sum((pred - y_mean)**2, axis=0)
        tss += np.sum((true - y_mean)**2, axis=0)
        rss += np.sum((true - pred  )**2, axis=0)
        print("  Batch "+str(i+1)+"/"+str(num_batches), end='\r')
    print('')

    R2_score = 1 - rss / tss

    return R2_score


