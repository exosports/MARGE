"""
This file contains utility functions that improve the usage of MARGE.

make_dir: Creates a directory if it does not already exist.

limit_mem: Sets a limit on the amount of memory the program can use.
           Not currently used by MARGE.

get_free_mem: Gets the amount of free memory on the system.
              Not currently used by MARGE.

get_num_per_file: Calculates the number of cases per file in a data set.
                  Not currently used by MARGE.

data_set_size: Calculates the number of cases in a data set.

concatdat: Handles concatenation of non-contiguous data within a data set.
           Not currently used by MARGE.

scale: Scales some data according to min, max, and a desired range.

descale: Descales some data according to min, max, and scaled range.

normalize: Normalizes some data according to mean & stdev.

denormalize: Denormalizes some data according to mean & stdev.

_float_feature: helper function to make Feature definition

_bytes_feature: helper function to make Feature definition

get_file_names: Find all file names in the data directory with a given file 
                extension.

make_TFRecord: Creates TFRecords representation of a data set.

get_TFR_file_names: Loads file names of the TFRecords files.

_parse_function: Helper function for loading TFRecords dataset objects.

load_TFdataset: Loads a TFRecords dataset for usage.

"""

import sys, os
import resource
import multiprocessing as mp
import functools
import glob
import numpy as np
import scipy.stats as ss
import tensorflow as tf

import loader as L


def make_dir(some_dir):
    """
    Handles creation of a directory.

    Inputs
    ------
    some_dir: string. Directory to be created.

    Outputs
    -------
    None. Creates `some_dir` if it does not already exist. 
    Raises an error if the directory cannt be created.
    """
    try:
      os.mkdir(some_dir)
    except OSError as e:
      if e.errno == 17: # Already exists
        pass
      else:
        print("Cannot create folder '{:s}'. {:s}.".format(model_dir,
                                              os.strerror(e.errno)))
        sys.exit()
    return


def limit_mem():
    """
    Function written to limit memory usage. 
    Unfortunately, it limits GPU usage too, so it isn't used.
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (get_free_mem() * 1024 * 0.9, hard))


def get_free_mem():
    """
    Function written to return available memory.
    See limit_mem().
    """
    with open('/proc/meminfo', 'r') as meminfo:
        free_memory = 0
        for line in meminfo:
            entries = line.split()
            if str(entries[0]) == 'MemAvailable:':
                free_memory = int(entries[1])
                break
    return free_memory


def get_num_per_file(foos, nfoo=10):
    """
    Loads a few files to determine the number of entries per data file.
    Not used by MARGE, but left here in case a user wants to use it for 
    some data set where the data files adhere to the same # of cases 
    per data file.

    Inputs
    ------
    foos: list, strings. Paths/to/data files.
    nfoo: int. Number of files to consider for the calculation.

    Outputs
    -------
    num_per_file: int. Number of entries per data file.
    """
    num_in_file = np.zeros(min(len(foos), nfoo))
    for i in range(min(len(foos), nfoo)):
        num_in_file[i] = np.load(foos[i]).shape[0]
    return int(ss.mode(num_in_file, None, 'raise')[0][0])


def count_cases(foo):
    """
    Helper function for multiprocessing. Counts number of cases in file.
    """
    return np.load(foo).shape[0]


def data_set_size(foos, ncores=1):
    """
    Loads a data set and counts the total number of cases.

    Inputs
    ------
    foos: list, strings. Files of the data set.

    Outputs
    -------
    ncases: int. Number of cases in the data set.

    Notes
    -----
    It is assumed that each case in the data set occurs along axis 0.
    """
    pool   = mp.Pool(ncores)
    # Count all cases in each file, in parallel
    ncases = pool.map(count_cases, foos)
    pool.close()
    pool.join()
    return np.sum(ncases)


def concatdat(xi, xlen, yi, ylen, 
              arr1, arr2):
    """
    Helper function to handle slicing and concatenation of non-contiguous data.
    Not used by MARGE, but left available in case a user wishes to utilize it.

    Inputs
    ------
    xi  : array-like. Holds the starting indices for slices.
    xlen: array-like. Holds the length of slices. Must be same shape as `xi`.
    yi  : array-like. Holds the starting indices for slices.
    ylen: array-like. Holds the length of slices. Must be same shape as `yi`.
    arr1: array-like. Data to be sliced according to `xi`,     `xlen`, 
                                                     `yi`, and `ylen`.
    arr2: array-like. Data related to `arr1`.

    Outputs
    -------
    x1: array. `arr1` sliced and concatenated according to `xi` and `xlen`.
    x2: array. `arr2` '   '   '   '   '   '   '   '   '   '   '   '   '   '
    y1: array. `arr1` '   '   '   '   '   '   '   '   '    `yi` and `ylen`.
    y2: array. `arr2` '   '   '   '   '   '   '   '   '   '   '   '   '   '
    """
    # Make the initial slices
    x1 = arr1[xi[0]:xi[0]+xlen[0]]
    x2 = arr2[xi[0]:xi[0]+xlen[0]]
    y1 = arr1[yi[0]:yi[0]+ylen[0]]
    y2 = arr2[yi[0]:yi[0]+ylen[0]]

    # Concatenate non-contiguous regions of the array, if needed
    if len(xi) > 1:
        for i in range(1, len(xi)):
            ibeg = xi[i]
            iend = xi[i]+xlen[i]
            x1   = np.concatenate((x1, arr1[ibeg:iend]))
            x2   = np.concatenate((x2, arr2[ibeg:iend]))
    if len(yi) > 1:
        for i in range(1, len(yi)):
            ibeg = yi[i]
            iend = yi[i]+ylen[i]
            y1   = np.concatenate((y1, arr1[ibeg:iend]))
            y2   = np.concatenate((y2, arr2[ibeg:iend]))

    return x1, x2, y1, y2


def scale(val, vmin, vmax, scalelims):
    """
    Scales a value according to min/max values and scaling limits.

    Inputs
    ------
    val      : array. Values to be scaled.
    vmin     : array. Minima of `val`.
    vmax     : array. Maxima of `val`.
    scalelims: list, floats. [min, max] of range of scaled data.

    Outputs
    -------
    Array of scaled data.
    """
    return (scalelims[1] - scalelims[0]) * (val - vmin) / \
           (vmax - vmin) + scalelims[0]


def descale(val, vmin, vmax, scalelims):
    """
    Descales a value according to min/max values and scaling limits.

    Inputs
    ------
    val      : array. Values to be descaled.
    vmin     : array. Minima of `val`.
    vmax     : array. Maxima of `val`.
    scalelims: list, floats. [min, max] of range of scaled data.

    Outputs
    -------
    Array of descaled data.
    """
    return (val - scalelims[0]) / (scalelims[1] - scalelims[0]) * \
           (vmax - vmin) + vmin


def normalize(val, vmean, vstd):
    """
    Normalizes a value according to a mean and standard deviation.

    Inputs
    ------
    val  : array. Values to be normalized.
    vmean: array. Mean  values of `val`.
    vstd : array. Stdev values of `val`.

    Outputs
    -------
    Array of normalized data.
    """
    return (val - vmean) / vstd


def denormalize(val, vmean, vstd):
    """
    Denormalizes a value according to a mean and standard deviation.

    Inputs
    ------
    val  : array. Values to be denormalized.
    vmean: array. Mean  values of `val`.
    vstd : array. Stdev values of `val`.

    Outputs
    -------
    Array of denormalized data.
    """
    return val * vstd + vmean


def _float_feature(value):
    """
    Helper function to make feature definition more readable
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """
    Helper function to make feature definition more readable
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def make_TFRecord(fname, files, inD, ilog, olog, 
                  batch_size, e_batches, split=1, verb=1):
    """
    Function to write TFRecords for large data sets.

    Inputs
    ------
    fname: string. Base name for TFRecords files
    files: list, strings. Files to process into TFRecords.
    inD  : int. Dimension of the inputs/features
    ilog : bool. Determines if to take the log of inputs/features
    olog : bool. Determines if to take the log of outputs/targets
    batch_size: int. Size of batches for training.
    e_batches : int. Expected number of batches to be processed.
    split: int. Determines the number of `files` to process before 
                starting a new TFRecords file.
    verb : int. Verbosity level.

    Outputs
    -------
    TFRecords files.
    """
    if verb > 1:
        print("\nWriting TFRecords file...")

    # Track number of files left to load
    filesleft = len(files)
    # Track number of cases written to TFRecord
    N_batches = 0
    count     = 0

    for i in range(int(np.ceil(len(files)/split))):
        # TFRecords file
        thisfile = fname.replace('.tfrecords', '_'+str(i).zfill(3)+'.tfrecords')
        writer = tf.python_io.TFRecordWriter(thisfile)
        for j in range(min(split, filesleft)):
            # Load file
            x, y = L.load_data_file(files[i*split+j], 
                                                inD, ilog, olog)
            filesleft -= 1
            # Print progress updates
            if verb:
                print(str(int(100/len(files)/split*(i + j/split))) + \
                      "% complete", end='\r')
            # TF is lame and requires arrays to be 1D. Write each sequentially
            for k in range(x.shape[0]):
                # Check for Nans
                if np.any(np.isnan(x[k])) or np.any(np.isnan(y[k])):
                    if verb:
                        print("Nan alert!", files[i*split+j], k)
                    continue
                # Define feature
                feature = {'x' : _bytes_feature(
                                        tf.compat.as_bytes(x[k].tostring())),
                           'y' : _bytes_feature(
                                        tf.compat.as_bytes(y[k].tostring()))}
                # Create an example protocol buffer
                example = tf.train.Example(features=tf.train.Features(feature=
                                                                      feature))
                # Serialize to string and write on the file
                writer.write(example.SerializeToString())
                # Make sure that the TFRecord has exactly N*batch_size entries
                count += 1
                if count==batch_size:
                    count      = 0
                    N_batches += 1
                if N_batches==e_batches:
                    writer.close()
                    if verb:
                        print("100% complete")
                    if verb > 1:
                        print("Ended writing TFRecords to ensure " + \
                              "N*batch_size entries.")
                        print("Writing TFRecords file complete.")
                    return
        writer.close()
    if verb:
        print("100% complete")
    if verb > 1:
        print("Writing TFRecords file complete.")
        print(N_batches, 'batches written,', e_batches, 'batches expected.')
        print(count, 'remaining count.')
    return


def _parse_function(proto, xlen, ylen, 
                    x_mean=None, x_std=None, y_mean=None, y_std=None, 
                    x_min=None,  x_max=None, y_min=None,  y_max=None, 
                    scalelims=None):
    """
    Helper function for loading TFRecords
    
    Inputs
    ------
    proto : object. Tensorflow Dataset.
    xlen  : int.    Number of inputs.
    ylen  : int.    Number of outputs.
    x_mean: array.  Mean  values of input  data.
    x_std : array.  Stdev values of input  data.
    y_mean: array.  Mean  values of output data.
    y_std : array.  Stdev values of output data.
    x_min : array.  Minima of input  data.
    x_max : array.  Maxima of input  data.
    y_min : array.  Minima of output data.
    y_max : array.  Maxima of output data.
    scalelims: list, floats. [min, max] of range of scaled data.

    Outputs
    -------
    x: Parsed inputs.
    y: Parsed outputs.
    """
    # Define the TFRecord
    keys_to_features = {"x" : tf.FixedLenFeature([], tf.string),
                        "y" : tf.FixedLenFeature([], tf.string)}
    
    # Load one example
    parsed_features = tf.parse_single_example(proto, keys_to_features)

    # Turn string into array
    x = tf.decode_raw(parsed_features["x"], tf.float64)
    y = tf.decode_raw(parsed_features["y"], tf.float64)

    # Make sure it has the right shape
    x = tf.reshape(x, (np.sum(xlen),))
    y = tf.reshape(y, (np.sum(ylen),))

    # Parameters to process data
    norm    = (x_mean, x_std, y_mean, y_std)
    scaling = (x_min,  x_max, y_min,  y_max, scalelims)
    # Set defaults if not specified
    if any(v is None for v in norm):
        x_mean    = 0
        x_std     = 1
        y_mean    = 0
        y_std     = 1
    if any(v is None for v in scaling):
        x_min     = 0
        x_max     = 1
        y_min     = 0
        y_max     = 1
        scalelims = [0, 1]

    # Normalize and scale
    x = scale(normalize(x, x_mean, x_std), x_min, x_max, scalelims)
    y = scale(normalize(y, y_mean, y_std), y_min, y_max, scalelims)

    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)

    return x, y


def load_TFdataset(files, ncores, batch_size, buffer_size, 
                   xlen, ylen, 
                   x_mean=None, x_std=None, y_mean=None, y_std=None,
                   x_min=None,  x_max=None, y_min=None,  y_max=None, 
                   scalelims=None, shuffle=False):
    """
    Builds data loading pipeline for TFRecords.

    Inputs
    ------
    files      : list, str. Path/to/files for TFRecords.
    ncores     : int. Number of cores to use for parallel loading.
    batch_size : int. Batch size.
    buffer_size: int. Number of times `batch_size` to use for buffering.
    xlen       : int. Number of inputs.
    ylen       : int. Number of outputs.
    x_mean     : float.  Mean of inputs.
    x_std      : float.  Standard deviation of inputs.
    y_mean     : float.  Mean of outputs.
    y_std      : float.  Standard deviation of outputs.
    x_min      : array.  Minima of input  data.
    x_max      : array.  Maxima of input  data.
    y_min      : array.  Minima of output data.
    y_max      : array.  Maxima of output data.
    scalelims  : list, floats. [min, max] of range of scaled data.
    shuffle    : bool.         Determines whether to shuffle the order or not.

    Outputs
    -------
    x_data: Parsed input  data.
    y_data: Parsed output data.
    """
    # Make dataset
    dataset = tf.data.TFRecordDataset(files)
    # Make static parse_function
    parse_function = functools.partial(_parse_function, 
                                       xlen=xlen, ylen=ylen, 
                                       x_mean=x_mean, x_std=x_std, 
                                       y_mean=y_mean, y_std=y_std,
                                       x_min=x_min,   x_max=x_max, 
                                       y_min=y_min,   y_max=y_max, 
                                       scalelims=scalelims)
    # Maps the parser on every filepath in the array
    dataset = dataset.map(parse_function, num_parallel_calls=ncores)
    # Shuffle buffer -- train in random order
    if shuffle:
        dataset = dataset.shuffle(buffer_size*batch_size, 
                                  reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.repeat() # Go forever! Until fit() stops it
    dataset = dataset.prefetch(buffer_size)

    # Make iterator to handle loading the data
    iterator = dataset.make_one_shot_iterator()
    # Create TF representation of the iterator
    x_data, y_data = iterator.get_next()

    return x_data, y_data


