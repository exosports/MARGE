"""
This file contains utility functions that improve the usage of MARGE.

make_dir: Creates a directory if it does not already exist.

load_path: Helper function for reading in config parameters with file paths.

prepare_layers: Helper function to ensure the supplied architecture parameters
                are valid

prepare_gridsearch: Helper function to ensure the set of architectures
                    specified in the config file are valid

count_cases: Helper function for multiprocessing.
             Counts number of cases in file.

data_set_size: Calculates the number of cases in a data set.

get_data_set_sizes:  Calls data_set_size and calculates the number of batches 
                     in each data subset.

concatdat: Handles concatenation of non-contiguous data within a data set.
           Not currently used by MARGE.

scale: Scales some data according to min, max, and a desired range.

descale: Descales some data according to min, max, and scaled range.

normalize: Normalizes some data according to mean & standard deviation.

denormalize: Denormalizes some data according to mean & standard deviation.

_float_feature: helper function to make Feature definition

_bytes_feature: helper function to make Feature definition

get_file_names: Find all file names in the data directory with a given file
                extension.

make_TFRecord: Creates TFRecords representation of a data set.

get_TFR_file_names: Loads file names of the TFRecords files.

_parse_function: Helper function for loading TFRecords dataset objects.

load_TFdataset: Loads a TFRecords dataset for usage.

check_TFRecords: Checks if TFRecords files need to be (re)created.

write_ordered_optuna_summary: Saves a human-readable summary of an Optuna run.

"""

import sys, os
import multiprocessing as mp
import functools
import glob
import logging
logging.setLoggerClass(logging.Logger)
import types
import numpy as np
import pickle
import scipy.stats as ss
import tensorflow as tf

import loader as L
import custom_logger as CL
logging.setLoggerClass(CL.MARGE_Logger)

logger = logging.getLogger('MARGE.'+__name__)


# Dictionary of layer names, with the value a bool whether it has nodes or not
# spatialconcretedropout does not currently work w/ TFv2 - needs to be updated
layer_types_has_nodes = {
            "dense" : True, "dropout" : False, 
            "concretedropout" : True, #"spatialconcretedropout" : True, 
            "conv1d" : True, "conv2d" : True, "conv3d" : True, 
            "conv2dtranspose" : True, "conv2dtranspose" : True, 
            "separableconv1d" : True, "separableconv2d" : True, 
            "depthwiseconv1d" : True, "depthwiseconv2d" : True, 
            "maxpool1d" : False, "maxpool2d" : False, "maxpool3d" : False,
            "avgpool1d" : False, "avgpool2d" : False, "avgpool3d" : False,
            "flatten" : False, "batchnorm" : False
}


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
        os.makedirs(some_dir)
        logger.info("Created directory: " + some_dir)
    except OSError as e:
        if e.errno == 17: # Already exists
            logger.info("Directory already exists: " + some_dir)
            pass
        else:
            logger.error("Cannot create directory: {:s}\n{:s}.".format(some_dir,
                                                          os.strerror(e.errno)))
            sys.exit(1)
    return


def load_path(key, basedir=""):
    """
    Helper function for reading in config parameters with file paths.
    Returns an absolute path.
    If `key` is a relative path, then it is assumed to be relative to `basedir`.
    """
    if os.path.isabs(key):
        return key
    else:
        pth = os.path.join(basedir, key)
        logger.debug("Relative file path '" + key + "' assumed to be " + pth)
        return pth


def prepare_layers(layers, lay_params, nodes, activations=None, act_params=None):
    """
    Helper function to ensure the supplied architecture parameters are valid

    Inputs
    ------
    layers     : list, strings. Layer types for each layer in the architecture.
    lay_params : list, ints or None.  Parameters for each layer, where needed.
    nodes      : list, ints.  Number of nodes in each layer that contains nodes.
    activations: list, strings. Activation function types for each layer in the
                                architecture.
    act_params : list, ints or None. Parameters for each activation function,
                                     where needed.

    Outputs
    -------
    lay_params : like the input, except Nones have been replaced by defaults
                 for layer types that have a free parameter
    activations: like the input, except the strings are now function objects
    """
    # Check for allowed layer types
    for lay in layers:
        if lay not in layer_types_has_nodes.keys():
            raise TypeError('Invalid layer type specified.\n'    \
                          + 'Given layer type: ' + lay + '\n'    \
                          + 'Allowed options:\n'                 \
                          + ', '.join(list(layer_types_has_nodes.keys())))
    # Make sure the right number of entries exist
    #nlay = layers.count("dense")  + layers.count("conv1d") + \
    #       layers.count("conv2d") + layers.count("conv3d") + \
    #       layers.count("conv2dtranspose") + layers.count("conv3dtranspose") + \
    #       layers.count("concretedropout") + layers.count("spatialconcretedropout")
    nlay = np.sum([layers.count(l) for l in layer_types_has_nodes.keys() \
                   if layer_types_has_nodes[l]])
    if nlay != len(nodes):
        raise ValueError("Number of Dense/Conv layers does "  \
            + "not match the number of hidden\nlayers with " \
            + "nodes.")
    if len(layers) != len(lay_params):
        raise ValueError("Number of layer types does not match " \
            + "the number of layer parameters.")
    else:
        # Set default layer parameters if needed
        for j in range(len(layers)):
            if lay_params[j] in [None, 'None', 'none']:
                if 'conv1d' in layers[j]:
                    lay_params[j] = 3
                elif 'conv2d' in layers[j]:
                    lay_params[j] = (3,3)
                elif 'conv3d' in layers[j]:
                    lay_params[j] = (3,3,3)
                elif layers[j] in ['maxpool1d', 'avgpool1d']:
                    lay_params[j] = 2
                elif layers[j] in ['maxpool2d', 'avgpool2d']:
                    lay_params[j] = (2,2)
                elif layers[j] in ['maxpool3d', 'avgpool3d']:
                    lay_params[j] = (2,2,2)
                else:
                    lay_params[j] = None
            elif layers[j] == 'dropout':
                lay_params[j] = float(lay_params[j])
            else:
                lay_params[j] = int(lay_params[j])
    if activations is not None and act_params is not None:
        if len(activations) != len(nodes):
            raise ValueError("Number of activation functions does "    \
                + "not match the number of hidden\nlayers with nodes.")
        if len(activations) != len(act_params):
            raise ValueError("Number of activation functions does not " \
                + "match the number of\nactivation function "          \
                + "parameters.")
        else:
            # Load the activation functions
            for j in range(len(activations)):
                activations[j] = L.load_activation(activations[j],
                                                   act_params[j])
    return lay_params, activations


def prepare_gridsearch(layers, lay_params, nodes, activations, act_params):
    """
    Helper function to ensure the set of architectures specified in the config
    file are valid.

    Inputs
    ------
    layers, lay_params, nodes, activations, and act_params are lists of lists.
    See prepare_layers() for a description of individual list elements.

    Outputs
    -------
    lay_params : like the input, except Nones have been replaced by defaults
                 for layer types that have a free parameter
    activations: like the input, except the strings are now function objects
    """
    # Check that there are the proper number of entries for each key
    if len(layers) != len(lay_params):
        raise ValueError("Number of sets of layers and sets " \
                       + "of layer parameters do not match.")
    elif len(layers) != len(nodes):
        raise ValueError("Number of sets of layers and sets " \
                       + "of nodes do not match.")
    elif len(layers) != len(activations):
        raise ValueError("Number of sets of layers and sets " \
                       + "of activations do not match.")
    elif len(layers) != len(act_params):
        raise ValueError("Number of sets of layers and sets " \
                       + "of activation parameters do not match.")
    # Check that each architecture is consistent and able to be set up
    for i in range(len(nodes)):
        lay_params[i], activations[i] = prepare_layers(layers[i], lay_params[i],
                                                       nodes[i], activations[i],
                                                       act_params[i])
    return lay_params, activations


def count_cases(foo, ishape, oshape):
    """
    Helper function for multiprocessing. Counts number of cases in file.
    """
    data = np.load(foo)
    x = data['x']
    y = data['y']
    if x.shape[0 ] == y.shape[0] and \
       x.shape[1:] == ishape     and \
       y.shape[1:] == oshape:
        # File has multiple cases
        return x.shape[0]
    else:
        # File has a single case
        """
        if x.shape != ishape:
            # Shapes don't match
            raise ValueError("This file has improperly shaped input data:\n"+\
                             foo+ "\nExpected shape: " +\
                             str(ishape) + "\nReceived shape: " + str(x.shape))
        elif y.shape != oshape:
            raise ValueError("This file has improperly shaped output data:\n"+\
                             foo+ "\nExpected shape: " +\
                             str(oshape) + "\nReceived shape: " + str(y.shape))
        else:
            raise ValueError("The inputs and outputs of this file do not "+\
                             "have the same number of cases:\n" + foo     +\
                             "\nInput  shape: "+str(x.shape)+
                             "\nOutput shape: "+str(y.shape))
        """
        raise ValueError("It appears as if each data file has a single case, but " +\
                         "there is no explicit batch dimension to indicate that. " +\
                         "Reshape your data to have a batch dimension at the " +\
                         "zero-th axis and try again.\n" +\
                         "Expected input  shape: (Nbatch," + str(ishape).replace('(', '') + '\n' +\
                         "Received input  shape: " + str(x.shape) + '\n' +\
                         "Expected output shape: (Nbatch," + str(oshape).replace('(', '') + '\n' +\
                         "Received output shape: " + str(y.shape) + '\n')


def data_set_size(foos, ishape, oshape, ncores=1):
    """
    Loads a data set and counts the total number of cases.

    Inputs
    ------
    foos: list, strings. Files of the data set.
    ishape: tuple, ints. Dimensionality of the input  data, without the batch size.
    oshape: tuple, ints. Dimensionality of the output data, without the batch size.
    ncores: int.  Number of cores to use to load data in parallel.  Default: 1.
    
    Outputs
    -------
    ncases: int. Number of cases in the data set.

    Notes
    -----
    It is assumed that each case in the data set occurs along axis 0.
    """
    # Create counter function
    fcount = functools.partial(count_cases, ishape=ishape, oshape=oshape)
    pool = mp.Pool(ncores)
    # Count all cases in each file, in parallel
    ncases = pool.map(fcount, foos)
    pool.close()
    pool.join()
    return np.sum(ncases)


def get_data_set_sizes(fsize, datadir, ishape, oshape, batch_size, ncores=1):
    """
    Counts the number of data cases in each subset and computes the number of 
    batches per subset.
    
    Inputs
    ------
    fsize: string.  Path to save file containing number of data cases in each subset.
    datadir: string.  Path to directory holding the training, validation, and test subsets.
    ishape: tuple, ints. Dimensionality of the input  data, without the batch size.
    oshape: tuple, ints. Dimensionality of the output data, without the batch size.
    batch_size: int.  Size of batches for training/validating/testing.
    ncores: int.  Number of cores to use to load data in parallel.  Default: 1.
    
    Outputs
    -------
    train_batches: int.  Number of batches in the training   set.
    valid_batches: int.  Number of batches in the validation set.
    test_batches : int.  Number of batches in the test       set.
    """
    logger.newline()
    logger.info('Loading files & calculating total number of batches...')
    
    # Check if we can depend on `fsize` or if we need to (re)create it

    try:
        datsize   = np.load(fsize)
        num_train = datsize[0]
        num_valid = datsize[1]
        num_test  = datsize[2]
    except:
        num_train = data_set_size(glob.glob(datadir + 'train' + os.sep + '*.npz'), ishape, oshape, ncores)
        num_valid = data_set_size(glob.glob(datadir + 'valid' + os.sep + '*.npz'), ishape, oshape, ncores)
        num_test  = data_set_size(glob.glob(datadir + 'test'  + os.sep + '*.npz'), ishape, oshape, ncores)
        datsize   = np.array([num_train, num_valid, num_test], dtype=int)
        np.save(fsize, datsize)
    
    if not num_train:
        raise ValueError("No training data provided.\n"+\
                         "Are you sure your data are located at\n"+\
                         datadir+'train'+os.sep+'*.npz?')
    if not num_valid:
        raise ValueError("No validation data provided.\n"+\
                         "Are you sure your data are located at\n"+\
                         datadir+'valid'+os.sep+'*.npz?')
    if not num_test:
        raise ValueError("No test data provided.\n"+\
                         "Are you sure your data are located at\n"+\
                         datadir+'test'+os.sep+'*.npz?')
    
    dset_str  = "Data set sizes:\n"
    dset_str += "    Training   data: " + str(num_train).rjust(len(str(datsize.sum()))) + "\n"
    dset_str += "    Validation data: " + str(num_valid).rjust(len(str(datsize.sum()))) + "\n"
    dset_str += "    Testing    data: " + str(num_test).rjust(len(str(datsize.sum()))) + "\n"
    dset_str += "    Total          : " + str(num_train + num_valid + num_test).rjust(len(str(datsize.sum())))
    logger.info(dset_str)
    
    train_batches = num_train // batch_size
    valid_batches = num_valid // batch_size
    test_batches  = num_test  // batch_size

    return train_batches, valid_batches, test_batches


def concatdat(xi, xlen, yi, ylen,
              arr1, arr2):
    """
    Helper function to handle slicing and concatenation of non-contiguous data.
    No longer used by MARGE, but left available in case a user wishes to utilize it.

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


def make_TFRecord(fname, files, ilog, olog, 
                  batch_size, e_batches, split=1, verb=1):
    """
    Function to write TFRecords for large data sets.

    Inputs
    ------
    fname: string. Base name for TFRecords files
    files: list, strings. Files to process into TFRecords.
    ilog : bool or array of bools. Determines if to take the log of inputs/features.
    olog : bool or array of bools. Determines if to take the log of outputs/targets.
    batch_size: int. Size of batches for training.
    e_batches : int. Expected number of batches to be processed.
    split: int. Determines the number of `files` to process before
                starting a new TFRecords file.
    verb : int. Verbosity level.

    Outputs
    -------
    TFRecords files.
    """
    logger.info("Writing TFRecords files...")

    # Track number of files left to load
    filesleft = len(files)
    # Track number of cases written to TFRecord
    N_batches = 0
    count     = 0

    prog = 0
    fullbreak = False
    for i in range(int(np.ceil(len(files)/split))):
        if fullbreak:
            break
        # TFRecords file
        thisfile = fname.replace('.tfrecords', '_'+str(i).zfill(3)+'.tfrecords')
        writer = tf.io.TFRecordWriter(thisfile)
        for j in range(min(split, filesleft)):
            if fullbreak:
                break
            # Load file
            x, y = L.load_data_file(files[i*split+j], ilog, olog)
            if x.shape[0] != y.shape[0]:
                raise ValueError("The inputs and outputs of this file do not "+\
                                 "have the same number of cases:\n"           +\
                                 files[i*split+j])
            # Save/check typing
            if not i and not j:
                xdtype = x.dtype
                ydtype = y.dtype
            else:
                assert x.dtype == xdtype, "x array data type in file "+files[i*split+j]+" does not match data type in file "+files[0]
                assert y.dtype == ydtype, "y array data type in file "+files[i*split+j]+" does not match data type in file "+files[0]
            filesleft -= 1
            # Print progress updates every ~25%
            new_prog = int(100/len(files)/split*(i + j/split))
            if new_prog >= prog:
                logger.info("  " + str(new_prog).rjust(3) + "% complete")
                while new_prog >= prog:
                    prog += 25
            # TF is lame and requires arrays to be 1D. Write each sequentially
            for k in range(x.shape[0]):
                # Check for Nans
                if np.any(np.isnan(x[k])) or np.any(np.isnan(y[k])):
                    logger.error("NaN found in file " + files[i*split+j] + " at index " + str(k))
                    sys.exit(1)
                # Define feature
                feature = {'x'  : _bytes_feature(
                                        tf.compat.as_bytes(x[k].flatten().tostring())),
                           'y'  : _bytes_feature(
                                        tf.compat.as_bytes(y[k].flatten().tostring()))}
                # Create an example protocol buffer
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                # Serialize to string and write on the file
                writer.write(example.SerializeToString())
                # Make sure that the TFRecord has exactly N*batch_size entries
                count += 1
                if count==batch_size:
                    count      = 0
                    N_batches += 1
                if N_batches==e_batches:
                    fullbreak = True
                    break
        writer.close()
    logger.info("  100% complete")
    if fullbreak:
        logger.info("Ended writing TFRecords to ensure " + \
                    "N*batch_size entries.")
    logger.info("Writing TFRecords files complete.")
    if N_batches != e_batches:
        logger.warning(str(N_batches) + ' batches written, ' + str(e_batches) + ' batches expected.')
    else:
        logger.info(str(N_batches) + ' batches written.')
    if count:
        logger.info(str(count) + ' remaining count.')
    logger.newline()
    return xdtype, ydtype


def _parse_function(proto, ishape, oshape, xdtype, ydtype, 
                    x_mean=None, x_std=None, y_mean=None, y_std=None,
                    x_min=None,  x_max=None, y_min=None,  y_max=None,
                    scalelims=None):
    """
    Helper function for loading TFRecords

    Inputs
    ------
    proto : object. Tensorflow Dataset.
    ishape: tuple, ints. Shape of the input  data.
    oshape: tuple, ints. Shape of the output data.
    xdtype: obj.  Data type of the input  data.
    ydtype: obj.  Data type of the output data.
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
    keys_to_features = {"x" : tf.io.FixedLenFeature([], tf.string),
                        "y" : tf.io.FixedLenFeature([], tf.string)}

    # Load one example
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)

    # Turn string into array
    x = tf.io.decode_raw(parsed_features["x"], tf.dtypes.as_dtype(xdtype))
    y = tf.io.decode_raw(parsed_features["y"], tf.dtypes.as_dtype(ydtype))

    # Make sure it has the right shape
    x = tf.reshape(x, ishape)
    y = tf.reshape(y, oshape)
    
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
                   ishape, oshape, 
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
    ishape: tuple, ints. Shape of the input  data.
    oshape: tuple, ints. Shape of the output data.
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
    if shuffle:
        # Interleaves reads from multiple files - credit: https://keras.io/examples/keras_recipes/tfrecord/
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False
        dataset = dataset.with_options(ignore_order)
    # Load dtype info needed for parse_function
    fdtype = os.path.join(os.path.dirname(files[0]), 'dtype.npz')
    dtype_data = np.load(fdtype)
    # Make static parse_function
    parse_function = functools.partial(_parse_function,
                                       ishape=ishape, oshape=oshape,
                                       xdtype=dtype_data['xdtype'].dtype, 
                                       ydtype=dtype_data['ydtype'].dtype, 
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
    dataset = dataset.prefetch(buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    if shuffle:
        dataset = dataset.repeat() # Go until fit() stops it

    return dataset


def check_TFRecords(inputdir, TFRfile, datadir, ilog, olog, 
                    batch_size, train_batches, valid_batches, test_batches):
    """
    Checks if TFRecords files exist.
    If they don't, creates the TFRecords files.
    If they do, checks timestamps to know if they need to be re-created.
    
    Inputs
    ------
    inputdir: string. Path/to/directory of inputs.
    TFRfile: string.  Prefix for TFRecords filenames.
    datadir: string.  Path/to/directory of data.
    batch_size: int.  Size of batches for training/validating/testing.
    train_batches: int.  Number of training batches.
    valid_batches: int.  Number of validation batches.
    test_batches:  int.  Number of test batches.
    
    Outputs
    -------
    ftrain_TFR: list of strings.  TFRecords training   files.
    fvalid_TFR: list of strings.  TFRecords validation files.
    ftest_TFR:  list of strings.  TFRecords test       files.
    """
    logger.info('Loading TFRecords file names...')
    TFRpath = os.path.join(inputdir, 'TFRecords', TFRfile)
    ftrain_TFR = glob.glob(TFRpath + 'train*.tfrecords')
    fvalid_TFR = glob.glob(TFRpath + 'valid*.tfrecords')
    ftest_TFR  = glob.glob(TFRpath +  'test*.tfrecords')

    ftrain_NPZ = glob.glob(datadir + 'train' + os.sep + '*.npz')
    fvalid_NPZ = glob.glob(datadir + 'valid' + os.sep + '*.npz')
    ftest_NPZ  = glob.glob(datadir + 'test'  + os.sep + '*.npz')
    
    # File for dtypes
    fdtype = os.path.join(inputdir, 'TFRecords', 'dtype.npz')
    if os.path.exists(fdtype):
        last_dtype = np.load(fdtype)
        last_xdtype = last_dtype['xdtype'].dtype
        last_ydtype = last_dtype['ydtype'].dtype
    else:
        last_xdtype = None
        last_ydtype = None
    
    # Look for the logarithm info file
    flog = os.path.join(inputdir, 'TFRecords', 'log.npz')
    if not os.path.exists(flog):
        # We don't know which indices were log-scaled, if any
        # Force the TFRecords to be made this time
        ftrain_TFR = []
        fvalid_TFR = []
        ftest_TFR  = []
        same_logarithm = False
    else:
        # Check if the current run matches what was used last time
        last_log = np.load(flog)
        last_ilog = last_log['ilog']
        last_olog = last_log['olog']
        same_logarithm = np.all(last_ilog == ilog) and np.all(last_olog == olog)
        if not same_logarithm:
            logger.info("New logarithm pre-processing found.  Re-creating TFRecords.")
    
    # Modification time of this file
    last_changed = os.path.getmtime(__file__)
    
    dset_name = ['train', 'valid', 'test']
    
    if len(ftrain_TFR) == 0:
        logger.info("Making TFRecords for training data...")
        make_train = True
    else:
        # Check most recent modification times
        time_NPZ = os.path.getmtime(max(ftrain_NPZ, key=os.path.getmtime))
        time_TFR = os.path.getmtime(max(ftrain_TFR, key=os.path.getmtime))
                
        if time_NPZ > time_TFR or last_changed > time_TFR:
            if time_NPZ > time_TFR:
                logger.info("Most recent training NPZ file is newer than training TFRecords.")
            elif last_changed > time_TFR:
                logger.info("Most recent change to utils.py is newer than training TFRecords.")
            logger.info("Re-creating TFRecords for training data...")
            # Delete all existing TFRecords
            for foo in ftrain_TFR:
                try:
                    os.remove(foo)
                except Exception as e:
                    logger.error("Error deleting " + foo + ":\n" + str(e))
            make_train = True
        else:
            make_train = False
    if make_train or not same_logarithm:
        xdtype_train, ydtype_train = make_TFRecord(TFRpath + 'train.tfrecords',
                                                   ftrain_NPZ, ilog, olog, 
                                                   batch_size, train_batches)
        ftrain_TFR = glob.glob(TFRpath + 'train*.tfrecords')
    else:
        xdtype_train = None
        ydtype_train = None

    if len(fvalid_TFR) == 0:
        logger.info("Making TFRecords for validation data...")
        make_valid = True
    else:
        # Check most recent modification times
        time_NPZ = os.path.getmtime(max(fvalid_NPZ, key=os.path.getmtime))
        time_TFR = os.path.getmtime(max(fvalid_TFR, key=os.path.getmtime))
        if time_NPZ > time_TFR or last_changed > time_TFR:
            if time_NPZ > time_TFR:
                logger.info("Most recent validation NPZ file is newer than validation TFRecords.")
            elif last_changed > time_TFR:
                logger.info("Most recent change to utils.py is newer than validation TFRecords.")
            logger.info("Re-creating TFRecords for validation data...")
            # Delete all existing TFRecords
            for foo in fvalid_TFR:
                try:
                    os.remove(foo)
                except Exception as e:
                    logger.error("Error deleting " + foo + ":\n" + str(e))
            make_valid = True
        else:
            make_valid = False
    if make_valid or not same_logarithm:
        xdtype_valid, ydtype_valid = make_TFRecord(TFRpath + 'valid.tfrecords',
                                                   fvalid_NPZ, ilog, olog, 
                                                   batch_size, valid_batches)
        fvalid_TFR = glob.glob(TFRpath + 'valid*.tfrecords')
    else:
        xdtype_valid = None
        ydtype_valid = None

    if len(ftest_TFR) == 0:
        logger.info("Making TFRecords for test data...")
        make_test = True
    else:
        # Check most recent modification times
        time_NPZ = os.path.getmtime(max(ftest_NPZ, key=os.path.getmtime))
        time_TFR = os.path.getmtime(max(ftest_TFR, key=os.path.getmtime))
        if time_NPZ > time_TFR or last_changed > time_TFR:
            if time_NPZ > time_TFR:
                logger.info("Most recent test NPZ file is newer than test TFRecords.")
            elif last_changed > time_TFR:
                logger.info("Most recent change to utils.py is newer than test TFRecords.")
            logger.info("Re-creating TFRecords for test data...")
            # Delete all existing TFRecords
            for foo in ftest_TFR:
                try:
                    os.remove(foo)
                except Exception as e:
                    logger.error("Error deleting " + foo + ":\n" + str(e))
            make_test = True
        else:
            make_test = False
    if make_test or not same_logarithm:
        xdtype_test, ydtype_test = make_TFRecord(TFRpath + 'test.tfrecords',
                                                 ftest_NPZ, ilog, olog, 
                                                 batch_size, test_batches)
        ftest_TFR  = glob.glob(TFRpath +  'test*.tfrecords')
    else:
        xdtype_test = None
        ydtype_test = None
    
    # Ensure dtypes all match
    if (make_train and make_valid) or not same_logarithm:
        assert xdtype_train == xdtype_valid, "X training set dtype is "+\
                                             str(xdtype_train)+\
                                             ", but X validation set dtype "+\
                                             "is "+str(xdtype_valid)
        assert ydtype_train == ydtype_valid, "Y training set dtype is "+\
                                             str(ydtype_train)+\
                                             ", but Y validation set dtype "+\
                                             "is "+str(ydtype_valid)
    if (make_train and make_test) or not same_logarithm:
        assert xdtype_train == xdtype_test,  "X training set dtype is "+\
                                             str(xdtype_train)+\
                                             ", but X test set dtype is "+\
                                             str(xdtype_test)
        assert ydtype_train == ydtype_test,  "Y training set dtype is "+\
                                             str(ydtype_train)+\
                                             ", but Y test set dtype is "+\
                                             str(ydtype_test)
    if (make_valid and make_test) or not same_logarithm:
        assert xdtype_valid == xdtype_test,  "X validation set dtype is "+\
                                             str(xdtype_valid)+\
                                             ", but X test set dtype is "+\
                                             str(xdtype_test)
        assert ydtype_valid == ydtype_test,  "Y validation set dtype is "+\
                                             str(ydtype_valid)+\
                                             ", but Y test set dtype is "+\
                                             str(ydtype_test)

    # Compare vs. old dtypes, if applicable
    if xdtype_train is not None:
        xdtype = xdtype_train
    elif xdtype_valid is not None:
        xdtype = xdtype_valid
    elif xdtype_test is not None:
        xdtype = xdtype_test
    else:
        xdtype = None
    if ydtype_train is not None:
        ydtype = ydtype_train
    elif ydtype_valid is not None:
        ydtype = ydtype_valid
    elif ydtype_test is not None:
        ydtype = ydtype_test
    else:
        ydtype = None
    
    if not (make_train and make_valid and make_test) and same_logarithm:
        # Not all TFRecords were created this time
        # Check that dtypes are still consistent with the previous processing
        if last_xdtype is not None and xdtype is not None:
            assert last_xdtype == xdtype, "X data set dtype from the current "+\
                                          "TFRecords processing does not "    +\
                                          "match the dtype from the previous "+\
                                          "processing."
        if last_ydtype is not None and ydtype is not None:
            assert last_ydtype == ydtype, "Y data set dtype from the current "+\
                                          "TFRecords processing does not "    +\
                                          "match the dtype from the previous "+\
                                          "processing."


    if make_train or make_valid or make_test or not same_logarithm:
        # Clear out the stats files so they can be (re-)created as well
        fdel = glob.glob(os.path.join(inputdir, 'x*.npy')) + \
               glob.glob(os.path.join(inputdir, 'y*.npy'))
        for foo in fdel:
            try:
                os.remove(foo)
            except Exception as e:
                logger.error("Error deleting " + foo + ":\n" + str(e))
        # Save new ilog and olog files
        np.savez(flog.replace('.npz', ''), ilog=ilog, olog=olog)
        # Save dtype info so that it can be used when loading TFRecords
        np.savez(fdtype.replace('.npz', ''), xdtype=np.array([], dtype=xdtype), 
                                             ydtype=np.array([], dtype=ydtype))
        logger.info("TFRecords creation complete.")
    else:
        logger.info("TFRecords already exist.")

    return ftrain_TFR, fvalid_TFR, ftest_TFR


def write_ordered_optuna_summary(fout, trials, optlayer, 
                                 optnodes=None, nprint=-1, width=9):
    """
    Writes results of Optuna Bayesian optimization run to human-readable text file.
    
    Inputs
    ------
    fout: str.  Output file to save the results.
    trials: list of trial objects.  Trials considered in the optimization run.
    optlayer: list of str.  Layer types that were considered in the optimization.
    optnodes: list of int.  Number of nodes per layer, if it was not varied in the optimization.
                            If the number of nodes per layer was optimized, set this variable to None (default).
    nprint: int.  Number of trials to write to file.  If -1, it writes all trials (default).
    width: int.  Width of each cell in the output file.
    """
    if nprint == -1:
        nprint = len(trials)
    minvalloss = np.zeros(len(trials))
    maxlayers = len(optlayer)
    for i,trial in enumerate(trials):
        if trial.values is not None:
            minvalloss[i] = trial.values[0]
        else:
            minvallloss[i] = np.inf # Model was not trained
    iorder = np.argsort(minvalloss)
    nrank = len(str(nprint))
    hdr = "|".join(["#".center(nrank)] + \
                   [("Layer "+str(i+1)).center(width) for i in range(maxlayers)] + \
                   ["Min. val. loss"])
    div = "+".join(['-'*nrank] + \
                   ['-'*width for _ in range(maxlayers)] + \
                   ['-'*14])
    with open(fout, "w") as foo:
        foo.write(hdr + "\n")
        foo.write(div + "\n")
    for rank,i in enumerate(iorder[:nprint][::-1]):
        trialrank = str(nprint-rank).rjust(nrank)
        layers = []
        nodes  = []
        acts   = []
        actpar = []
        for j in range(maxlayers):
            if optnodes is not None:
                layers.append(optlayer[j])
                if optlayer[j] not in ['flatten']:
                    nodes.append(optnodes[j])
                else:
                    nodes.append("None")
            elif "nodes_"+str(j+1) in trials[i].params.keys():
                layers.append(optlayer[j])
                nodes.append(trials[i].params["nodes_"+str(j+1)])
            else:
                layers.append(" "*width)
                nodes.append(" "*width)
            if "activation_"+str(j+1) in trials[i].params.keys():
                acts.append(trials[i].params["activation_"+str(j+1)])
            else:
                acts.append(" "*width)
            if "act_val_"+str(j+1) in trials[i].params.keys():
                actpar.append("{:.{}f}".format(trials[i].params["act_val_"+str(j+1)], width-4))
            else:
                actpar.append(" "*width)
        line1 = "|".join([trialrank] + \
                         [lay.center(width) for lay in layers] + \
                         ["{:.6e}".format(trials[i].values[0])])
        line2 = "|".join([" "*nrank] + \
                         [str(n).center(width) for n in nodes] + \
                         [" "])
        line3 = "|".join([" "*nrank] + \
                         [act.center(width) for act in acts] + \
                         [" "])
        line4 = "|".join([" "*nrank] + \
                         [aval.center(width) for aval in actpar] + \
                         [" "])
        with open(fout, "a") as foo:
            foo.write(line1 + "\n")
            foo.write(line2 + "\n")
            foo.write(line3 + "\n")
            foo.write(line4 + "\n")
            foo.write(div   + "\n")
    return

