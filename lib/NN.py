"""
Contains classes/functions related to NN models.

NNModel: class that builds out a specified NN.

driver: function that handles data & model initialization, and 
        trains/validates/tests the model.

"""

import sys, os, platform
import warnings
import time
import random
from io import StringIO
import glob
import pickle

import numpy as np
import matplotlib as mpl
if platform.system() == 'Darwin':
    # Mac fix: use a different backend
    mpl.use("TkAgg")
import matplotlib.pyplot as plt

from sklearn import metrics

# Keras
import keras
from keras import backend as K
from keras.models  import Sequential, Model
from keras.metrics import binary_accuracy
from keras.layers  import (Input, Convolution1D, Dense, Reshape,
                           MaxPooling1D, AveragePooling1D, Dropout, Flatten,  
                           Lambda, Wrapper, merge, concatenate)
from keras.engine  import InputSpec
#from keras.layers.core  import Dense, Dropout, Activation, Layer, Lambda, Flatten
from keras.regularizers import l2
from keras.optimizers   import RMSprop, Adadelta, adam
from keras import initializers
import tensorflow as tf
from tensorflow.python import debug as tf_debug

import onnx
import keras2onnx
from   onnx2keras import onnx_to_keras

import callbacks as C
import loader    as L
import utils     as U
import plotter   as P
import stats     as S

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


class NNModel:
    """
    Builds/trains NN model according to specified parameters.

    __init__: Initialization of the NN model.

    train: Trains the NN model.
    """
    
    def __init__(self, ftrain_TFR, fvalid_TFR, ftest_TFR, 
                 xlen, ylen, olog, 
                 x_mean, x_std, y_mean, y_std, 
                 x_min,  x_max, y_min,  y_max, scalelims, 
                 ncores, buffer_size, batch_size, nbatches, 
                 layers, lay_params, activations, act_params, nodes, 
                 lengthscale = 1e-3, max_lr=1e-1, 
                 clr_mode='triangular', clr_steps=2000, 
                 weight_file = 'weights.h5', stop_file = './STOP', 
                 train_flag = True, 
                 epsilon=1e-6, 
                 debug=False, shuffle=False, resume=False):
        """
        ftrain_TFR : list, strings. TFRecords for the training   data.
        fvalid_TFR : list, strings. TFRecords for the validation data.
        ftest_TFR  : list, strings. TFRecords for the test       data.
        xlen       : int.   Dimensionality of the inputs.
        ylen       : int.   Dimensionality of the outputs.
        olog       : bool.  Determines if the target values are log10-scaled.
        x_mean     : array. Mean  values of the input  data.
        x_std      : array. Stdev values of the input  data.
        y_mean     : array. Mean  values of the output data.
        y_std      : array. Stdev values of the output data.
        x_min      : array. Minima of the input  data.
        x_max      : array. Maxima of the input  data.
        y_min      : array. Minima of the output data.
        y_max      : array. Maxima of the output data.
        scalelims  : list, floats. [min, max] of the scaled data range.
        ncores     : int. Number of cores to use for parallel data loading.
        buffer_size: int. Number of cases to pre-load in memory.
        batch_size : int. Size of batches for training/validation/testing.
        nbatches   : list, ints. Number of batches in the 
                                 [training, validation, test] (in that order) 
                                 sets.
        layers     : list, str.  Types of hidden layers.
        lay_params : list, ints. Parameters for the layer type 
                                 E.g., kernel size
        activations: list, str.  Activation functions for each hidden layer.
        act_params : list, floats. Parameters for the activation functions.
        nodes      : list, ints. For the layers with nodes, 
                                 number of nodes per layer.
        lengthscale: float. Minimum learning rate.
        max_lr     : float. Maximum learning rate.
        clr_mode   : string. Cyclical learning rate function.
        clr_steps  : int. Number of steps per cycle of learning rate.
        weight_file: string. Path/to/file to save the NN weights.
        stop_file  : string. Path/to/file to check for manual stopping of 
                             training.
        train_flag : bool.   Determines whether to train a model or not.
        epsilon    : float. Added to log() arguments to prevent log(0)
        debug      : bool.  If True, turns on Tensorflow's debugger.
        shuffle    : bool.  Determines whether to shuffle the data.
        resume     : bool.  Determines whether to resume training a model.
        """
        # Make sure everything is on the same graph
        if not debug and K.backend() == 'tensorflow':
            K.clear_session()
        else:
            sess = K.get_session()
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
            K.set_session(sess)

        # Load data
        self.X,    self.Y    = U.load_TFdataset(ftrain_TFR, ncores, batch_size, 
                                                buffer_size, xlen, ylen, 
                                                x_mean, x_std, y_mean, y_std,
                                                x_min,  x_max, y_min,  y_max, 
                                                scalelims, shuffle)
        self.Xval, self.Yval = U.load_TFdataset(fvalid_TFR, ncores, batch_size, 
                                                buffer_size, xlen, ylen, 
                                                x_mean, x_std, y_mean, y_std,
                                                x_min,  x_max, y_min,  y_max, 
                                                scalelims, shuffle)
        self.Xte,  self.Yte  = U.load_TFdataset(ftest_TFR,  ncores, batch_size, 
                                                buffer_size, xlen, ylen, 
                                                x_mean, x_std, y_mean, y_std,
                                                x_min,  x_max, y_min,  y_max, 
                                                scalelims, shuffle)
        # Other variables
        self.inD  = xlen
        self.outD = ylen
        self.olog = olog

        self.y_mean = y_mean
        self.y_std  = y_std
        self.y_min  = y_min
        self.y_max  = y_max
        self.scalelims = scalelims

        self.batch_size    = batch_size
        self.train_batches = nbatches[0]
        self.valid_batches = nbatches[1]
        self.test_batches  = nbatches[2]

        self.weight_file = weight_file
        self.stop_file   = stop_file

        self.lengthscale = lengthscale
        self.max_lr      = max_lr
        self.clr_mode    = clr_mode
        self.clr_steps   = clr_steps

        self.epsilon    = epsilon # To ensure log(0) never happens

        self.train_flag = train_flag
        self.resume     = resume
        self.shuffle    = shuffle
        
        ### Build model
        # Input layer
        if shuffle:
            inp = Input(shape=(xlen,), tensor=self.X)
        else:
            inp = Input(shape=(xlen,))
        x = inp
        # Hidden layers
        n = 0 # Counter for layers with nodes
        for i in range(len(layers)):
            if layers[i] == 'conv1d':
                tshape = tuple(val for val in K.int_shape(x) if val is not None)
                if i == 0 or (i > 0 and layers[i-1] != 'conv1d'):
                    # Add channel for convolution
                    x = Reshape(tshape + (1,))(x)
                if type(activations[n]) == str:
                    # Simple activation: pass as layer parameter
                    x = Convolution1D(nb_filter=nodes[n], 
                                      kernel_size=lay_params[i], 
                                      activation=activations[n], 
                                      padding='same')(x)
                else:
                    # Advanced activation: use as its own layer
                    x = Convolution1D(nb_filter=nodes[n], 
                                      kernel_size=lay_params[i], 
                                      padding='same')(x)
                    x = activations[n](x)
                n += 1
            elif layers[i] == 'dense':
                if i > 0:
                    if layers[i-1] == 'conv1d':
                        print('WARNING: Dense layer follows Conv1d layer. ' \
                              + 'Flattening.')
                        x = Flatten()(x)
                if type(activations[n]) == str:
                    x = Dense(nodes[n], activation=activations[n])(x)
                else:
                    x = Dense(nodes[n])(x)
                    x = activations[n] (x)
                n += 1
            elif layers[i] == 'maxpool1d':
                if layers[i-1] == 'dense' or layers[i-1] == 'flatten':
                    raise Exception('MaxPool layers must follow Conv1d or ' \
                                    + 'Pool layer.')
                x = MaxPooling1D(pool_size=lay_params[i])(x)
            elif layers[i] == 'avgpool1d':
                if layers[i-1] == 'dense' or layers[i-1] == 'flatten':
                    raise Exception('AvgPool layers must follow Conv1d or ' \
                                    + 'Pool layer.')
                x = AveragePooling1D(pool_size=lay_params[i])(x)
            elif layers[i] == 'dropout':
                if self.train_flag:
                    x = Dropout(lay_params[i])(x)
            elif layers[i] == 'flatten':
                x = Flatten()(x)
        # Output layer
        out = Dense(ylen)(x)

        self.model = Model(inp, out)
            
        # Compile model
        if shuffle:
            self.model.compile(optimizer=adam(lr=self.lengthscale, amsgrad=True), 
                               loss=keras.losses.mean_squared_error, 
                               target_tensors=[self.Y])
        else:
            self.model.compile(optimizer=adam(lr=self.lengthscale, amsgrad=True), 
                               loss=keras.losses.mean_squared_error)
        print(self.model.summary())

    
    def train(self, train_steps, valid_steps, 
              epochs=100, patience=50):
        """
        Trains model.

        Inputs
        ------
        train_steps: Number of steps before the dataset repeats.
        valid_steps: '   '   '   '   '   '   '   '   '   '   '
        epochs     : Maximum number of iterations through the dataset to train.
        patience   : If no model improvement after `patience` epochs, 
                     ends training.
        """
        # Directory containing the save files
        savedir = os.sep.join(self.weight_file.split(os.sep)[:-1]) + os.sep
        fhistory = savedir + 'history.npz'

        # Resume properly, if requested
        if self.resume:
            if self.weight_file[-5:] == '.onnx':
                #onnx_model = onnx.load(self.weight_file)
                #self.model = onnx_to_keras(onnx_model, ['input_1'], 
                #                           name_policy='short')
                #self.model.compile(optimizer=adam(lr=self.lengthscale, 
                #                                  amsgrad=True), 
                #                   loss=keras.losses.mean_squared_error, 
                #                   target_tensors=[self.Y])
                #self.weight_file = self.weight_file.replace('.onnx', '.h5')
                raise Exception('Resuming training for .onnx models is not '  \
                                + 'yet available. Please specify a\n.h5 file.')
            else:
                self.model.load_weights(self.weight_file)
            try:
                init_epoch = len(np.load(fhistory)['loss'])
            except:
                warning.warn("Resume specified, but history file not found.\n" \
                           + "Training a new model.")
                init_epoch = 0
        # Train a new model
        else:
            init_epoch = 0

        ### Callbacks
        # Save the model weights
        model_checkpoint = keras.callbacks.ModelCheckpoint(self.weight_file,
                                        monitor='val_loss',
                                        save_best_only=True,
                                        save_weights_only=False,
                                        mode='auto',
                                        verbose=1)
        # Handle Ctrl+C or STOP file to halt training
        sig        = C.SignalStopping(stop_file=self.stop_file)
        # Cyclical learning rate
        clr        = C.CyclicLR(base_lr=self.lengthscale, max_lr=self.max_lr,
                                step_size=self.clr_steps, mode=self.clr_mode)
        # Early stopping criteria
        Early_Stop = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                   min_delta=0, 
                                                   patience=patience, 
                                                   verbose=1, mode='auto')
        # Stop if NaN loss occurs
        Nan_Stop = keras.callbacks.TerminateOnNaN()

        ### Train the model
        if self.train_flag:
            # Ensure at least 1 epoch happens when training
            if init_epoch >= epochs:
                print('The requested number of training epochs ('+str(epochs) +\
                      ') is less than or equal to the\nepochs that the model '+\
                      'has already been trained for ('+str(init_epoch)+').  ' +\
                      'The model has\nbeen loaded, but not trained further.')
                self.historyNN.history = np.load(fhistory)
                return
            # Batch size is commented out because that is handled by TFRecords
            self.historyNN = self.model.fit(initial_epoch=init_epoch, 
                                             epochs=epochs, 
                                             steps_per_epoch=train_steps, 
                                             #batch_size=self.batch_size, 
                                             verbose=2, 
                                             validation_data=(self.Xval, 
                                                              self.Yval), 
                                             validation_steps=valid_steps, 
                                             callbacks=[clr, sig, Early_Stop, 
                                                        Nan_Stop, 
                                                        model_checkpoint])
            # Save out the history
            self.historyCLR = clr.history
            if not os.path.exists(fhistory) or not self.resume:
                np.savez(fhistory, loss=self.historyNN.history['loss'], 
                               val_loss=self.historyNN.history['val_loss'])
            else:
                history  = np.load(fhistory)
                loss     = np.concatenate((history['loss'], 
                            self.historyNN.history['loss']))
                val_loss = np.concatenate((history['val_loss'], 
                            self.historyNN.history['val_loss']))
                np.savez(fhistory, loss=loss, val_loss=val_loss)

        # Load best set of weights
        self.model.load_weights(self.weight_file)

    def Yeval(self, mode, dataset, preddir, denorm=False):
        """
        Saves out .NPY files of the true or predicted Y values for a 
        specified data set.

        Inputs
        ------
        mode   : string. 'pred' or 'true'. Specifies whether to save the true 
                         or predicted Y values of  `dataset`.
        dataset: string. 'train', 'valid', or 'test'. Specifies the data set to 
                         make predictions on.
        preddir: string. Path/to/directory where predictions will be saved.
        denorm : bool.   Determines whether to denormalize the predicted values.
        """
        if self.shuffle:
            raise ValueError("This model has shuffled TFRecords.\nCreate a " +\
                        "new NNModel object with shuffle=False and try again.")

        if dataset == 'train':
            X = self.X
            Y = self.Y
            num_batches = self.train_batches
        elif dataset == 'valid':
            X = self.Xval
            Y = self.Yval
            num_batches = self.valid_batches
        elif dataset == 'test':
            X = self.Xte
            Y = self.Yte
            num_batches = self.test_batches
        else:
            raise ValueError("Invalid specification for `dataset` parameter " +\
                    "of NNModel.Yeval().\nAllowed options: 'train', 'valid'," +\
                    " or 'test'\nPlease correct this and try again.")

        # Prefix for the savefiles
        if mode == 'pred' or mode == 'true':
            fname = ''.join([preddir, dataset, os.sep, mode])
        else:
            raise ValueError("Invalid specification for `mode` parameter of " +\
                             "NNModel.Yeval().\nAllowed options: 'pred' or "  +\
                             "'true'\nPlease correct this and try again.")

        if denorm:
            fname = ''.join([fname, '-denorm_'])
        else:
            fname = ''.join([fname, '-norm_'])

        U.make_dir(preddir+dataset) # Ensure the directory exists

        # Save out the Y values
        y_batch = np.zeros((self.batch_size, self.outD))
        for i in range(num_batches):
            foo = ''.join([fname, str(i).zfill(len(str(num_batches))), '.npy'])
            if mode == 'pred': # Predicted Y values
                x_batch = K.eval(X)
                y_batch = self.model.predict(x_batch)
            else:  # True Y values
                y_batch = K.eval(Y)
            if denorm:
                y_batch = U.denormalize(U.descale(y_batch, 
                                                  self.y_min, self.y_max, 
                                                  self.scalelims),
                                        self.y_mean, self.y_std)
                if self.olog:
                    y_batch[:, self.olog] = 10**y_batch[:, self.olog]
            np.save(foo, y_batch)
            print(''.join(['  Batch ', str(i+1), '/', str(num_batches)]), end='\r')
        print('')

        return fname



def driver(inputdir, outputdir, datadir, plotdir, preddir, 
           trainflag, validflag, testflag, 
           normalize, fmean, fstdev, 
           scale, fmin, fmax, scalelims, 
           fsize, rmse_file, r2_file, 
           inD, outD, ilog, olog, 
           TFRfile, batch_size, ncores, buffer_size, 
           gridsearch, architectures, 
           layers, lay_params, activations, act_params, nodes, 
           lengthscale, max_lr, clr_mode, clr_steps, 
           epochs, patience, 
           weight_file, resume, 
           plot_cases, fxvals, xlabel, ylabel,
           filters=None, filt2um=1.):
    """
    Driver function to handle model training and evaluation.

    Inputs
    ------
    inputdir   : string. Path/to/directory of inputs.
    outputdir  : string. Path/to/directory of outputs.
    datadir    : string. Path/to/directory of data.
    plotdir    : string. Path/to/directory of plots.
    preddir    : string. Path/to/directory of predictions.
    trainflag  : bool.   Determines whether to train    the NN model.
    validflag  : bool.   Determines whether to validate the NN model.
    testflag   : bool.   Determines whether to test     the NN model.
    normalize  : bool.   Determines whether to normalize the data.
    fmean      : string. Path/to/file of mean  of training data.
    fstdev     : string. Path/to/file of stdev of training data.
    scale      : bool.   Determines whether to scale the data.
    fmin       : string. Path/to/file of minima of training data.
    fmax       : string. Path/to/file of maxima of training data.
    scalelims  : list, floats. [min, max] of range of scaled data.
    rmse_file  : string. Prefix for savefiles for RMSE calculations.
    r2_file    : string. Prefix for savefiles for R2 calculations.
    inD        : int.    Dimensionality of the input  data.
    outD       : int.    Dimensionality of the output data.
    ilog       : bool.   Determines whether to take the log10 of intput  data.
    olog       : bool.   Determines whether to take the log10 of output data.
    TFRfile    : string. Prefix for TFRecords files.
    batch_size : int.    Size of batches for training/validating/testing.
    ncores     : int.    Number of cores to use to load data cases.
    buffer_size: int.    Number of data cases to pre-load in memory.
    gridsearch : bool.   Determines whether to perform a grid search over 
                         `architectures`.
    architectures: list. Model architectures to consider.
    layers     : list, str.  Types of hidden layers.
    lay_params : list, ints. Parameters for the layer type 
                             E.g., kernel size
    activations: list, str.  Activation functions for each hidden layer.
    act_params : list, floats. Parameters for the activation functions.
    nodes      : list, ints. For layers with nodes, number of nodes per layer.
    lengthscale: float.  Minimum learning rat.e
    max_lr     : float.  Maximum learning rate.
    clr_mode   : string. Sets the cyclical learning rate function.
    clr_steps  : int.    Number of steps per cycle of the learning rate.
    epochs     : int.    Max number of iterations through dataset for training.
    patience   : int.    If no model improvement after `patience` epochs, 
                         halts training.
    weight_file: string. Path/to/file where NN weights are saved.
    resume     : bool.   Determines whether to resume training.
    plot_cases : list, ints. Cases from test set to plot.
    fxvals     : string. Path/to/file of X-axis values to correspond to 
                         predicted Y values.
    xlabel     : string. X-axis label for plotting.
    ylabel     : string. Y-axis label for plotting.
    filters    : list, strings.  Paths/to/filter files.  Default: None
                         If specified, will compute RMSE/R2 stats over the 
                         integrated filter bandpasses.
    filt2um    : float.  Conversion factor for filter file wavelengths to 
                         microns.  Default: 1.0
    """
    # Get file names, calculate number of cases per file
    print('Loading files & calculating total number of batches...')

    try:
        datsize   = np.load(inputdir + fsize)
        num_train = datsize[0]
        num_valid = datsize[1]
        num_test  = datsize[2]
    except:
        ftrain = glob.glob(datadir + 'train' + os.sep + '*.npy')
        fvalid = glob.glob(datadir + 'valid' + os.sep + '*.npy')
        ftest  = glob.glob(datadir + 'test'  + os.sep + '*.npy')
        num_train = U.data_set_size(ftrain, ncores)
        num_valid = U.data_set_size(fvalid, ncores)
        num_test  = U.data_set_size(ftest,  ncores)
        np.save(inputdir + fsize, np.array([num_train, num_valid, num_test], dtype=int))
        del ftrain, fvalid, ftest

    print("Data set sizes")
    print("Training   data:", num_train)
    print("Validation data:", num_valid)
    print("Testing    data:", num_test)
    print("Total      data:", num_train + num_valid + num_test)

    train_batches = num_train // batch_size
    valid_batches = num_valid // batch_size
    test_batches  = num_test  // batch_size

    # Update `clr_steps`
    if clr_steps == "range test":
        clr_steps = train_batches * epochs
        rng_test  = True
    else:
        clr_steps = train_batches * int(clr_steps)
        rng_test  = False

    # Get mean/stdev for normalizing
    if normalize:
        print('\nNormalizing the data...')
        try:
            mean   = np.load(inputdir + fmean)
            stdev  = np.load(inputdir + fstdev)
        except:
            print("Calculating the mean and standard deviation of the data " +\
                  "using Welford's method.")
            # Compute stats
            ftrain = glob.glob(datadir + 'train' + os.sep + '*.npy')
            mean, stdev, datmin, datmax = S.mean_stdev(ftrain, inD, ilog, olog)
            np.save(inputdir + fmean,  mean)
            np.save(inputdir + fstdev, stdev)
            np.save(inputdir + fmin,   datmin)
            np.save(inputdir + fmax,   datmax)
            del datmin, datmax, ftrain
        print("mean :", mean)
        print("stdev:", stdev)
        # Slice desired indices
        x_mean, y_mean = mean [:inD], mean [inD:]
        x_std,  y_std  = stdev[:inD], stdev[inD:]
        # Memory cleanup -- no longer need full mean/stdev arrays
        del mean, stdev
    else:
        x_mean = 0.
        x_std  = 1.
        y_mean = 0.
        y_std  = 1.

    if olog:
        # To properly calculate RMSE & R2 for log-scaled output
        try:
            y_mean_delog = np.load(inputdir + 
                                   fmean.replace(".npy", "_delog.npy"))
        except:
            mean_delog = S.mean_stdev(ftrain, inD, ilog, False)[0]
            y_mean_delog = mean_delog[inD:]
            np.save(inputdir + fmean.replace(".npy", "_delog.npy"), y_mean_delog)
    else:
        y_mean_delog = y_mean

    # Get min/max values for scaling
    if scale:
        print('\nScaling the data...')
        try:
            datmin = np.load(inputdir + fmin)
            datmax = np.load(inputdir + fmax)
        except:
            ftrain = glob.glob(datadir + 'train' + os.sep + '*.npy')
            mean, stdev, datmin, datmax = S.mean_stdev(ftrain, inD, ilog, olog)
            np.save(inputdir + fmean,  mean)
            np.save(inputdir + fstdev, stdev)
            np.save(inputdir + fmin,   datmin)
            np.save(inputdir + fmax,   datmax)
            del mean, stdev, ftrain
        print("min  :", datmin)
        print("max  :", datmax)
        # Slice desired indices
        x_min, y_min = datmin[:inD], datmin[inD:]
        x_max, y_max = datmax[:inD], datmax[inD:]
        # Memory cleanup -- no longer need min/max arrays
        del datmin, datmax

        # Normalize min/max values
        if normalize:
            x_min = U.normalize(x_min, x_mean, x_std)
            x_max = U.normalize(x_max, x_mean, x_std)
            y_min = U.normalize(y_min, y_mean, y_std)
            y_max = U.normalize(y_max, y_mean, y_std)
    else:
        x_min     =  0.
        x_max     =  1.
        y_min     =  0.
        y_max     =  1.
        scalelims = [0., 1.]

    # Get TFRecord file names
    print('\nLoading TFRecords file names...')
    TFRpath = inputdir +'TFRecords' + os.sep + TFRfile
    ftrain_TFR = glob.glob(TFRpath + 'train*.tfrecords')
    fvalid_TFR = glob.glob(TFRpath + 'valid*.tfrecords')
    ftest_TFR  = glob.glob(TFRpath +  'test*.tfrecords')

    if len(ftrain_TFR) == 0 or len(fvalid_TFR) == 0 or len(ftest_TFR) == 0:
        # Doesn't exist -- make them
        print("\nSome TFRecords files do not exist yet.")
        ftrain = glob.glob(datadir + 'train' + os.sep + '*.npy')
        fvalid = glob.glob(datadir + 'valid' + os.sep + '*.npy')
        ftest  = glob.glob(datadir + 'test'  + os.sep + '*.npy')
        if len(ftrain_TFR) == 0:
            print("Making TFRecords for training data...")
            U.make_TFRecord(inputdir+'TFRecords'+os.sep+TFRfile+'train.tfrecords', 
                            ftrain, inD, ilog, olog, batch_size, train_batches)
        if len(fvalid_TFR) == 0:
            print("\nMaking TFRecords for validation data...")
            U.make_TFRecord(inputdir+'TFRecords'+os.sep+TFRfile+'valid.tfrecords', 
                            fvalid, inD, ilog, olog, batch_size, valid_batches)
        if len(ftest_TFR) == 0:
            print("\nMaking TFRecords for test data...")
            U.make_TFRecord(inputdir+'TFRecords'+os.sep+TFRfile+'test.tfrecords', 
                            ftest,  inD, ilog, olog, batch_size, test_batches)
        print("\nTFRecords creation complete.")
        # Free memory
        del ftrain, fvalid, ftest
        # Get TFR file names for real this time
        ftrain_TFR = glob.glob(TFRpath + 'train*.tfrecords')
        fvalid_TFR = glob.glob(TFRpath + 'valid*.tfrecords')
        ftest_TFR  = glob.glob(TFRpath +  'test*.tfrecords')

    # Load the xvals
    if fxvals is not None:
        xvals = np.load(fxvals)
    else:
        xvals = None

    # Perform grid search
    if gridsearch:
        # Train a model for each architecture, w/ unique directories
        print("\nPerforming a grid search.\n")
        maxlen = 0
        for i, arch in enumerate(architectures):
            if len(arch) > maxlen:
                maxlen = len(arch)
            archdir = os.path.join(outputdir, arch, '')
            wsplit  = weight_file.rsplit(os.sep, 1)[1].rsplit('.', 1)
            wfile   = ''.join([archdir, wsplit[0], '_', arch, '.', wsplit[1]])
            U.make_dir(archdir)
            nn = NNModel(ftrain_TFR, fvalid_TFR, ftest_TFR, 
                         inD, outD, olog, 
                         x_mean, x_std, y_mean, y_std, 
                         x_min,  x_max, y_min,  y_max, scalelims, 
                         ncores, buffer_size, batch_size, 
                         [train_batches, valid_batches, test_batches], 
                         layers[i], lay_params[i], 
                         activations[i], act_params[i], nodes[i], 
                         lengthscale, max_lr, clr_mode, clr_steps, 
                         wfile, stop_file='./STOP', resume=resume, 
                         train_flag=True, shuffle=True)
            nn.train(train_batches, valid_batches, epochs, patience)
            P.loss(nn, archdir)
        # Print/save out the minmium validation loss for each architecture
        minvl = np.ones(len(architectures)) * np.inf
        print('Grid search summary')
        print('-------------------')
        with open(outputdir + 'gridsearch.txt', 'w') as foo:
            foo.write('Grid search summary\n')
            foo.write('-------------------\n')
        for i, arch in enumerate(architectures):
            archdir  = os.path.join(outputdir, arch, '')
            history  = np.load(archdir+'history.npz')
            minvl[i] = np.amin(history['val_loss'])
            print(arch.ljust(maxlen, ' ') + ': ' + str(minvl[i]))
            with open(outputdir + 'gridsearch.txt', 'a') as foo:
                foo.write(arch.ljust(maxlen, ' ') + ': ' \
                          + str(minvl[i]) + '\n')
        return

    # Train a model
    if trainflag:
        print('\nBeginning model training.\n')
        nn = NNModel(ftrain_TFR, fvalid_TFR, ftest_TFR, 
                     inD, outD, olog, 
                     x_mean, x_std, y_mean, y_std, 
                     x_min,  x_max, y_min,  y_max, scalelims, 
                     ncores, buffer_size, batch_size, 
                     [train_batches, valid_batches, test_batches], 
                     layers, lay_params, activations, act_params, nodes, 
                     lengthscale, max_lr, clr_mode, clr_steps, 
                     weight_file, stop_file='./STOP', 
                     train_flag=True, shuffle=True, resume=resume)
        nn.train(train_batches, valid_batches, epochs, patience)
        # Plot the loss
        P.loss(nn, plotdir)

    # Call new model with shuffle=False
    nn = NNModel(ftrain_TFR, fvalid_TFR, ftest_TFR, 
                 inD, outD, olog, 
                 x_mean, x_std, y_mean, y_std, 
                 x_min,  x_max, y_min,  y_max, scalelims, 
                 ncores, buffer_size, batch_size, 
                 [train_batches, valid_batches, test_batches], 
                 layers, lay_params, activations, act_params, nodes, 
                 lengthscale, max_lr, clr_mode, clr_steps, 
                 weight_file, stop_file='./STOP', 
                 train_flag=False, shuffle=False, resume=False)
    if '.h5' in weight_file or '.hdf5' in weight_file:
        nn.model.load_weights(weight_file) # Load the model
        # Save in ONNX format
        try:
            onnx_model = keras2onnx.convert_keras(nn.model)
            onnx.save_model(onnx_model, nn.weight_file.rsplit('.', 1)[0] + '.onnx')
        except Exception as e:
            print("Unable to convert the Keras model to ONNX:")
            print(e)
    else:
        nn.model = onnx_to_keras(onnx.load_model(weight_file), ['input_1'])

    # Validate model
    if (validflag or trainflag) and not rng_test:
        print('\nValidating the model...\n')
        # Y values
        print('  Predicting...')
        fvalpred = nn.Yeval('pred', 'valid', preddir, 
                            denorm=(normalize==False and scale==False))
        fvalpred = glob.glob(fvalpred + '*')

        print('  Loading the true Y values...')
        fvaltrue = nn.Yeval('true', 'valid', preddir, 
                            denorm=(normalize==False and scale==False))
        fvaltrue = glob.glob(fvaltrue + '*')
        ### RMSE & R2
        print('\n Calculating RMSE & R2...')
        if not normalize and not scale:
            val_stats = S.rmse_r2(fvalpred, fvaltrue, y_mean, 
                                  olog=olog, y_mean_delog=y_mean_delog, 
                                  x_vals=xvals, 
                                  filters=filters, filt2um=filt2um)
        else:
            val_stats = S.rmse_r2(fvalpred, fvaltrue, y_mean, 
                                  y_std, y_min, y_max, scalelims, 
                                  olog, y_mean_delog, 
                                  xvals, filters, filt2um)
        # RMSE
        if np.any(val_stats[0] != -1) and np.any(val_stats[1] != -1):
            print('  Normalized RMSE       : ', val_stats[0])
            print('  Mean normalized RMSE  : ', np.mean(val_stats[0]))
            print('  Denormalized RMSE     : ', val_stats[1])
            print('  Mean denormalized RMSE: ', np.mean(val_stats[1]))
            np.savez(outputdir+rmse_file+'_val_norm.npz', 
                     rmse=val_stats[0], rmse_mean=np.mean(val_stats[0]))
            saveRMSEnorm   = True
            saveRMSEdenorm = True
        elif np.any(val_stats[0] != -1):
            print('  RMSE     : ', val_stats[0])
            print('  Mean RMSE: ', np.mean(val_stats[0]))
            saveRMSEnorm   = True
            saveRMSEdenorm = False
        elif np.any(val_stats[1] != -1):
            print('  RMSE     : ', val_stats[1])
            print('  Mean RMSE: ', np.mean(val_stats[1]))
            saveRMSEnorm   = False
            saveRMSEdenorm = True
        else:
            print("  No files passed in to compute RMSE.")
            saveRMSEnorm   = False
            saveRMSEdenorm = False
        if saveRMSEnorm:
            P.plot(''.join([plotdir, rmse_file, '_val_norm.png']), 
                   xvals, val_stats[0], xlabel, 'RMSE')
            np.savez(outputdir+rmse_file+'_val_norm.npz', 
                     rmse=val_stats[0], rmse_mean=np.mean(val_stats[0]))
        if saveRMSEdenorm:
            P.plot(''.join([plotdir, rmse_file, '_val_denorm.png']), 
                   xvals, val_stats[1], xlabel, 'RMSE')
            np.savez(outputdir+rmse_file+'_val_denorm.npz', 
                     rmse=val_stats[1], rmse_mean=np.mean(val_stats[1]))
        # R2
        if np.any(val_stats[2] != -1) and np.any(val_stats[3] != -1):
            print('  Normalized R2       : ', val_stats[2])
            print('  Mean normalized R2  : ', np.mean(val_stats[2]))
            print('  Denormalized R2     : ', val_stats[3])
            print('  Mean denormalized R2: ', np.mean(val_stats[3]))
            saveR2norm   = True
            saveR2denorm = True
        elif np.any(val_stats[2] != -1):
            print('  R2     : ', val_stats[2])
            print('  Mean R2: ', np.mean(val_stats[2]))
            saveR2norm   = True
            saveR2denorm = False
        elif np.any(val_stats[3] != -1):
            print('  R2     : ', val_stats[3])
            print('  Mean R2: ', np.mean(val_stats[3]))
            saveR2norm   = False
            saveR2denorm = True
        else:
            print("  No files passed in to compute R2.")
            saveR2norm   = False
            saveR2denorm = False
        if saveR2norm:
            P.plot(''.join([plotdir, r2_file, '_val_norm.png']), 
                   xvals, val_stats[2], xlabel, '$R^2$')
            np.savez(outputdir+r2_file+'_val_norm.npz', 
                     r2=val_stats[2], r2_mean=np.mean(val_stats[2]))
        if saveR2denorm:
            P.plot(''.join([plotdir, r2_file, '_val_denorm.png']), 
                   xvals, val_stats[3], xlabel, '$R^2$')
            np.savez(outputdir+r2_file+'_val_denorm.npz', 
                     r2=val_stats[3], r2_mean=np.mean(val_stats[3]))

    # Evaluate model on test set
    if testflag and not rng_test:
        print('\nTesting the model...\n')
        # Y values
        print('  Predicting...')
        ftestpred = nn.Yeval('pred', 'test', preddir, 
                             denorm=(normalize==False and scale==False))
        ftestpred = glob.glob(ftestpred + '*')

        print('  Loading the true Y values...')
        ftesttrue = nn.Yeval('true', 'test', preddir, 
                             denorm=(normalize==False and scale==False))
        ftesttrue = glob.glob(ftesttrue + '*')
        ### RMSE & R2
        print('\n Calculating RMSE & R2...')
        if not normalize and not scale:
            test_stats = S.rmse_r2(ftestpred, ftesttrue, y_mean, 
                                   olog=olog, y_mean_delog=y_mean_delog, 
                                   x_vals=xvals, 
                                   filters=filters, filt2um=filt2um)
        else:
            test_stats = S.rmse_r2(ftestpred, ftesttrue, y_mean, 
                                   y_std, y_min, y_max, scalelims, 
                                   olog, y_mean_delog, 
                                   xvals, filters, filt2um)
        # RMSE
        if np.any(test_stats[0] != -1) and np.any(test_stats[1] != -1):
            print('  Normalized RMSE       : ', test_stats[0])
            print('  Mean normalized RMSE  : ', np.mean(test_stats[0]))
            print('  Denormalized RMSE     : ', test_stats[1])
            print('  Mean denormalized RMSE: ', np.mean(test_stats[1]))
            np.savez(outputdir+rmse_file+'_val_norm.npz', 
                     rmse=test_stats[0], rmse_mean=np.mean(test_stats[0]))
            saveRMSEnorm   = True
            saveRMSEdenorm = True
        elif np.any(test_stats[0] != -1):
            print('  RMSE     : ', test_stats[0])
            print('  Mean RMSE: ', np.mean(test_stats[0]))
            saveRMSEnorm   = True
            saveRMSEdenorm = False
        elif np.any(test_stats[1] != -1):
            print('  RMSE     : ', test_stats[1])
            print('  Mean RMSE: ', np.mean(test_stats[1]))
            saveRMSEnorm   = False
            saveRMSEdenorm = True
        else:
            print("  No files passed in to compute RMSE.")
            saveRMSEnorm   = False
            saveRMSEdenorm = False
        if saveRMSEnorm:
            P.plot(''.join([plotdir, rmse_file, '_test_norm.png']), 
                   xvals, test_stats[0], xlabel, 'RMSE')
            np.savez(outputdir+rmse_file+'_test_norm.npz', 
                     rmse=test_stats[0], rmse_mean=np.mean(test_stats[0]))
        if saveRMSEdenorm:
            P.plot(''.join([plotdir, rmse_file, '_test_denorm.png']), 
                   xvals, test_stats[1], xlabel, 'RMSE')
            np.savez(outputdir+rmse_file+'_test_denorm.npz', 
                     rmse=test_stats[1], rmse_mean=np.mean(test_stats[1]))
        # R2
        if np.any(test_stats[2] != -1) and np.any(test_stats[3] != -1):
            print('  Normalized R2       : ', test_stats[2])
            print('  Mean normalized R2  : ', np.mean(test_stats[2]))
            print('  Denormalized R2     : ', test_stats[3])
            print('  Mean denormalized R2: ', np.mean(test_stats[3]))
            saveR2norm   = True
            saveR2denorm = True
        elif np.any(test_stats[2] != -1):
            print('  R2     : ', test_stats[2])
            print('  Mean R2: ', np.mean(test_stats[2]))
            saveR2norm   = True
            saveR2denorm = False
        elif np.any(test_stats[3] != -1):
            print('  R2     : ', test_stats[3])
            print('  Mean R2: ', np.mean(test_stats[3]))
            saveR2norm   = False
            saveR2denorm = True
        else:
            print("  No files passed in to compute R2.")
            saveR2norm   = False
            saveR2denorm = False
        if saveR2norm:
            P.plot(''.join([plotdir, r2_file, '_test_norm.png']), 
                   xvals, test_stats[2], xlabel, '$R^2$')
            np.savez(outputdir+r2_file+'_test_norm.npz', 
                     r2=test_stats[2], r2_mean=np.mean(test_stats[2]))
        if saveR2denorm:
            P.plot(''.join([plotdir, r2_file, '_test_denorm.png']), 
                   xvals, test_stats[3], xlabel, '$R^2$')
            np.savez(outputdir+r2_file+'_test_denorm.npz', 
                     r2=test_stats[3], r2_mean=np.mean(test_stats[3]))

    # Plot requested cases
    if not rng_test:
        predfoo = sorted(glob.glob(preddir+'test'+os.sep+'pred*'))
        truefoo = sorted(glob.glob(preddir+'test'+os.sep+'true*'))
        if len(predfoo) > 0 and len(truefoo) > 0:
            print("\nPlotting the requested cases...")
            nplot = 0
            for v in plot_cases:
                fname    = plotdir + 'spec' + str(v) + '_pred-vs-true.png'
                predspec = np.load(predfoo[v // batch_size])[v % batch_size]
                predspec = U.denormalize(U.descale(predspec, 
                                                   y_min, y_max, scalelims),
                                         y_mean, y_std)
                truespec = np.load(truefoo[v // batch_size])[v % batch_size]
                truespec = U.denormalize(U.descale(truespec, 
                                                   y_min, y_max, scalelims),
                                         y_mean, y_std)
                if olog:
                    predspec[olog] = 10**predspec[olog]
                    truespec[olog] = 10**truespec[olog]
                P.plot_spec(fname, predspec, truespec, xvals, xlabel, ylabel)
                nplot += 1
                print("  Plot " + str(nplot) + "/" + str(len(plot_cases)), end='\r')
            print("")
        else:
            raise Exception("No predictions found in " + preddir + "test.")

    return


