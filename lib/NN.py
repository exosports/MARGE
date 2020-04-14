"""
Contains classes/functions related to NN models.

NNModel: class that builds out a specified NN.

driver: function that handles data & model initialization, and 
        trains/validates/tests the model.

"""

import sys, os
import time
import random
from io import StringIO
import glob
import pickle

import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics

# Keras
import keras
from keras import backend as K
from keras.models  import Sequential, Model
from keras.metrics import binary_accuracy
from keras.layers  import Convolution1D, Dense, MaxPooling1D, Flatten, Input, Lambda, Wrapper, merge, concatenate
from keras.engine  import InputSpec
from keras.layers.core  import Dense, Dropout, Activation, Layer, Lambda, Flatten
from keras.regularizers import l2
from keras.optimizers   import RMSprop, Adadelta, adam
from keras.layers.advanced_activations import LeakyReLU
from keras import initializers
import tensorflow as tf
from tensorflow.python import debug as tf_debug

import onnx
import keras2onnx
#from   onnx2keras import onnx_to_keras

import GPyOpt

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
                 xlen, ylen, 
                 x_mean, x_std, y_mean, y_std, 
                 x_min,  x_max, y_min,  y_max, scalelims, 
                 ncores, batch_size, buffer_size, 
                 convlayers, denselayers, 
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
        batch_size : int. Size of batches for training/validation/testing.
        buffer_size: int. Number of cases to pre-load in memory.
        convlayers : list, ints. Dimensionality of conv layers.
        denselayers: list, ints. Dimensionality of dense layers.
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
                                                scalelims, shuffle, convlayers)
        self.Xval, self.Yval = U.load_TFdataset(fvalid_TFR, ncores, batch_size, 
                                                buffer_size, xlen, ylen, 
                                                x_mean, x_std, y_mean, y_std,
                                                x_min,  x_max, y_min,  y_max, 
                                                scalelims, shuffle, convlayers)
        self.Xte,  self.Yte  = U.load_TFdataset(ftest_TFR,  ncores, batch_size, 
                                                buffer_size, xlen, ylen, 
                                                x_mean, x_std, y_mean, y_std,
                                                x_min,  x_max, y_min,  y_max, 
                                                scalelims, shuffle, convlayers)
        # Other variables
        self.batch_size  = batch_size
        self.weight_file = weight_file
        self.stop_file   = stop_file
        self.lengthscale = lengthscale
        self.max_lr      = max_lr
        self.clr_mode    = clr_mode
        self.clr_steps   = clr_steps
        # To ensure log(0) never happens
        self.epsilon     = epsilon
        self.train_flag  = train_flag
        self.resume      = resume
        
        # Build model
        if convlayers is not None:
            # Convolutional layers to mix inputs
            if shuffle:
                inp = Input(shape=(xlen,1), tensor=self.X)
            else:
                inp = Input(shape=(xlen,1))
            x = inp
            x = Convolution1D(nb_filter=convlayers[0], kernel_size=5, 
                              activation='relu',      padding='same')(x)
            #x = MaxPooling1D(pool_size=2)(x)
            for jj in range(1, len(convlayers)):
                x = Convolution1D(nb_filter=convlayers[jj], kernel_size=3, 
                                  activation='relu',       padding='same')(x)
                if jj == len(convlayers) - 1:
                    x = MaxPooling1D(pool_size=2)(x)
            x = Flatten()(x)
        else:
            # Dense net
            if shuffle:
                inp = Input(shape=(xlen,), tensor=self.X)
            else:
                inp = Input(shape=(xlen,))
            x = inp

        # Dense layers
        x = Dense(denselayers[0], activation='relu')(x)
        for jj in range(1, len(denselayers)):
            x = Dense(denselayers[jj], activation='relu')(x)
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
        # ModelCheckpoint replaced with ModelCheckpointEnhanced, below
        model_checkpoint = keras.callbacks.ModelCheckpoint(self.weight_file,
                                        monitor='val_loss',
                                        save_best_only=True,
                                        save_weights_only=True,
                                        mode='auto',
                                        verbose=1)

        # Directory containing the save files
        savedir = '/'.join(self.weight_file.split('/')[:-1]) + '/'

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
                print('Resuming training for .onnx models is not yet available.')
                print('Please specify a .h5 file of model weights.')
                sys.exit()
            else:
                self.model.load_weights(self.weight_file)
            sigpickle  = glob.glob(savedir + 'sig.epoch*.pickle')[-1]
            sig        = pickle.load(open(sigpickle, "rb"))
            init_epoch = int(sigpickle.split('.')[-2].strip('epoch'))
        # Train a new model
        else:
            # Handle Ctrl+C or STOP file to halt training
            sig        = C.SignalStopping(stop_file=self.stop_file)
            init_epoch = 0
        sig_savefile     = savedir + 'sig.epoch{epoch:04d}.pickle'
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

        # Super-charged model saving & resuming training
        #super_checkpoint = C.ModelCheckpointEnhanced(filepath=self.weight_file, 
        #                        monitor='val_loss',
        #                        save_best_only=True,
        #                        mode='auto',
        #                        verbose=1, 
        #                        callbacks_to_save =[sig], 
        #                        callbacks_filepath=[sig_savefile])


        # Train the model
        if self.train_flag:
            # Ensure at least 1 epoch happens when training
            if init_epoch >= epochs:
                epochs = init_epoch + 1
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
            self.historyCLR = clr.history
            fhistory = savedir + 'history.npz'
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


        self.model.load_weights(self.weight_file)


def driver(inputdir, outputdir, datadir, plotdir, preddir, 
           trainflag, validflag, testflag, 
           normalize, fmean, fstdev, 
           scale, fmin, fmax, scalelims, 
           fsize, rmse_file, r2_file, 
           inD, outD, ilog, olog, 
           TFRfile, batch_size, ncores, buffer_size, 
           gridsearch, architectures, 
           convlayers, denselayers, 
           lengthscale, max_lr, clr_mode, clr_steps, 
           epochs, patience, 
           weight_file, resume, 
           plot_cases, xvals, xlabel, ylabel):
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
    convlayers : list, ints. Dimensionality of conv  layers.
    denselayers: list, ints. Dimensionality of dense layers.
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
    xvals      : array.  X-axis values to correspond to predicted Y values.
    xlabel     : string. X-axis label for plotting.
    ylabel     : string. Y-axis label for plotting.
    """
    # Get file names, calculate number of cases per file
    print('Loading files & calculating total number of batches...')

    try:
        datsize   = np.load(inputdir + fsize)
        num_train = datsize[0]
        num_valid = datsize[1]
        num_test  = datsize[2]
    except:
        ftrain = glob.glob(datadir + 'train/' + '*.npy')
        fvalid = glob.glob(datadir + 'valid/' + '*.npy')
        ftest  = glob.glob(datadir + 'test/'  + '*.npy')
        num_train = U.data_set_size(ftrain, ncores)
        num_valid = U.data_set_size(fvalid, ncores)
        num_test  = U.data_set_size(ftest,  ncores)
        np.save(inputdir + fsize, np.array([num_train, num_valid, num_test]))
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
            ftrain = glob.glob(datadir + 'train/' + '*.npy')
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
        # Memory cleanup -- no longer need mean/stdev arrays
        del mean, stdev
    else:
        x_mean = 0.
        x_std  = 1.
        y_mean = 0.
        y_std  = 1.

    # Get min/max values for scaling
    if scale:
        print('\nScaling the data...')
        try:
            datmin = np.load(inputdir + fmin)
            datmax = np.load(inputdir + fmax)
        except:
            ftrain = glob.glob(datadir + 'train/' + '*.npy')
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
    TFRpath = inputdir +'TFRecords/' + TFRfile
    ftrain_TFR = glob.glob(TFRpath + 'train*.tfrecords')
    fvalid_TFR = glob.glob(TFRpath + 'valid*.tfrecords')
    ftest_TFR  = glob.glob(TFRpath +  'test*.tfrecords')

    if len(ftrain_TFR) == 0 or len(fvalid_TFR) == 0 or len(ftest_TFR) == 0:
        # Doesn't exist -- make them
        print("\nSome TFRecords files do not exist yet.")
        ftrain = glob.glob(datadir + 'train/' + '*.npy')
        fvalid = glob.glob(datadir + 'valid/' + '*.npy')
        ftest  = glob.glob(datadir + 'test/'  + '*.npy')
        if len(ftrain_TFR) == 0:
            print("Making TFRecords for training data...")
            U.make_TFRecord(inputdir+'TFRecords/'+TFRfile+'train.tfrecords', 
                            ftrain, inD, ilog, olog, batch_size, train_batches)
        if len(fvalid_TFR) == 0:
            print("\nMaking TFRecords for validation data...")
            U.make_TFRecord(inputdir+'TFRecords/'+TFRfile+'valid.tfrecords', 
                            fvalid, inD, ilog, olog, batch_size, valid_batches)
        if len(ftest_TFR) == 0:
            print("\nMaking TFRecords for test data...")
            U.make_TFRecord(inputdir+'TFRecords/'+TFRfile+'test.tfrecords', 
                            ftest,  inD, ilog, olog, batch_size, test_batches)
        print("\nTFRecords creation complete.")
        # Free memory
        del ftrain, fvalid, ftest
        # Get TFR file names for real this time
        ftrain_TFR = glob.glob(TFRpath + 'train*.tfrecords')
        fvalid_TFR = glob.glob(TFRpath + 'valid*.tfrecords')
        ftest_TFR  = glob.glob(TFRpath +  'test*.tfrecords')

    if gridsearch:
        # Train a model for each architecture, w/ unique directories
        print("\nPerforming a grid search.\n")
        archrep = []
        maxlen  = np.inf
        for arch in architectures:
            archstr = '-'.join([str(num) for num in arch])
            archrep.append(archstr)
            if len(archstr) < maxlen:
                maxlen = len(archstr)
            archdir = os.path.join(outputdir, archstr, '')
            wsplit  = weight_file.rsplit('/', 1)[1].rsplit('.', 1)
            wfile   = archdir + wsplit[0] + '_' + archstr + '.' + wsplit[1]
            U.make_dir(archdir)
            nn = NNModel(ftrain_TFR, fvalid_TFR, ftest_TFR, 
                         inD, outD, 
                         x_mean, x_std, y_mean, y_std, 
                         x_min,  x_max, y_min,  y_max, scalelims, 
                         ncores, batch_size, buffer_size, 
                         None, arch, 
                         lengthscale, max_lr, clr_mode, clr_steps, 
                         wfile, stop_file='./STOP', 
                         train_flag=True, shuffle=True)
            nn.train(train_batches, valid_batches, epochs, patience)
            P.loss(nn, archdir)
        # Print/save out the minmium validation loss for each architecture
        minvl = np.ones(len(architectures)) * np.inf
        print('Grid search summary')
        print('-------------------')
        for i, arch in enumerate(architectures):
            archstr  = archrep[i]
            archdir  = os.path.join(outputdir, archstr, '')
            history  = np.load(archdir+'history.npz')
            minvl[i] = np.amin(history['val_loss'])
            print(archstr.ljust(maxlen, ' ') + ': ' + str(minvl[i]))
            if i == 0:
                mode = 'w'
            else:
                mode = 'a'
            with open(outputdir + 'gridsearch.txt', mode) as foo:
                foo.write(archstr.ljust(maxlen, ' ') + ': ' \
                          + str(minvl[i]) + '\n')
        return

    # Train a model
    if trainflag:
        print('\nBeginning model training.\n')
        nn = NNModel(ftrain_TFR, fvalid_TFR, ftest_TFR, 
                     inD, outD, 
                     x_mean, x_std, y_mean, y_std, 
                     x_min,  x_max, y_min,  y_max, scalelims, 
                     ncores, batch_size, buffer_size, 
                     convlayers, denselayers, 
                     lengthscale, max_lr, clr_mode, clr_steps, 
                     weight_file, stop_file='./STOP', 
                     train_flag=True, shuffle=True, resume=resume)
        nn.train(train_batches, valid_batches, epochs, patience)
        # Plot the loss
        P.loss(nn, plotdir)

    # Call new model with shuffle=False
    nn = NNModel(ftrain_TFR, fvalid_TFR, ftest_TFR, 
                 inD, outD, 
                 x_mean, x_std, y_mean, y_std, 
                 x_min,  x_max, y_min,  y_max, scalelims, 
                 ncores, batch_size, buffer_size, 
                 convlayers, denselayers, 
                 lengthscale, max_lr, clr_mode, clr_steps, 
                 weight_file, stop_file='./STOP', 
                 train_flag=False, shuffle=False, resume=False)
    nn.model.load_weights(weight_file) # Load the model
    # Save in ONNX format
    onnx_model = keras2onnx.convert_keras(nn.model)
    onnx.save_model(onnx_model, nn.weight_file.rsplit('.', 1)[0] + '.onnx')

    # Validate model
    if (validflag or trainflag) and not rng_test:
        print('\nValidating the model...\n')
        rmse_val  = S.rmse(nn, batch_size, valid_batches, outD,
                           y_mean, y_std, y_min, y_max, scalelims, 
                           preddir, mode='val')
        rmse_mean = np.mean(rmse_val, 0)
        print('Val RMSE     :', rmse_val)
        print('Val Mean RMSE:', rmse_mean)
        np.savez(outputdir+rmse_file+'_val.npz', 
                 rmse=rmse_val, rmse_mean=rmse_mean)

        r2_val = S.r2(nn, batch_size, valid_batches, outD,
                      y_mean, y_std, y_min, y_max, scalelims, 
                      preddir, mode='val')
        r2_mean = np.mean(r2_val, 0)
        print('Val R2     :', r2_val)
        print('Val Mean R2:', r2_mean)
        np.savez(outputdir+r2_file+'_val.npz', 
                 r2=r2_val, r2_mean=r2_mean)

    # Evaluate model on test set
    if testflag and not rng_test:
        print('\nTesting the model...\n')
        rmse_test  = S.rmse(nn, batch_size, test_batches, outD,
                            y_mean, y_std, y_min, y_max, scalelims, 
                            preddir, mode='test')
        rmse_mean = np.mean(rmse_test, 0)
        print('Test RMSE     :', rmse_test)
        print('Test Mean RMSE:', rmse_mean)
        np.savez(outputdir+rmse_file+'_test.npz', 
                 rmse=rmse_test, rmse_mean=rmse_mean)

        # R2
        r2_test = S.r2(nn, batch_size, test_batches, outD,
                       y_mean, y_std, y_min, y_max, scalelims, 
                       preddir, mode='test')
        r2_mean = np.mean(r2_test, 0)
        print('Test R2     :', r2_test)
        print('Test Mean R2:', r2_mean)
        np.savez(outputdir+r2_file+'_test.npz', 
                 r2=r2_test, r2_mean=r2_mean)

    # Plot requested cases
    predfoo = sorted(glob.glob(preddir+'test/testpred*'))
    truefoo = sorted(glob.glob(preddir+'test/testtrue*'))
    if len(predfoo) > 0 and len(truefoo) > 0:
        for val in plot_cases:
            fname    = plotdir + 'spec' + str(val) + '_pred-vs-true.png'
            predspec = np.load(predfoo[val // batch_size])[val % batch_size]
            predspec = U.denormalize(U.descale(predspec, 
                                               y_min, y_max, scalelims),
                                     y_mean, y_std)
            truespec = np.load(truefoo[val // batch_size])[val % batch_size]
            truespec = U.denormalize(U.descale(truespec, 
                                               y_min, y_max, scalelims),
                                     y_mean, y_std)
            P.plot_spec(fname, predspec, truespec, xvals, xlabel, ylabel, olog)

    return


