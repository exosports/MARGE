"""
Contains classes/functions related to NN models.

NNModel: class that builds out a specified NN.

driver: function that handles data & model initialization, and
        trains/validates/tests the model.

"""

import sys, os, platform
import datetime
import time
import random
import glob
import pickle
import functools
import logging
logging.setLoggerClass(logging.Logger)
import types

import numpy as np
import matplotlib as mpl
if platform.system() == 'Darwin':
    # Mac fix: use a different backend
    mpl.use("TkAgg")
import matplotlib.pyplot as plt

from sklearn import metrics

import tensorflow as tf
# Keras
import tensorflow.keras as keras
K = keras.backend
from tensorflow.keras.models  import Sequential, Model
from tensorflow.keras.layers  import (Input, Dense, Reshape,
                           Conv1D,    Conv2D,    Conv3D,
                           Conv2DTranspose, Conv3DTranspose, 
                           SeparableConv1D, SeparableConv2D, 
                           DepthwiseConv1D, DepthwiseConv2D, 
                           MaxPooling1D,     MaxPooling2D,     MaxPooling3D,
                           AveragePooling1D, AveragePooling2D, AveragePooling3D,
                           BatchNormalization, 
                           Dropout, Flatten, Lambda, Layer, Wrapper, concatenate)
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.optimizers   import RMSprop, Adadelta, Adam
from tensorflow.python import debug as tf_debug

import optuna
import dask.distributed

#from   onnx2keras import onnx_to_keras
try:
    import tf2onnx
except:
    tf2onnx = None

import callbacks as C
from layers import ConcreteDropout
import loader    as L
import utils     as U
import plotter   as P
import stats     as S
import custom_logger as CL
logging.setLoggerClass(CL.MARGE_Logger)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

logger = logging.getLogger('MARGE.'+__name__)


class NNModel:
    """
    Builds/trains NN model according to specified parameters.

    __init__: Initialization of the NN model.

    train: Trains the NN model.
    
    Yeval: Makes predictions on the chosen dataset and saves them as NPY files.
    """

    def __init__(self, ftrain_TFR, fvalid_TFR, ftest_TFR,
                 ishape, oshape, olog,
                 x_mean, x_std, y_mean, y_std,
                 x_min,  x_max, y_min,  y_max, scalelims,
                 ncores, buffer_size, batch_size, nbatches,
                 layers, lay_params, activations, nodes,
                 kernel_regularizer, 
                 lossfunc, lengthscale = 1e-3, max_lr=1e-1,
                 model_predict=None, model_evaluate=None, 
                 clr_mode='triangular', clr_steps=2000,
                 weight_file = 'nn_weights.keras', stop_file = './STOP',
                 train_flag = True,
                 epsilon=1e-6,
                 debug=False, shuffle=False, resume=False,
                 verb=2):
        """
        ftrain_TFR : list, strings. TFRecords for the training   data.
        fvalid_TFR : list, strings. TFRecords for the validation data.
        ftest_TFR  : list, strings. TFRecords for the test       data.
        ishape     : tuple, ints. Shape of the input  data.
        oshape     : tuple, ints. Shape of the output data.
        olog       : bool or array of bool.  Determines if the target values are log10-scaled.
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
        nodes      : list, ints. For the layers with nodes,
                                 number of nodes per layer.
        kernel_regularizer: func.  Specifies the kernel regularizer to use for each layer.
                                   Use None for no regularization.
        lossfunc   : function.  Loss function to use.  If None, uses a mean-squared-error loss.
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
        self.train_dataset   = U.load_TFdataset(ftrain_TFR, ncores, batch_size,
                                                buffer_size, ishape, oshape,
                                                x_mean, x_std, y_mean, y_std,
                                                x_min,  x_max, y_min,  y_max,
                                                scalelims, shuffle)
        self.valid_dataset   = U.load_TFdataset(fvalid_TFR, ncores, batch_size,
                                                buffer_size, ishape, oshape,
                                                x_mean, x_std, y_mean, y_std,
                                                x_min,  x_max, y_min,  y_max,
                                                scalelims, shuffle)
        self.test_dataset    = U.load_TFdataset(ftest_TFR,  ncores, batch_size,
                                                buffer_size, ishape, oshape,
                                                x_mean, x_std, y_mean, y_std,
                                                x_min,  x_max, y_min,  y_max,
                                                scalelims, shuffle)
        # Other variables
        self.ishape = ishape
        self.oshape = oshape
        self.D      = np.product(oshape)
        self.olog   = olog

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

        if lossfunc is None:
            lossfunc = keras.losses.MeanSquaredError
            lossfunc.__name__ = 'mse'
        #else:
        #    self.lossfunc = lossfunc
        self.lengthscale = lengthscale
        self.max_lr      = max_lr
        self.clr_mode    = clr_mode
        self.clr_steps   = clr_steps
        
        # only used for ConcreteDropout / SpatialConcreteDropout
        Ntrain = self.train_batches * self.batch_size
        wd = self.lengthscale**2. / Ntrain
        dd = 2. / Ntrain
        
        self.epsilon    = epsilon # To ensure log(0) never happens

        self.train_flag = train_flag
        self.resume     = resume
        self.shuffle    = shuffle
        
        if verb:
            # 1 line of output per epoch
            self.verb = 2
        else:
            # Silent - no output during training
            self.verb = 0
        
        # String identifier for the architecture - filled in below
        self.architecture = ''

        ### Build model
        inp = Input(shape=ishape)
        x = inp
        # Hidden layers
        n = 0 # Counter for layers with nodes
        for i in range(len(layers)):
            if layers[i] in ['conv1d', 'conv2d', 'conv3d', 
                             'conv2dtranspose', 'conv3dtranspose', 
                             'conv1dseparable', 'conv2dseparable', 
                             'conv1ddepthwise', 'conv2ddepthwise']:
                tshape = tuple(val for val in K.int_shape(x) if val is not None)
                if 'conv1d' in layers[i]:
                    if 'separable' in layers[i]:
                        Conv = SeparableConv1D
                    elif 'depthwise' in layers[i]:
                        Conv = DepthwiseConv1D
                    else:
                        Conv = Conv1D
                    dval = 1
                elif 'conv2d' in layers[i]:
                    if 'transpose' in layers[i]:
                        Conv = Conv2DTranspose
                    elif 'separable' in layers[i]:
                        Conv = SeparableConv2D
                    elif 'depthwise' in layers[i]:
                        Conv = DepthwiseConv2D
                    else:
                        Conv = Conv2D
                    dval = 2
                elif 'conv3d' in layers[i]:
                    if 'transpose' in layers[i]:
                        Conv = Conv3DTranspose
                    else:
                        Conv = Conv3D
                    dval = 3
                format_layparam = (str(lay_params[i]).replace('(', '')
                                                     .replace(')', '')
                                                     .replace(',', '.')
                                                     .replace(' ', ''))
                self.architecture += layers[i] + \
                                     '-kernel' + format_layparam + \
                                     '-nodes'  + str(nodes[n]) + '-'
                # Check if a channel must be added for conv features
                if len(tshape) == dval:
                    logger.warning("Conv layer dimensionality matches the " +\
                                   "input's dimensionality.  Adding a channel.")
                    x = Reshape(tshape + (1,))(x)
                if type(activations[n]) == str:
                    # Simple activation: pass as layer parameter
                    x = Conv(filters=nodes[n],
                             kernel_size=lay_params[i],
                             activation=activations[n],
                             kernel_regularizer=kernel_regularizer,
                             padding='same')(x)
                    self.architecture += activations[n]
                else:
                    # Advanced activation: use as its own layer
                    x = Conv(filters=nodes[n],
                             kernel_size=lay_params[i],
                             kernel_regularizer=kernel_regularizer,
                             padding='same')(x)
                    x = activations[n](x)
                    if activations[n] is not None:
                        self.architecture += activations[n].str
                n += 1
            elif layers[i] in ['dense', 'concretedropout']:
                if i > 0:
                    if layers[i-1] in ['conv1d', 'conv2d', 'conv3d',
                                       'maxpool1d', 'maxpool2d', 'maxpool3d',
                                       'avgpool1d', 'avgpool2d', 'avgpool3d']:
                        logger.warning('Dense layer follows a not-Dense layer. Flattening.')
                        x = Flatten()(x)
                        self.architecture += 'flat--'
                elif i==0 and len(ishape) > 1:
                    logger.warning("Dense layer follows >1D input layer. Flattening.")
                    x = Flatten()(x)
                    self.architecture += 'flat--'
                if layers[i] == 'dense':
                    self.architecture += layers[i] + '-nodes' + str(nodes[n]) + '-'
                    if type(activations[n]) == str:
                        x = Dense(nodes[n], activation=activations[n],
                                  kernel_regularizer=kernel_regularizer)(x)
                        self.architecture += activations[n]
                    else:
                        x = Dense(nodes[n], kernel_regularizer=kernel_regularizer)(x)
                        x = activations[n] (x)
                        if activations[n] is not None:
                            self.architecture += activations[n].str
                elif layers[i] == 'concretedropout':
                    x, _ = ConcreteDropout(nodes[n], activation=activations[n], 
                                              weight_regularizer=wd, 
                                              dropout_regularizer=dd,
                                              kernel_regularizer=kernel_regularizer)(x)
                    self.architecture += 'concdrop-nodes' + str(nodes[n]) + '-'
                    # Handle activation
                    if type(activations[n]) == str:
                        self.architecture += activations[n]
                    elif activations[n] is not None:
                        self.architecture += activations[n].str
                n += 1
            elif 'pool' in layers[i]:
                if layers[i] in ['maxpool1d', 'maxpool2d', 'maxpool3d']:
                    if layers[i-1] == 'dense' or layers[i-1] == 'flatten':
                        raise ValueError('MaxPool layers must follow Conv1d or ' \
                                       + 'Pool layer.')
                    if layers[i] == 'maxpool1d':
                        x = MaxPooling1D(pool_size=lay_params[i])(x)
                    elif layers[i] == 'maxpool2d':
                        x = MaxPooling2D(pool_size=lay_params[i])(x)
                    elif layers[i] == 'maxpool3d':
                        x = MaxPooling3D(pool_size=lay_params[i])(x)
                    self.architecture += 'mpool'
                elif layers[i] in ['avgpool1d', 'avgpool2d', 'avgpool3d']:
                    if layers[i-1] == 'dense' or layers[i-1] == 'flatten':
                        raise ValueError('AvgPool layers must follow Conv1d or ' \
                                       + 'Pool layer.')
                    if layers[i] == 'avgpool1d':
                        x = AveragePooling1D(pool_size=lay_params[i])(x)
                    elif layers[i] == 'avgpool2d':
                        x = AveragePooling2D(pool_size=lay_params[i])(x)
                    elif layers[i] == 'avgpool3d':
                        x = AveragePooling3D(pool_size=lay_params[i])(x)
                    self.architecture += 'apool'
                format_layparam = (str(lay_params[i]).replace('(', '')
                                                     .replace(')', '')
                                                     .replace(',', '.')
                                                     .replace(' ', ''))
                self.architecture +=  layers[i][-2] + 'd-' + format_layparam
            elif layers[i] == 'dropout':
                if self.train_flag:
                    x = Dropout(lay_params[i])(x)
                    self.architecture += 'drop'+str(lay_params[i])[:6]
            elif layers[i] == 'flatten':
                x = Flatten()(x)
                self.architecture += 'flat'
            elif layers[i] == 'batchnorm':
                x = BatchNormalization()(x)
                self.architecture += 'bn'
            else:
                raise ValueError("Unknown layer type: " + layers[i])
            if i < len(layers) - 1:
                self.architecture += '--'
        # Clean up architecture name
        self.architecture = (self.architecture.replace('sigmoid', 'sig')
                                              .replace('linear', 'lin')
                                              .replace('exponential', 'exp')
                                              .replace('_', '')
                                              .replace('None', ''))
        # Output layer(s)
        if 'heteroscedastic' in lossfunc.__name__:
            mean, _ = ConcreteDropout(self.D, 
                                      weight_regularizer=wd, 
                                      dropout_regularizer=dd,
                                      kernel_regularizer=kernel_regularizer)(x)
            log_var, _ = ConcreteDropout(int(self.D * (self.D+1)/2), 
                                      weight_regularizer=wd, 
                                      dropout_regularizer=dd,
                                      kernel_regularizer=kernel_regularizer)(x)
            out = concatenate([mean, log_var])
        else:
            x = Dense(np.prod(oshape), kernel_regularizer=kernel_regularizer)(x)
            out = Reshape(oshape)(x)

        self.model = Model(inp, out)

        # Compile model
        self.model.compile(optimizer=Adam(learning_rate=self.lengthscale, amsgrad=True),
                           loss=lossfunc())
        if self.verb:
            self.model.summary(print_fn=logger.info)

    def train(self, epochs=100, patience=50, save=True, parallel=False):
        """
        Trains model.

        Inputs
        ------
        epochs     : int. Maximum number of iterations through the dataset to train.
        patience   : int. If no model improvement after `patience` epochs,
                          ends training.
        save       : bool. Determines whether to save the model.
        parallel   : bool. If True, Ctrl+C does not terminate model training (models are training in parallel, and Ctrl+C is not communicated to those processes).
                           If False, Ctrl+C will terminate model training at the end of the next epoch.
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
                raise Exception('Resuming training for .onnx models is not ' + \
                                'yet available. Please specify a .keras file.')
            else:
                self.model.load_weights(self.weight_file)
                # TODO: Want to switch to the following, but it fails w/ ConcreteDropout layers:
                #self.model = keras.models.load_model(self.weight_file)
            try:
                init_epoch = len(np.load(fhistory)['loss'])
            except:
                logger.warning("Resume specified, but history file not found.\n" \
                             + "Training a new model.")
                init_epoch = 0
        # Train a new model
        else:
            init_epoch = 0

        ### Callbacks
        # Save the model weights
        callbacks = []
        if save:
            model_checkpoint = keras.callbacks.ModelCheckpoint(self.weight_file,
                                            monitor='val_loss',
                                            save_best_only=True,
                                            save_weights_only=False,
                                            mode='auto',
                                            verbose=1)
            callbacks.append(model_checkpoint)
        # Handle Ctrl+C or STOP file to halt training
        if not parallel:
            sig = C.SignalStopping(stop_file=self.stop_file)
            callbacks.append(sig)
        # Cyclical learning rate
        clr = C.CyclicLR(base_lr=self.lengthscale, max_lr=self.max_lr,
                         step_size=self.clr_steps, mode=self.clr_mode)
        callbacks.append(clr)
        # Early stopping criteria
        Early_Stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   min_delta=0,
                                                   patience=patience,
                                                   verbose=1, mode='auto')
        callbacks.append(Early_Stop)
        # Stop if NaN loss occurs
        Nan_Stop = keras.callbacks.TerminateOnNaN()
        callbacks.append(Nan_Stop)
        # Log the per-epoch output to the log file
        if self.verb > 1:
            logfile = open(logger.flog, 'a')
            log_callback = keras.callbacks.LambdaCallback(
                                on_epoch_end = lambda epoch, logs: logfile.write(f"[{datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] Epoch: {epoch+1:>{len(str(epochs))}}/{epochs} --- loss: {logs['loss']:<.16f} --- val_loss: {logs['val_loss']:<.16f}\n"),  
                                on_train_end = lambda logs: (logfile.write('\n'), logfile.close())
                            )
            callbacks.append(log_callback)

        ### Train the model
        if self.train_flag:
            # Ensure at least 1 epoch happens when training
            if init_epoch >= epochs:
                logger.warning('The requested number of training epochs ('+str(epochs) +\
                      ') is less than or equal to the\nepochs that the model '+\
                      'has already been trained for ('+str(init_epoch)+').  ' +\
                      'The model has\nbeen loaded, but not trained further.')
                self.historyNN = tf.keras.callbacks.History()
                self.historyNN.history = np.load(fhistory)
                self.historyCLR = None
                return
            # Run the training
            self.historyNN = self.model.fit(self.train_dataset, 
                                             initial_epoch=init_epoch,
                                             epochs=epochs,
                                             steps_per_epoch=self.train_batches,
                                             verbose=self.verb,
                                             validation_data=self.valid_dataset,
                                             validation_steps=self.valid_batches,
                                             validation_freq=1,
                                             callbacks=callbacks)
            # Save out the history
            self.historyCLR = clr.history
            if save:
                if not self.resume:
                    np.savez(fhistory, loss=self.historyNN.history['loss'],
                                   val_loss=self.historyNN.history['val_loss'],
                                         lr=self.historyCLR['lr'],
                                   clr_loss=self.historyCLR['loss'])
                else:
                    history  = np.load(fhistory)
                    loss     = np.concatenate((history['loss'],
                                self.historyNN.history['loss']))
                    val_loss = np.concatenate((history['val_loss'],
                                self.historyNN.history['val_loss']))
                    np.savez(fhistory, loss=loss, val_loss=val_loss)
                    self.historyNN.history['loss'] = loss
                    self.historyNN.history['val_loss'] = val_loss

        # Load best set of weights
        if save:
            self.model.load_weights(self.weight_file)
            # TODO: Want to switch to the following, but it fails w/ ConcreteDropout layers:
            #self.model = keras.models.load_model(self.weight_file)
            # Save the full model
            # TODO: fails when reloading activation functions that were specified as layers
            #self.model.save(self.weight_file, include_optimizer=True)

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
        assert not self.shuffle, "This model has shuffled TFRecords.\nCreate "+\
                       "a new NNModel object with shuffle=False and try again."

        assert dataset in ['train', 'valid', 'test'], "Invalid specification "+\
                            "for `dataset` parameter of NNModel.Yeval().\n" +\
                            "Allowed options: 'train', 'valid', or 'test'\n" +\
                            "Please correct this and try again."
        TFRdataset = getattr(self, dataset+'_dataset').as_numpy_iterator()
        num_batches = getattr(self, dataset+'_batches')

        # Prefix for the savefiles
        assert mode in ['pred', 'true'], "Invalid specification for `mode` "+\
                                         "parameter of NNModel.Yeval().\n"+\
                                         "Allowed options: 'pred' or 'true'\n"+\
                                         "Please correct this and try again."
        fname = ''.join([preddir, dataset, os.sep, mode])

        if denorm:
            fname = ''.join([fname, '-denorm_'])
        else:
            fname = ''.join([fname, '-norm_'])

        U.make_dir(preddir+dataset) # Ensure the directory exists

        # Save out the Y values
        y_batch = np.zeros((self.batch_size, *self.oshape))
        for i in range(num_batches):
            foo = ''.join([fname, str(i).zfill(len(str(num_batches))), '.npy'])
            batch = TFRdataset.next() # Get next batch
            if mode == 'pred': # Predicted Y values
                x_batch = batch[0] # X values - inputs
                y_batch = self.model.predict(x_batch, verbose=0)
            else:  # True Y values
                y_batch = batch[1] # Y values - outputs
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


def objective(trial, ftrain_TFR, fvalid_TFR, ftest_TFR,
              weight_file, ishape, oshape, olog,
              x_mean, x_std, y_mean, y_std,
              x_min,  x_max, y_min,  y_max, scalelims,
              ncores, buffer_size, batch_size,
              nbatches, lossfunc, lengthscale, max_lr,
              clr_mode, clr_steps,
              optnlays, optlayer, optnnode, optconvnode, optactiv, optactrng,
              optminlr, optmaxlr, 
              nodes, layers, kernel_regularizer, 
              epochs, patience,
              igpu):
    """
    Objective function to minimize during hyperparameter optimization.
    functools.partial() should be used to freeze the non-varying parameters.
    """
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[igpu], 'GPU')
        if optnlays is not None:
            if len(optnlays) == 2:
                nlayers = trial.suggest_int("layers", optnlays[0], optnlays[1])
            elif len(optnlays) == 1:
                nlayers = optnlays
            else:
                raise ValueError("`optnlays` specification not understood.\n" +\
                    "Expected to receive 1 or 2 ints.\nReceived: " + str(optnlays))
        else:
            nlayers = len(layers)
        n = 0
        nnodes = []
        activs = []
        actval = []
        activations = []
        selected_layers = []
        for i in range(nlayers):
            #if i==0 and optlayer is not None:
            #    layers.append(trial.suggest_categorical("layer_"+str(i+1), optlayer))
            #else:
            #    layers.append('dense')
            if optlayer is None:
                selected_layers.append(layers[i])
            else:
                selected_layers.append(optlayer[i])
            if optnnode is None:
                if selected_layers[i] not in ['flatten']:
                    nnodes.append(nodes[n])
                    n += 1
            else:
                if selected_layers[-1] in ['conv1d', 'conv2d', 'conv3d'] and optconvnode is not None:
                    nnodes.append(trial.suggest_categorical("nodes_"+str(i+1), optconvnode))
                else:
                    nnodes.append(trial.suggest_categorical("nodes_"+str(i+1), optnnode))
            if selected_layers[-1] not in ['flatten']:
                activs.append(trial.suggest_categorical("activation_"+str(i+1), optactiv))
                if activs[-1] in ['leakyrelu', 'elu'] and optactrng is not None:
                    actval.append(trial.suggest_float("act_val_"+str(i+1), optactrng[0], optactrng[1]))
                else:
                    actval.append(None)
            activations.append(L.load_activation(activs[-1], actval[-1]))
        if None not in [optminlr, optmaxlr]:
            lengthscale = trial.suggest_float("min_lr", optminlr, optmaxlr)
            max_lr = trial.suggest_float("max_lr", lengthscale, optmaxlr)
        lay_params, _ = U.prepare_layers(selected_layers, [None]*nlayers, nnodes)
        model = NNModel(ftrain_TFR, fvalid_TFR, ftest_TFR,
                        ishape, oshape, olog,
                        x_mean, x_std, y_mean, y_std,
                        x_min,  x_max, y_min,  y_max, scalelims,
                        ncores, buffer_size, batch_size, nbatches,
                        selected_layers, lay_params, activations, nnodes,
                        kernel_regularizer, 
                        lossfunc=lossfunc, lengthscale=lengthscale, max_lr=max_lr,
                        clr_mode=clr_mode, clr_steps=clr_steps,
                        weight_file=weight_file, train_flag=True,
                        epsilon=1e-6, shuffle=True, verb=0)
        model.train(epochs, patience, save=False, parallel=True)
        return np.amin(model.historyNN.history['val_loss'])
    except Exception as e:
        logger.error("Trial " + str(trial.number) + "failed with error:\n" + e)
        return 1e100

def driver(inputdir, outputdir, datadir, plotdir, preddir,
           trainflag, validflag, testflag,
           normalize, fxmean, fxstd, fymean, fystd,
           scale, fxmin, fxmax, fymin, fymax, scalelims,
           rmse_file, r2_file, statsaxes,
           ishape, oshape, ilog, olog,
           fTFR, batch_size, set_nbatches, ncores, buffer_size,
           optimize, optfunc, optngpus, opttime, optnlays, optlayer, optnnode,
           optactiv, optactrng, optminlr, optmaxlr, optmaxconvnode,
           gridsearch, 
           layers, lay_params, activations, nodes, kernel_regularizer, 
           lossfunc, lengthscale, max_lr, clr_mode, clr_steps,
           model_predict, model_evaluate, 
           epochs, patience,
           weight_file, resume,
           plot_cases, fxvals, xlabel, ylabel, smoothing,
           filters=None, filtconv=1., 
           use_cpu=False, verb=2):
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
    fxmean     : string. Path/to/file of mean  of training data inputs.
    fymean     : string. Path/to/file of mean  of training data outputs.
    fxstd      : string. Path/to/file of standard deviation of training data inputs.
    fystd      : string. Path/to/file of standard deviation of training data outputs.
    scale      : bool.   Determines whether to scale the data.
    fxmin      : string. Path/to/file of minima of training data inputs.
    fxmax      : string. Path/to/file of maxima of training data inputs.
    fymin      : string. Path/to/file of minima of training data outputs.
    fymax      : string. Path/to/file of maxima of training data outputs.
    scalelims  : list, floats. [min, max] of range of scaled data.
    rmse_file  : string. Prefix for savefiles for RMSE calculations.
    r2_file    : string. Prefix for savefiles for R2 calculations.
    statsaxes  : string. Determines which axes to compute stats over.
                         Options: all - all axes except last axis
                                batch - only 0th axis
    ishape     : tuple, ints. Dimensionality of the input  data, without the batch size.
    oshape     : tuple, ints. Dimensionality of the output data, without the batch size.
    ilog       : bool.   Determines whether to take the log10 of intput  data.
    olog       : bool.   Determines whether to take the log10 of output data.
    fTFR       : list of list of strings.  Contains the TFRecords file paths.
                         Element 0: list of training TFRecords files
                         Element 1: list of validation TFRecords files
                         Element 2: list of test TFRecords files
    batch_size : int.    Size of batches for training/validating/testing.
    set_nbatches: tuple, ints.  Number of batches in the training/validation/test sets.
    ncores     : int.    Number of cores to use to load data cases.
    buffer_size: int.    Number of data cases to pre-load in memory.
    optimize   : int.    Determines whether to use hyperparameter optimization.
                         If >0, determines number of trials to use when optimizing.
    optfunc    : function. Function to use while optimizing with Optuna.
                           If None, defaults to `objective` in this module.
    optngpus   : int.    Number of GPUs to use while optimizing hyperparameters.
    opttime    : int.    Sets the timeout limit for optimization.
                         If None, runs until `optimize` trials have been completed.
    optnlays   : list, ints. Determines the number of hidden layers to optimize over.
    optlayer   : list, strs. Determines which types of hidden layers to optimize over.  Only affects the first layer.
    optnnode   : list, ints. Determines the number of nodes per hidden layer to optimize over.
    optactiv   : list, strs. Determines the activation functions to optimize over.
    optactrng  : list, floats. Determines the range to explore for advanced activation functions to optimize over.
    optminlr   : float.  Minimum learning rate when optimizing.
    optmaxlr   : float.  Maximum learning rate when optimizing.
    optmaxconvnode: int. Maximum number of features allowed for convolutional layers.
    gridsearch : bool.   Determines whether to perform a grid search over
                         multiple architectures.
    layers     : list, str.  Types of hidden layers.
    lay_params : list, ints. Parameters for the layer type
                             E.g., kernel size
    activations: list, str.  Activation functions for each hidden layer.
    nodes      : list, ints. For layers with nodes, number of nodes per layer.
    kernel_regularizer: func.  Specifies the kernel regularizer to use for each layer.
                               Use None for no regularization.
    lossfunc   : function.  Function to use for the loss function.  If None, uses a mean-squared-error loss
    lengthscale: float.  Minimum learning rat.e
    max_lr     : float.  Maximum learning rate.
    clr_mode   : string. Sets the cyclical learning rate function.
    clr_steps  : int.    Number of steps per cycle of the learning rate.
    model_predict : func. Function to override Keras Model.predict  API.  Use None for default API.
    model_evaluate: func. Function to override Keras Model.evaluate API.  Use None for default API.
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
    smoothing  : int.    Sets the window size for a Savitsky-Golay filter to plot the smoothed results.
                         If 0 or False, does not apply smoothing during plotting.
    filters    : list, strings.  Paths/to/filter files.  Default: None
                         If specified, will compute RMSE/R2 stats over the
                         integrated filter bandpasses.
    filtconv   : float.  Conversion factor for filter file x-axis values to
                         desired unit.  Default: 1.0
    use_cpu    : bool.   Determines whether to run the NN model on the CPU (True) or GPU (False).
                         Default: False
    verb       : int.    Controls how much output is printed to terminal.
                         Default: 2
    """
    if use_cpu:
        # Ensure that a GPU is not used if there is one available
        tf.config.set_visible_devices([], 'GPU')
    # Get number of batches in each data set
    train_batches, valid_batches, test_batches = set_nbatches
    # Update `clr_steps`
    if clr_steps == "range test":
        clr_steps = train_batches * epochs
        rng_test  = True
    else:
        clr_steps = train_batches * int(clr_steps)
        rng_test  = False

    # Get mean/stdev for normalizing, min/max for scaling
    if normalize or scale:
        if normalize:
            logger.info('Normalizing the data...')
        if scale:
            logger.info('Scaling the data...')
        
        try:
            x_mean = np.load(fxmean)
            x_std  = np.load(fxstd)
            y_mean = np.load(fymean)
            y_std  = np.load(fystd)
            x_min  = np.load(fxmin)
            x_max  = np.load(fxmax)
            y_min  = np.load(fymin)
            y_max  = np.load(fymax)
        except:
            logger.info("Determining the minimum, maximum, mean, and " +\
                        "standard deviation based on Welford's method.\n")
            # Compute stats
            x_mean, x_std, x_min, x_max, \
            y_mean, y_std, y_min, y_max  = S.save_stats_files(glob.glob(datadir + 'train' + os.sep + '*.npz'),
                                                              ilog, olog, ishape, oshape,
                                                              fxmean, fxstd, fxmin,  fxmax,
                                                              fymean, fystd, fymin,  fymax, statsaxes, verb)
        if normalize:
            # Sanity check
            if np.any(x_std == 0):
                ibad = np.where(x_std == 0)
                raise ValueError("A standard deviation of 0 was found at the "+\
                                 "following indices for the input data:\n"+\
                                 str(ibad))
            elif np.any(y_std == 0):
                ibad = np.where(y_std == 0)
                raise ValueError("A standard deviation of 0 was found at the "+\
                                 "following indices for the output data:\n"+\
                                 str(ibad))
            if verb >= 3:
                logger.info("X mean:\n" + str(x_mean) + "\n")
                logger.info("Y mean:\n" + str(y_mean) + "\n")
                logger.info("X stdev:\n" + str(x_std) + "\n")
                logger.info("Y stdev:\n" + str(y_std) + "\n")
            if scale:
                x_min = U.normalize(x_min, x_mean, x_std)
                x_max = U.normalize(x_max, x_mean, x_std)
                y_min = U.normalize(y_min, y_mean, y_std)
                y_max = U.normalize(y_max, y_mean, y_std)
        else:
            x_mean = 0.
            x_std  = 1.
            y_mean = 0.
            y_std  = 1.
        if scale:
            if np.any(x_min == x_max):
                ibad = np.where(x_min == x_max)
                raise ValueError("The minimum and maximum are equal at the " +\
                                 "following indices for the input data:\n"+\
                                 str(ibad))
            elif np.any(y_min == y_max):
                ibad = np.where(y_min == y_max)
                raise ValueError("The minimum and maximum are equal at the " +\
                                 "following indices for the output data:\n"+\
                                 str(ibad))
            if verb >= 3:
                logger.info("X min:\n" +str (x_min) + "\n")
                logger.info("Y min:\n" + str(y_min) + "\n")
                logger.info("X max:\n" + str(x_max) + "\n")
                logger.info("Y max:\n" + str(y_max) + "\n")
        else:
            x_min =  0.
            x_max =  1.
            y_min =  0.
            y_max =  1.
            scalelims = [0., 1.]
    else:
        x_mean = 0.
        x_std  = 1.
        y_mean = 0.
        y_std  = 1.
        x_min  =  0.
        x_max  =  1.
        y_min  =  0.
        y_max  =  1.
        scalelims = [0., 1.]

    if olog:
        # To later properly calculate RMSE & R2 for log-scaled output
        try:
            y_mean_delog = np.load(fymean.replace(".npy", "_delog.npy"))
        except:
            logger.info("Calculating mean/stdev/min/max for the output data " +\
                        "without log-scaling, for RMSE and R2 calculations later.")
            y_mean_delog = S.get_stats(glob.glob(datadir + 'train' + os.sep + '*.npz'),
                                       ilog, False, ishape, oshape, statsaxes)[4]
            np.save(fymean.replace(".npy", "_delog.npy"), y_mean_delog)
    else:
        y_mean_delog = y_mean

    # Get TFRecord file names
    ftrain_TFR, fvalid_TFR, ftest_TFR = fTFR

    # Load the xvals
    if fxvals is not None:
        xvals = np.load(fxvals)
    else:
        xvals = None

    # Perform grid search
    if gridsearch:
        # Train a model for each architecture, w/ unique directories
        logger.info("Performing a grid search.\n")
        maxlen = 0
        architectures = []
        for i in range(len(layers)):
            nn = NNModel(ftrain_TFR, fvalid_TFR, ftest_TFR,
                         ishape, oshape, olog,
                         x_mean, x_std, y_mean, y_std,
                         x_min,  x_max, y_min,  y_max, scalelims,
                         ncores, buffer_size, batch_size,
                         [train_batches, valid_batches, test_batches],
                         layers[i], lay_params[i],
                         activations[i], nodes[i], kernel_regularizer, 
                         lossfunc, lengthscale, max_lr, model_predict, model_evaluate, 
                         clr_mode, clr_steps,
                         weight_file, stop_file='./STOP', resume=resume,
                         train_flag=True, shuffle=True, verb=verb)
            arch = nn.architecture
            if len(arch) > maxlen:
                maxlen = len(arch)
            architectures.append(arch)
            archdir = os.path.join(outputdir, arch, '')
            wsplit  = weight_file.rsplit(os.sep, 1)[1].rsplit('.', 1)
            wfile   = ''.join([archdir, wsplit[0], '_', arch, '.', wsplit[1]])
            nn.weight_file = wfile
            U.make_dir(archdir)
            nn.train(epochs, patience)
            P.loss(nn, archdir)
        # Print/save out the minmium validation loss for each architecture
        minvl = np.ones(len(architectures)) * np.inf
        gridsearch_output_str = '\nGrid search summary\n-------------------\n'
        with open(os.path.join(outputdir, 'gridsearch.txt'), 'w') as foo:
            foo.write('Grid search summary\n')
            foo.write('-------------------\n')
        for i, arch in enumerate(architectures):
            archdir  = os.path.join(outputdir, arch, '')
            history  = np.load(archdir+'history.npz')
            minvl[i] = np.amin(history['val_loss'])
            gridsearch_output_str += arch.ljust(maxlen, ' ') + ': ' + \
                                     str(minvl[i]) + '\n'
            with open(os.path.join(outputdir, 'gridsearch.txt'), 'a') as foo:
                foo.write(arch.ljust(maxlen, ' ') + ': ' \
                          + str(minvl[i]) + '\n')
        logger.info(gridsearch_output_str)
        return

    # Hyperparameter optimization
    if optimize:
        opt_str  = "Running hyperparameter optimization:\n"
        opt_str += "  Number of trials per GPU: " + str(optimize) + '\n'
        opt_str += "  Number of layers: " + str(optnlays[0]) + '-' + str(optnlays[1]) + '\n'
        opt_str += "  Number of nodes per layer: " + str(optnnode) + '\n'
        opt_str += "  Activation functions: " + str(optactiv) + '\n'
        if optactrng is not None:
            opt_str += "  Activation parameter range: " + str(optactrng[0]) + '-' + str(optactrng[1]) + '\n'
        if None not in [optminlr, optmaxlr]:
            opt_str += "  Minimum learning rate: " + str(optminlr) + '\n'
            opt_str += "  Maximum learning rate: " + str(optmaxlr) + '\n'
        if opttime is not None:
            opt_str += "  Time limit:" + str(opttime/60./60.) + "hrs\n"
        logger.info(opt_str)
        if optngpus>1:
            client = dask.distributed.Client()
            futures = []
            storage = optuna.integration.dask.DaskStorage()
        else:
            storage = None
        sampler = optuna.samplers.TPESampler(n_startup_trials=5)
        study = optuna.create_study(storage=storage, sampler=sampler, direction='minimize')
        if optnnode is not None and optmaxconvnode is not None:
            optconvnode = [val for val in optnnode if val <= optmaxconvnode]
        else:
            optconvnode = None
        for i in range(optngpus):
            if optfunc is None:
                optfunc = objective
            theobjective = functools.partial(optfunc,
                        ftrain_TFR=ftrain_TFR, fvalid_TFR=fvalid_TFR, ftest_TFR=ftest_TFR,
                        weight_file=weight_file, ishape=ishape, oshape=oshape, olog=olog,
                        x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std,
                        x_min=x_min,   x_max=x_max, y_min=y_min,   y_max=y_max, scalelims=scalelims,
                        ncores=ncores, buffer_size=buffer_size, batch_size=batch_size,
                        nbatches=[train_batches, valid_batches, test_batches],
                        lossfunc=lossfunc, lengthscale=lengthscale, max_lr=max_lr,
                        clr_mode=clr_mode, clr_steps=clr_steps,
                        optnlays=optnlays, optlayer=optlayer, optnnode=optnnode, optconvnode=optconvnode,
                        optactiv=optactiv, optactrng=optactrng,
                        optminlr=optminlr, optmaxlr=optmaxlr,
                        nodes=nodes, layers=layers, kernel_regularizer=kernel_regularizer, 
                        epochs=epochs, patience=patience, igpu=i)
            if optngpus>1:
                futures.append(client.submit(study.optimize, theobjective, n_trials=optimize, timeout=opttime, pure=False))
            else:
                study.optimize(theobjective, n_trials=optimize, timeout=opttime)
        if optngpus>1:
            dask.distributed.wait(futures)
        bestp_str = "Hyperparameter optimization complete.  Best trial:\n"
        for key, value in study.best_trial.params.items():
            bestp_str += "  {}: {}\n".format(key, value)
        logger.info(bestp_str)
        trials = study.get_trials()
        with open(os.path.join(outputdir, "optuna-study.dat"), "wb") as foo:
            pickle.dump(trials, foo)
        if optlayer is not None:
            U.write_ordered_optuna_summary(os.path.join(outputdir, 'optuna-ordered-summary.txt'), trials, optlayer)
        else:
            U.write_ordered_optuna_summary(os.path.join(outputdir, 'optuna-ordered-summary.txt'), trials, layers, nodes)
        return

    # Train a model
    if trainflag:
        logger.newline()
        logger.info('Beginning model training.\n')
        nn = NNModel(ftrain_TFR, fvalid_TFR, ftest_TFR,
                     ishape, oshape, olog,
                     x_mean, x_std, y_mean, y_std,
                     x_min,  x_max, y_min,  y_max, scalelims,
                     ncores, buffer_size, batch_size,
                     [train_batches, valid_batches, test_batches],
                     layers, lay_params, activations, nodes, kernel_regularizer, 
                     lossfunc, lengthscale, max_lr, model_predict, model_evaluate, 
                     clr_mode, clr_steps,
                     weight_file, stop_file='./STOP',
                     train_flag=True, shuffle=True, resume=resume, verb=verb)
        nn.train(epochs, patience)
        # Plot the loss
        P.loss(nn, plotdir)
    
    if not os.path.exists(weight_file):
        # No model trained this time or previously
        logger.error("No NN model file found.  Exiting.")
        return

    # Call new model with shuffle=False
    # lossfunc is set to None because we will not be training it, and
    # certain loss functions would throw an error here due to undefined shapes
    logger.info("Re-loading the trained model, with shuffling data turned off.")
    if lossfunc is not None:
        if 'heteroscedastic' not in lossfunc.__name__:
            lossfunc = None
    nn = NNModel(ftrain_TFR, fvalid_TFR, ftest_TFR,
                 ishape, oshape, olog,
                 x_mean, x_std, y_mean, y_std,
                 x_min,  x_max, y_min,  y_max, scalelims,
                 ncores, buffer_size, batch_size,
                 [train_batches, valid_batches, test_batches],
                 layers, lay_params, activations, nodes, kernel_regularizer, 
                 lossfunc, lengthscale, max_lr, model_predict, model_evaluate, 
                 clr_mode, clr_steps,
                 weight_file, stop_file='./STOP',
                 train_flag=False, shuffle=False, resume=False, verb=verb)
    if '.keras' in weight_file:
        nn.model.load_weights(weight_file) # Load the trained model
        # TODO: Want to switch to the following, but it fails w/ ConcreteDropout layers:
        #nn.model = keras.models.load_model(weight_file)
        # Save in ONNX format
        if tf2onnx is not None:
            logger.info("Converting Keras model to ONNX format.\n")
            # Need to explicitly pass in input_signature because Keras Functional models do not have _get_save_spec attribute
            input_signature = [tf.TensorSpec(nn.model.inputs[0].shape, nn.model.inputs[0].dtype, name='digit')]
            _ = tf2onnx.convert.from_keras(nn.model, input_signature, output_path=weight_file.replace('.keras', '.onnx'))
        else:
            logger.warning("tf2onnx package not found; model will not be converted to ONNX format.\n")
    #else:
    #    nn.model = onnx_to_keras(onnx.load_model(weight_file), ['input_1'])
    
    if rng_test:
        # Range test: do not validate or test model
        return

    # Stats for validation and/or test sets
    istat = ['valid', 'test']
    for i in range(2):
        if i==0 and (validflag or trainflag):
            logger.info('Validating the model...\n')
        elif i==0 and not (validflag or trainflag):
            # Don't validate the model
            continue
        if i==1 and testflag:
            logger.info('Testing the model...\n')
        elif i==1 and not testflag:
            # Don't test the model
            continue
        # Y values
        logger.info('Predicting...')
        fpred = nn.Yeval('pred', istat[i], preddir,
                            denorm=(normalize==False and scale==False))
        fpred = glob.glob(fpred + '*')

        logger.info('Loading the true Y values...')
        ftrue = nn.Yeval('true', istat[i], preddir,
                            denorm=(normalize==False and scale==False))
        ftrue = glob.glob(ftrue + '*')
        ### RMSE & R2
        logger.info('Calculating RMSE & R2...')
        if not normalize and not scale:
            pred_stats = S.rmse_r2(fpred, ftrue, y_mean,
                                  olog=olog, y_mean_delog=y_mean_delog,
                                  x_vals=xvals,
                                  filters=filters, filtconv=filtconv)
        else:
            pred_stats = S.rmse_r2(fpred, ftrue, y_mean,
                                  y_std, y_min, y_max, scalelims,
                                  olog, y_mean_delog,
                                  xvals, filters, filtconv)
        # RMSE
        if np.any(pred_stats[0] != -1) and np.any(pred_stats[1] != -1):
            logger.info('Normalized RMSE:\n' + str(pred_stats[0]) + "\n")
            logger.info('Mean normalized RMSE:\n    ' + str(np.mean(pred_stats[0])) + "\n")
            logger.info('Denormalized RMSE:\n' + str(pred_stats[1]) + "\n")
            logger.info('Mean denormalized RMSE:\n    ' + str(np.mean(pred_stats[1])) + "\n")
            np.savez(outputdir+rmse_file+'_'+istat[i]+'_norm.npz',
                     rmse=pred_stats[0], rmse_mean=np.mean(pred_stats[0]))
            saveRMSEnorm   = True
            saveRMSEdenorm = True
        elif np.any(pred_stats[0] != -1):
            logger.info('RMSE:\n' + str(pred_stats[0]) + "\n")
            logger.info('Mean RMSE:\n    ' + str(np.mean(pred_stats[0])) + "\n")
            saveRMSEnorm   = True
            saveRMSEdenorm = False
        elif np.any(pred_stats[1] != -1):
            logger.info('RMSE:\n' + str(pred_stats[1]) + "\n")
            logger.info('Mean RMSE:\n    ' + str(np.mean(pred_stats[1])) + "\n")
            saveRMSEnorm   = False
            saveRMSEdenorm = True
        else:
            logger.error("No files passed in to compute RMSE.")
            saveRMSEnorm   = False
            saveRMSEdenorm = False
        if saveRMSEnorm:
            try:
                P.plot(''.join([plotdir, rmse_file, '_'+istat[i]+'_norm.png']),
                       xvals, pred_stats[0], xlabel, 'RMSE')
            except:
                logger.warning("Normalized RMSE unable to be plotted.")
            np.savez(outputdir+rmse_file+'_'+istat[i]+'_norm.npz',
                     rmse=pred_stats[0], rmse_mean=np.mean(pred_stats[0]))
        if saveRMSEdenorm:
            try:
                P.plot(''.join([plotdir, rmse_file, '_'+istat[i]+'_denorm.png']),
                       xvals, pred_stats[1], xlabel, 'RMSE')
            except:
                logger.warning("Denormalized RMSE unable to be plotted.")
            np.savez(outputdir+rmse_file+'_'+istat[i]+'_denorm.npz',
                     rmse=pred_stats[1], rmse_mean=np.mean(pred_stats[1]))
        # R2
        if np.any(pred_stats[2] != -1) and np.any(pred_stats[3] != -1):
            logger.info('Normalized R2:\n' + str(pred_stats[2]) + "\n")
            logger.info('Mean normalized R2:\n    ' + str(np.mean(pred_stats[2])) + "\n")
            logger.info('Denormalized R2:\n' + str(pred_stats[3]) + "\n")
            logger.info('Mean denormalized R2:\n    ' + str(np.mean(pred_stats[3])) + "\n")
            saveR2norm   = True
            saveR2denorm = True
        elif np.any(pred_stats[2] != -1):
            logger.info('R2:\n' + str(pred_stats[2]) + "\n")
            logger.info('Mean R2:\n    ' + str(np.mean(pred_stats[2])) + "\n")
            saveR2norm   = True
            saveR2denorm = False
        elif np.any(pred_stats[3] != -1):
            logger.info('R2:\n' + str(pred_stats[3]) + "\n")
            logger.info('Mean R2:\n    ' + str(np.mean(pred_stats[3])) + "\n")
            saveR2norm   = False
            saveR2denorm = True
        else:
            logger.error("No files passed in to compute R2.")
            saveR2norm   = False
            saveR2denorm = False
        if saveR2norm:
            try:
                P.plot(''.join([plotdir, r2_file, '_'+istat[i]+'_norm.png']),
                       xvals, pred_stats[2], xlabel, '$R^2$')
            except:
                logger.warning("Normalized R2 unable to be plotted.")
            np.savez(outputdir+r2_file+'_'+istat[i]+'_norm.npz',
                     r2=pred_stats[2], r2_mean=np.mean(pred_stats[2]))
        if saveR2denorm:
            try:
                P.plot(''.join([plotdir, r2_file, '_'+istat[i]+'_denorm.png']),
                       xvals, pred_stats[3], xlabel, '$R^2$')
            except:
                logger.warning("Denormalized R2 unable to be plotted.")
            np.savez(outputdir+r2_file+'_'+istat[i]+'_denorm.npz',
                     r2=pred_stats[3], r2_mean=np.mean(pred_stats[3]))

    # Plot requested cases
    if not rng_test and plot_cases is not None:
        predfoo = sorted(glob.glob(os.path.join(preddir, 'test', 'pred*')))
        truefoo = sorted(glob.glob(os.path.join(preddir, 'test', 'true*')))
        if len(predfoo) > 0 and len(truefoo) > 0:
            logger.info("Plotting the requested cases...")
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
                P.plot_spec(fname, predspec, truespec, xvals, xlabel, ylabel, smoothing)
                nplot += 1
                print("  Plot " + str(nplot) + "/" + str(len(plot_cases)), end='\r')
            print("")
        else:
            logger.warning("Requested to plot test cases, but no predictions found in " + \
                           preddir + "test.")

    return nn
