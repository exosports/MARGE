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
import glob
import pickle
import functools

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
from tensorflow.keras.metrics import binary_accuracy
from tensorflow.keras.layers  import (Input, InputSpec, Dense, Reshape,
                           Convolution1D,    Convolution2D,    Convolution3D,
                           Convolution2DTranspose, Convolution3DTranspose, 
                           MaxPooling1D,     MaxPooling2D,     MaxPooling3D,
                           AveragePooling1D, AveragePooling2D, AveragePooling3D,
                           Dropout, Flatten, Lambda, Layer, Wrapper, concatenate)
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.optimizers   import RMSprop, Adadelta, Adam
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras import activations as tfactivations
from tensorflow.keras import regularizers
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python import debug as tf_debug

import optuna
import dask.distributed

#from   onnx2keras import onnx_to_keras
try:
    import tf2onnx
except:
    pass

import callbacks as C
import loader    as L
import utils     as U
import plotter   as P
import stats     as S

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


class ConcreteDropout(Layer):
  """Just your regular densely-connected NN layer, 
  plus a trainable dropout parameter.

  `Dense` implements the operation:
  `output = activation(dot(input, kernel) + bias)`
  where `activation` is the element-wise activation function
  passed as the `activation` argument, `kernel` is a weights matrix
  created by the layer, and `bias` is a bias vector created by the layer
  (only applicable if `use_bias` is `True`). These are all attributes of
  `Dense`.

  Note: If the input to the layer has a rank greater than 2, then `Dense`
  computes the dot product between the `inputs` and the `kernel` along the
  last axis of the `inputs` and axis 0 of the `kernel` (using `tf.tensordot`).
  For example, if input has dimensions `(batch_size, d0, d1)`,
  then we create a `kernel` with shape `(d1, units)`, and the `kernel` operates
  along axis 2 of the `input`, on every sub-tensor of shape `(1, 1, d1)`
  (there are `batch_size * d0` such sub-tensors).
  The output in this case will have shape `(batch_size, d0, units)`.

  Besides, layer attributes cannot be modified after the layer has been called
  once (except the `trainable` attribute).
  When a popular kwarg `input_shape` is passed, then keras will create
  an input layer to insert before the current layer. This can be treated
  equivalent to explicitly defining an `InputLayer`.

  Example:

  >>> # Create a `Sequential` model and add a Dense layer as the first layer.
  >>> model = tf.keras.models.Sequential()
  >>> model.add(tf.keras.Input(shape=(16,)))
  >>> model.add(tf.keras.layers.Dense(32, activation='relu'))
  >>> # Now the model will take as input arrays of shape (None, 16)
  >>> # and output arrays of shape (None, 32).
  >>> # Note that after the first layer, you don't need to specify
  >>> # the size of the input anymore:
  >>> model.add(tf.keras.layers.Dense(32))
  >>> model.output_shape
  (None, 32)

  Args:
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use.
      If you don't specify anything, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix.
    bias_initializer: Initializer for the bias vector.
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to
      the output of the layer (its "activation").
    kernel_constraint: Constraint function applied to
      the `kernel` weights matrix.
    bias_constraint: Constraint function applied to the bias vector.

  Input shape:
    N-D tensor with shape: `(batch_size, ..., input_dim)`.
    The most common situation would be
    a 2D input with shape `(batch_size, input_dim)`.

  Output shape:
    N-D tensor with shape: `(batch_size, ..., units)`.
    For instance, for a 2D input with shape `(batch_size, input_dim)`,
    the output would have shape `(batch_size, units)`.
  """

  def __init__(self,
               units,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               weight_regularizer=None,
               dropout_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               is_mc_dropout=True, 
               init_min=0.1, 
               init_max=0.1, 
               **kwargs):
    super(ConcreteDropout, self).__init__(
        activity_regularizer=activity_regularizer, **kwargs)
        
    self.units = int(units) if not isinstance(units, int) else units
    if self.units < 0:
      raise ValueError(f'Received an invalid value for `units`, expected '
                       f'a positive integer, got {units}.')
    self.activation = tfactivations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.weight_regularizer = weight_regularizer
    self.dropout_regularizer = dropout_regularizer
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.bias_constraint = constraints.get(bias_constraint)
    self.is_mc_dropout = is_mc_dropout
    
    self.input_spec = InputSpec(min_ndim=2)
    self.supports_masking = True
    
    self.p_logit = None
    self.init_min = np.log(init_min) - np.log(1. - init_min)
    self.init_max = np.log(init_max) - np.log(1. - init_max)
    
  def build(self, input_shape):
    dtype = dtypes.as_dtype(self.dtype or K.floatx())
    if not (dtype.is_floating or dtype.is_complex):
      raise TypeError('Unable to build `Dense` layer with non-floating point '
                      'dtype %s' % (dtype,))

    input_shape = tensor_shape.TensorShape(input_shape)
    last_dim = tensor_shape.dimension_value(input_shape[-1])
    if last_dim is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})
    self.kernel = self.add_weight(
        name='kernel',
        shape=(last_dim, self.units),
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        dtype=self.dtype,
        trainable=True)
    if self.use_bias:
      self.bias = self.add_weight(
          name='bias',
          shape=(self.units,),
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          dtype=self.dtype,
          trainable=True)
    else:
      self.bias = None
    self.p_logit = self.add_weight(name='p_logit',
                                   shape=(1,),
                                   initializer=tf.random_uniform_initializer(self.init_min, self.init_max),
                                   dtype=tf.dtypes.float32,
                                   trainable=True)
    self.built = True
  
  def concrete_dropout(self, x, p):
        """
        Concrete dropout - used at training time (gradients can be propagated)
        :param x: input
        :return:  approx. dropped out input
        """
        eps = 1e-07
        temp = 0.1

        unif_noise = tf.random.uniform(shape=tf.shape(x))
        drop_prob = (
            tf.math.log(p + eps)
            - tf.math.log(1. - p + eps)
            + tf.math.log(unif_noise + eps)
            - tf.math.log(1. - unif_noise + eps)
        )
        drop_prob = tf.math.sigmoid(drop_prob / temp)
        random_tensor = 1. - drop_prob

        retain_prob = 1. - p
        x *= random_tensor
        x /= retain_prob
        return x

  def call(self, inputs):
    # Apply dropout
    p = tf.math.sigmoid(self.p_logit)
    input_dim = int(inputs.shape[-1])
    weight = self.kernel
    kernel_regularizer = self.weight_regularizer * tf.reduce_sum(tf.square(weight)) / (1. - p)
    dropout_regularizer = p * tf.math.log(p) + (1. - p) * tf.math.log(1. - p)
    dropout_regularizer *= self.dropout_regularizer * input_dim
    regularizer = tf.reduce_sum(kernel_regularizer + dropout_regularizer)
    #if self.is_mc_dropout:
    inputs = self.concrete_dropout(inputs, p)
    
    # Now apply normal Dense calculations
    if self._dtype_policy.compute_dtype:
      if inputs.dtype.base_dtype != dtypes.as_dtype(self._dtype_policy.compute_dtype):
        inputs = math_ops.cast(inputs, dtype=dtypes.as_dtype(self._dtype_policy.compute_dtype))

    rank = inputs.shape.rank
    if rank == 2 or rank is None:
      # We use embedding_lookup_sparse as a more efficient matmul operation for
      # large sparse input tensors. The op will result in a sparse gradient, as
      # opposed to sparse_ops.sparse_tensor_dense_matmul which results in dense
      # gradients. This can lead to sigfinicant speedups, see b/171762937.
      if isinstance(inputs, sparse_tensor.SparseTensor):
        # We need to fill empty rows, as the op assumes at least one id per row.
        inputs, _ = sparse_ops.sparse_fill_empty_rows(inputs, 0)
        # We need to do some munging of our input to use the embedding lookup as
        # a matrix multiply. We split our input matrix into separate ids and
        # weights tensors. The values of the ids tensor should be the column
        # indices of our input matrix and the values of the weights tensor
        # can continue to the actual matrix weights.
        # The column arrangement of ids and weights
        # will be summed over and does not matter. See the documentation for
        # sparse_ops.sparse_tensor_dense_matmul a more detailed explanation
        # of the inputs to both ops.
        ids = sparse_tensor.SparseTensor(
            indices=inputs.indices,
            values=inputs.indices[:, 1],
            dense_shape=inputs.dense_shape)
        weights = inputs
        outputs = embedding_ops.embedding_lookup_sparse_v2(
            self.kernel, ids, weights, combiner='sum')
      else:
        outputs = gen_math_ops.MatMul(a=inputs, b=self.kernel)
    # Broadcast kernel to inputs.
    else:
      outputs = standard_ops.tensordot(inputs, self.kernel, [[rank - 1], [0]])
      # Reshape the output back to the original ndim of the input.
      if not context.executing_eagerly():
        shape = inputs.shape.as_list()
        output_shape = shape[:-1] + [self.kernel.shape[-1]]
        outputs.set_shape(output_shape)

    if self.use_bias:
      outputs = nn_ops.bias_add(outputs, self.bias)

    if self.activation is not None:
      outputs = self.activation(outputs)
    
    return outputs, regularizer
  """
  # No longer needed w/ newer versions of TF.  Left here for posterity.
  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if tensor_shape.dimension_value(input_shape[-1]) is None:
      raise ValueError(
          'The innermost dimension of input_shape must be defined, but saw: %s'
          % (input_shape,))
    return input_shape[:-1].concatenate(self.units)
  """
  def get_config(self):
    config = super(ConcreteDropout, self).get_config()
    config.update({
        'units': self.units,
        'activation': tfactivations.serialize(self.activation),
        'use_bias': self.use_bias,
        'kernel_initializer': initializers.serialize(self.kernel_initializer),
        'bias_initializer': initializers.serialize(self.bias_initializer),
        'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint': constraints.serialize(self.kernel_constraint),
        'bias_constraint': constraints.serialize(self.bias_constraint)
    })
    return config


class ConcreteDropoutImpl(Wrapper):
    """From https://github.com/yaringal/ConcreteDropout
    with modifications from https://github.com/deepskies/deeplyuncertain-public/blob/main/models/cd.py
    
    This wrapper allows to learn the dropout probability for any given input 
       Dense layer.
    ```python
        # as the first layer in a model
        model = Sequential()
        model.add(ConcreteDropout(Dense(8), input_shape=(16)))
        # now model.output_shape == (None, 8)
        # subsequent layers: no need for input_shape
        model.add(ConcreteDropout(Dense(32)))
        # now model.output_shape == (None, 32)
    ```
    `ConcreteDropout` can be used with arbitrary layers which have 2D
    kernels, not just `Dense`. However, Conv2D layers require different
    weighing of the regulariser (use SpatialConcreteDropout instead).
    # Arguments
        layer: a layer instance.
        weight_regularizer:
            A positive number which satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, 
            model precision $\\tau$ (inverse observation noise),
            and N the number of instances in the dataset.
            Note that kernel_regularizer is not needed.
        dropout_regularizer:
            A positive number which satisfies
                $dropout_regularizer = 2 / (\\tau * N)$
            with model precision $\tau$ (inverse observation noise) and N the 
            number of instances in the dataset.
            Note relation between dropout_regularizer and weight_regularizer:
                $weight_regularizer / dropout_regularizer = l**2 / 2$
            with prior lengthscale l. Note also that the factor of two should be
            ignored for cross-entropy loss, and used only for the eculedian loss
    """

    def __init__(self, layer, weight_regularizer=1e-6, dropout_regularizer=1e-5,
                 init_min=0.1, init_max=0.1, is_mc_dropout=True, **kwargs):
        assert 'kernel_regularizer' not in kwargs
        super(ConcreteDropout, self).__init__(layer, **kwargs)
        self.weight_regularizer  = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.is_mc_dropout       = is_mc_dropout
        self.supports_masking    = True
        self.p_logit             = None
        self.init_min            = np.log(init_min) - np.log(1. - init_min)
        self.init_max            = np.log(init_max) - np.log(1. - init_max)
        
    def build(self, input_shape=None):
        self.input_spec = InputSpec(shape=input_shape)
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        self.built = True
        super(ConcreteDropout, self).build()

        # initialise p
        """self.p_logit = self.layer.add_weight(
                                      name='p_logit',
                                      shape=(1,),
                                      initializer=initializers.RandomUniform(self.init_min, self.init_max),
                                      dtype=tf.dtypes.float32,
                                      trainable=True)"""
        
    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def concrete_dropout(self, x, p):
        '''
        Concrete dropout - used at training time (gradients can be propagated)
        :param x: input
        :return:  approx. dropped out input
        '''
        eps  = K.cast_to_floatx(K.epsilon())
        temp = 0.1

        unif_noise = K.random_uniform(shape=K.shape(x))
        drop_prob = (
            K.log(p + eps)
            - K.log(1. - p + eps)
            + K.log(unif_noise + eps)
            - K.log(1. - unif_noise + eps)
        )
        drop_prob = K.sigmoid(drop_prob / temp)
        random_tensor = 1. - drop_prob

        retain_prob  = 1. - p
        x           *= random_tensor
        x           /= retain_prob
        return x

    def call(self, inputs, training=None):
        p = tf.math.sigmoid(self.p_logit)
        
        # initialise regulariser / prior KL term
        input_dim = int(inputs.shape[-1])  # last dim
        weight = self.layer.kernel
        kernel_regularizer = self.weight_regularizer * tf.reduce_sum(tf.square(weight)) / (1. - p)
        dropout_regularizer = p * tf.math.log(p) + (1. - p) * tf.math.log(1. - p)
        dropout_regularizer *= self.dropout_regularizer * input_dim
        regularizer = tf.reduce_sum(kernel_regularizer + dropout_regularizer)
        
        if self.is_mc_dropout:
            return self.layer.call(self.concrete_dropout(inputs, p)), regularizer
        else:
            def relaxed_dropped_inputs():
                return self.layer.call(self.concrete_dropout(inputs, p)), regularizer
            return K.in_train_phase(relaxed_dropped_inputs,
                                    self.layer.call(inputs),
                                    training=training), regularizer


class SpatialConcreteDropout(Wrapper):
    """From https://github.com/yaringal/ConcreteDropout
    DOES NOT WORK CURRENTLY.  Needs to be updated for TFv2.
    
    This wrapper allows to learn the dropout probability for any given Conv2D input layer.
    ```python
        model = Sequential()
        model.add(SpatialConcreteDropout(Conv2D(64, (3, 3)),
                                  input_shape=(299, 299, 3)))
    ```
    # Arguments
        layer: a layer instance.
        weight_regularizer:
            A positive number which satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$ (inverse observation noise),
            and N the number of instances in the dataset.
            Note that kernel_regularizer is not needed.
        dropout_regularizer:
            A positive number which satisfies
                $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and N the number of
            instances in the dataset.
            Note the relation between dropout_regularizer and weight_regularizer:
                $weight_regularizer / dropout_regularizer = l**2 / 2$
            with prior lengthscale l. Note also that the factor of two should be
            ignored for cross-entropy loss, and used only for the eculedian loss.
    """
    def __init__(self, layer, weight_regularizer=1e-6, dropout_regularizer=1e-5,
                 init_min=0.1, init_max=0.1, is_mc_dropout=True, data_format=None, **kwargs):
        assert 'kernel_regularizer' not in kwargs
        super(SpatialConcreteDropout, self).__init__(layer, **kwargs)
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.is_mc_dropout = is_mc_dropout
        self.supports_masking = True
        self.p_logit = None
        self.p = None
        self.init_min = np.log(init_min) - np.log(1. - init_min)
        self.init_max = np.log(init_max) - np.log(1. - init_max)
        self.data_format = 'channels_last' if data_format is None else 'channels_first'

    def build(self, input_shape=None):
        self.input_spec = InputSpec(shape=input_shape)
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super(SpatialConcreteDropout, self).build()  # this is very weird.. we must call super before we add new losses

        # initialise p
        self.p_logit = self.layer.add_weight(name='p_logit',
                                            shape=(1,),
                                            initializer=initializers.RandomUniform(self.init_min, self.init_max),
                                            trainable=True)
        self.p = K.sigmoid(self.p_logit[0])

        # initialise regulariser / prior KL term
        assert len(input_shape) == 4, 'this wrapper only supports Conv2D layers'
        if self.data_format == 'channels_first':
            input_dim = int(input_shape[1]) # we drop only channels
        else:
            input_dim = int(input_shape[3])
        
        weight = self.layer.kernel
        kernel_regularizer = self.weight_regularizer * K.sum(K.square(weight)) / (1. - self.p)
        dropout_regularizer = self.p * K.log(self.p)
        dropout_regularizer += (1. - self.p) * K.log(1. - self.p)
        dropout_regularizer *= self.dropout_regularizer * input_dim
        regularizer = K.sum(kernel_regularizer + dropout_regularizer)
        self.layer.add_loss(regularizer)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def spatial_concrete_dropout(self, x):
        '''
        Concrete dropout - used at training time (gradients can be propagated)
        :param x: input
        :return:  approx. dropped out input
        '''
        eps = K.cast_to_floatx(K.epsilon())
        temp = 2. / 3.

        input_shape = K.shape(x)
        if self.data_format == 'channels_first':
            noise_shape = (input_shape[0], input_shape[1], 1, 1)
        else:
            noise_shape = (input_shape[0], 1, 1, input_shape[3])
        unif_noise = K.random_uniform(shape=noise_shape)
        
        drop_prob = (
            K.log(self.p + eps)
            - K.log(1. - self.p + eps)
            + K.log(unif_noise + eps)
            - K.log(1. - unif_noise + eps)
        )
        drop_prob = K.sigmoid(drop_prob / temp)
        random_tensor = 1. - drop_prob

        retain_prob = 1. - self.p
        x *= random_tensor
        x /= retain_prob
        return x

    def call(self, inputs, training=None):
        if self.is_mc_dropout:
            return self.layer.call(self.spatial_concrete_dropout(inputs))
        else:
            def relaxed_dropped_inputs():
                return self.layer.call(self.spatial_concrete_dropout(inputs))
            return K.in_train_phase(relaxed_dropped_inputs,
                                    self.layer.call(inputs),
                                    training=training)


class NNModel:
    """
    Builds/trains NN model according to specified parameters.

    __init__: Initialization of the NN model.

    train: Trains the NN model.
    
    Yeval: Makes predictions on the chosen dataset and saves them as NPY files.
    """

    def __init__(self, ftrain_TFR, fvalid_TFR, ftest_TFR,
                 ishape, oshape, ilog, olog,
                 x_mean, x_std, y_mean, y_std,
                 x_min,  x_max, y_min,  y_max, scalelims,
                 ncores, buffer_size, batch_size, nbatches,
                 layers, lay_params, activations, nodes,
                 kernel_regularizer, 
                 lossfunc, lengthscale = 1e-3, max_lr=1e-1,
                 model_predict=None, model_evaluate=None, 
                 clr_mode='triangular', clr_steps=2000,
                 weight_file = 'weights.h5.keras', stop_file = './STOP',
                 train_flag = True,
                 epsilon=1e-6,
                 debug=False, shuffle=False, resume=False):
        """
        ftrain_TFR : list, strings. TFRecords for the training   data.
        fvalid_TFR : list, strings. TFRecords for the validation data.
        ftest_TFR  : list, strings. TFRecords for the test       data.
        ishape     : tuple, ints. Shape of the input  data.
        oshape     : tuple, ints. Shape of the output data.
        ilog       : bool or array of bool.  Determines if the input  values are log10-scaled.
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

        ### Build model
        inp = Input(shape=ishape)
        x = inp
        # Hidden layers
        n = 0 # Counter for layers with nodes
        for i in range(len(layers)):
            if layers[i] in ['conv1d', 'conv2d', 'conv3d', 
                             'conv2dtranspose', 'conv3dtranspose']:
                tshape = tuple(val for val in K.int_shape(x) if val is not None)
                if layers[i] == 'conv1d':
                    Conv = Convolution1D
                    dval = 1
                elif 'conv2d' in layers[i]:
                    if 'transpose' in layers[i]:
                        Conv = Convolution2DTranspose
                    else:
                        Conv = Convolution2D
                    dval = 2
                elif 'conv3d' in layers[i]:
                    if 'transpose' in layers[i]:
                        Conv = Convolution3DTranspose
                    else:
                        Conv = Convolution3D
                    dval = 3
                # Check if a channel must be added for conv features
                if len(tshape) == dval:
                    x = Reshape(tshape + (1,))(x)
                if type(activations[n]) == str:
                    # Simple activation: pass as layer parameter
                    x = Conv(filters=nodes[n],
                             kernel_size=lay_params[i],
                             activation=activations[n],
                             kernel_regularizer=kernel_regularizer,
                             padding='same')(x)
                else:
                    # Advanced activation: use as its own layer
                    x = Conv(filters=nodes[n],
                             kernel_size=lay_params[i],
                             kernel_regularizer=kernel_regularizer,
                             padding='same')(x)
                    x = activations[n](x)
                n += 1
            elif layers[i] in ['dense', 'concretedropout']:
                if i > 0:
                    if layers[i-1] in ['conv1d', 'conv2d', 'conv3d',
                                       'maxpool1d', 'maxpool2d', 'maxpool3d',
                                       'avgpool1d', 'avgpool2d', 'avgpool3d']:
                        print('Dense layer follows a not-Dense layer. Flattening.')
                        x = Flatten()(x)
                elif i==0 and len(ishape) > 1:
                    print("Dense layer follows >1D input layer. Flattening.")
                    x = Flatten()(x)
                if layers[i] == 'dense':
                    if type(activations[n]) == str:
                        x = Dense(nodes[n], activation=activations[n],
                                  kernel_regularizer=kernel_regularizer)(x)
                    else:
                        x = Dense(nodes[n], kernel_regularizer=kernel_regularizer)(x)
                        x = activations[n] (x)
                elif layers[i] == 'concretedropout':
                    x, _ = ConcreteDropout(nodes[n], activation=activations[n], 
                                              weight_regularizer=wd, 
                                              dropout_regularizer=dd,
                                              kernel_regularizer=kernel_regularizer)(x)
                n += 1
            elif layers[i] in ['maxpool1d', 'maxpool2d', 'maxpool3d']:
                if layers[i-1] == 'dense' or layers[i-1] == 'flatten':
                    raise ValueError('MaxPool layers must follow Conv1d or ' \
                                   + 'Pool layer.')
                if layers[i] == 'maxpool1d':
                    x = MaxPooling1D(pool_size=lay_params[i])(x)
                elif layers[i] == 'maxpool2d':
                    x = MaxPooling2D(pool_size=lay_params[i])(x)
                elif layers[i] == 'maxpool3d':
                    x = MaxPooling3D(pool_size=lay_params[i])(x)
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
            elif layers[i] == 'dropout':
                if self.train_flag:
                    x = Dropout(lay_params[i])(x)
            elif layers[i] == 'flatten':
                x = Flatten()(x)
            else:
                raise ValueError("Unknown layer type: " + layers[i])
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
        print(self.model.summary())


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

        ### Train the model
        if self.train_flag:
            # Ensure at least 1 epoch happens when training
            if init_epoch >= epochs:
                print('The requested number of training epochs ('+str(epochs) +\
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
                                             verbose=2,
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
                y_batch = self.model.predict(x_batch)
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
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[igpu], 'GPU')
    if optnlays is not None:
        nlayers = trial.suggest_int("layers", optnlays[0], optnlays[1])
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
        #else:
        #    activs.append('linear')
        #    actval.append(None)
        activations.append(L.load_activation(activs[-1], actval[-1]))
    if None not in [optminlr, optmaxlr]:
        lengthscale = trial.suggest_float("min_lr", optminlr, optmaxlr)
        max_lr = trial.suggest_float("max_lr", lengthscale, optmaxlr)
    lay_params, _ = U.prepare_layers(selected_layers, [None]*nlayers, nnodes)
    try:
        model = NNModel(ftrain_TFR, fvalid_TFR, ftest_TFR,
                        ishape, oshape, ilog, olog,
                        x_mean, x_std, y_mean, y_std,
                        x_min,  x_max, y_min,  y_max, scalelims,
                        ncores, buffer_size, batch_size, nbatches,
                        selected_layers, lay_params, activations, nnodes,
                        kernel_regularizer, 
                        lossfunc=lossfunc, lengthscale=lengthscale, max_lr=max_lr,
                        clr_mode=clr_mode, clr_steps=clr_steps,
                        weight_file=weight_file, train_flag=True,
                        epsilon=1e-6, shuffle=True)
        model.train(epochs, patience, save=False, parallel=True)
        return np.amin(model.historyNN.history['val_loss'])
    except Exception as e:
        print(e)
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
           gridsearch, architectures,
           layers, lay_params, activations, nodes, kernel_regularizer, 
           lossfunc, lengthscale, max_lr, clr_mode, clr_steps,
           model_predict, model_evaluate, 
           epochs, patience,
           weight_file, resume,
           plot_cases, fxvals, xlabel, ylabel, smoothing,
           filters=None, filtconv=1., verb=1):
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
                         `architectures`.
    architectures: list. Model architectures to consider.
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
    verb       : int.    Controls how much output is printed to terminal.
    """
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
            print('\nNormalizing the data...', flush=True)
        else:
            print('\nScaling the data...', flush=True)
        
        try:
            x_mean = np.load(fxmean)
            x_std  = np.load(fxstd)
            y_mean = np.load(fymean)
            y_std  = np.load(fystd)
        except:
            print("Determining the minimum, maximum, mean, and standard " +\
                  "deviation based on Welford's method.", flush=True)
            # Compute stats
            x_mean, x_std, x_min, x_max, \
            y_mean, y_std, y_min, y_max  = S.save_stats_files(glob.glob(datadir + 'train' + os.sep + '*.npz'),
                                                              ilog, olog, ishape, oshape,
                                                              fxmean, fxstd, fxmin,  fxmax,
                                                              fymean, fystd, fymin,  fymax, statsaxes)
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
            if verb:
                print("mean :", x_mean, y_mean)
                print("stdev:", x_std,  y_std)
            if scale:
                x_min = U.normalize(x_min, x_mean, x_std)
                x_max = U.normalize(x_max, x_mean, x_std)
                y_min = U.normalize(y_min, y_mean, y_std)
                y_max = U.normalize(y_max, y_mean, y_std)
        else:
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
            if verb:
                print("min:", x_min, y_min)
                print("max:", x_max, y_max)
    else:
        x_mean = 0.
        x_std  = 1.
        y_mean = 0.
        y_std  = 1.
        x_min     =  0.
        x_max     =  1.
        y_min     =  0.
        y_max     =  1.
        scalelims = [0., 1.]

    if olog:
        # To later properly calculate RMSE & R2 for log-scaled output
        try:
            y_mean_delog = np.load(fymean.replace(".npy", "_delog.npy"))
        except:
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
        print("\nPerforming a grid search.\n", flush=True)
        maxlen = 0
        for i, arch in enumerate(architectures):
            if len(arch) > maxlen:
                maxlen = len(arch)
            archdir = os.path.join(outputdir, arch, '')
            wsplit  = weight_file.rsplit(os.sep, 1)[1].rsplit('.', 1)
            wfile   = ''.join([archdir, wsplit[0], '_', arch, '.', wsplit[1]])
            U.make_dir(archdir)
            nn = NNModel(ftrain_TFR, fvalid_TFR, ftest_TFR,
                         ishape, oshape, ilog, olog,
                         x_mean, x_std, y_mean, y_std,
                         x_min,  x_max, y_min,  y_max, scalelims,
                         ncores, buffer_size, batch_size,
                         [train_batches, valid_batches, test_batches],
                         layers[i], lay_params[i],
                         activations[i], nodes[i], kernel_regularizer, 
                         lossfunc, lengthscale, max_lr, model_predict, model_evaluate, 
                         clr_mode, clr_steps,
                         wfile, stop_file='./STOP', resume=resume,
                         train_flag=True, shuffle=True)
            nn.train(epochs, patience)
            P.loss(nn, archdir)
        # Print/save out the minmium validation loss for each architecture
        minvl = np.ones(len(architectures)) * np.inf
        print('Grid search summary')
        print('-------------------')
        with open(os.path.join(outputdir, 'gridsearch.txt'), 'w') as foo:
            foo.write('Grid search summary\n')
            foo.write('-------------------\n')
        for i, arch in enumerate(architectures):
            archdir  = os.path.join(outputdir, arch, '')
            history  = np.load(archdir+'history.npz')
            minvl[i] = np.amin(history['val_loss'])
            print(arch.ljust(maxlen, ' ') + ': ' + str(minvl[i]))
            with open(os.path.join(outputdir, 'gridsearch.txt'), 'a') as foo:
                foo.write(arch.ljust(maxlen, ' ') + ': ' \
                          + str(minvl[i]) + '\n')
        return

    # Hyperparameter optimization
    if optimize:
        print("\nRunning hyperparameter optimization:", flush=True)
        print("  Number of trials per GPU:", optimize)
        print("  Number of layers:", optnlays[0], '-', optnlays[1])
        print("  Number of nodes per layer:", optnnode)
        print("  Activation functions:", optactiv)
        if optactrng is not None:
            print("  Activation parameter range:", optactrng[0], '-', optactrng[1])
        if None not in [optminlr, optmaxlr]:
            print("  Minimum learning rate:", optminlr)
            print("  Maximum learning rate:", optmaxlr)
        if opttime is not None:
            print("  Time limit:", opttime/60./60., "hrs")
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
        print("\nHyperparameter optimization complete.  Best trial:", flush=True)
        for key, value in study.best_trial.params.items():
            print("  {}: {}".format(key, value))
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
        print('\nBeginning model training.\n', flush=True)
        nn = NNModel(ftrain_TFR, fvalid_TFR, ftest_TFR,
                     ishape, oshape, ilog, olog,
                     x_mean, x_std, y_mean, y_std,
                     x_min,  x_max, y_min,  y_max, scalelims,
                     ncores, buffer_size, batch_size,
                     [train_batches, valid_batches, test_batches],
                     layers, lay_params, activations, nodes, kernel_regularizer, 
                     lossfunc, lengthscale, max_lr, model_predict, model_evaluate, 
                     clr_mode, clr_steps,
                     weight_file, stop_file='./STOP',
                     train_flag=True, shuffle=True, resume=resume)
        nn.train(epochs, patience)
        # Plot the loss
        P.loss(nn, plotdir)
    
    if not os.path.exists(weight_file):
        # No model trained this time or previously
        if verb:
            print("No model file found.  Exiting.", flush=True)
        return

    # Call new model with shuffle=False
    # lossfunc is set to None because we will not be training it, and
    # certain loss functions would throw an error here due to undefined shapes
    if lossfunc is not None:
        if 'heteroscedastic' not in lossfunc.__name__:
            lossfunc = None
    nn = NNModel(ftrain_TFR, fvalid_TFR, ftest_TFR,
                 ishape, oshape, ilog, olog,
                 x_mean, x_std, y_mean, y_std,
                 x_min,  x_max, y_min,  y_max, scalelims,
                 ncores, buffer_size, batch_size,
                 [train_batches, valid_batches, test_batches],
                 layers, lay_params, activations, nodes, kernel_regularizer, 
                 lossfunc, lengthscale, max_lr, model_predict, model_evaluate, 
                 clr_mode, clr_steps,
                 weight_file, stop_file='./STOP',
                 train_flag=False, shuffle=False, resume=False)
    if '.keras' in weight_file:
        nn.model.load_weights(weight_file) # Load the trained model
        # Save in ONNX format
        try:
            _ = tf2onnx.convert.from_keras(nn.model, input_signature=np.zeros((batch_size,) + ishape, dtype=float), output_path=weight_file.replace('.keras', '.onnx'))
        except Exception as e:
            print("Unable to convert the Keras model to ONNX:")
            print(e)
    #else:
    #    nn.model = onnx_to_keras(onnx.load_model(weight_file), ['input_1'])

    # Validate model
    if (validflag or trainflag) and not rng_test:
        print('\nValidating the model...\n', flush=True)
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
                                  filters=filters, filtconv=filtconv)
        else:
            val_stats = S.rmse_r2(fvalpred, fvaltrue, y_mean,
                                  y_std, y_min, y_max, scalelims,
                                  olog, y_mean_delog,
                                  xvals, filters, filtconv)
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
            try:
                P.plot(''.join([plotdir, rmse_file, '_val_norm.png']),
                       xvals, val_stats[0], xlabel, 'RMSE')
            except:
                print("Normalized RMSE unable to be plotted.")
            np.savez(outputdir+rmse_file+'_val_norm.npz',
                     rmse=val_stats[0], rmse_mean=np.mean(val_stats[0]))
        if saveRMSEdenorm:
            try:
                P.plot(''.join([plotdir, rmse_file, '_val_denorm.png']),
                       xvals, val_stats[1], xlabel, 'RMSE')
            except:
                print("Denormalized RMSE unable to be plotted.")
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
            try:
                P.plot(''.join([plotdir, r2_file, '_val_norm.png']),
                       xvals, val_stats[2], xlabel, '$R^2$')
            except:
                print("Normalized R2 unable to be plotted.")
            np.savez(outputdir+r2_file+'_val_norm.npz',
                     r2=val_stats[2], r2_mean=np.mean(val_stats[2]))
        if saveR2denorm:
            try:
                P.plot(''.join([plotdir, r2_file, '_val_denorm.png']),
                       xvals, val_stats[3], xlabel, '$R^2$')
            except:
                print("Denormalized R2 unable to be plotted.")
            np.savez(outputdir+r2_file+'_val_denorm.npz',
                     r2=val_stats[3], r2_mean=np.mean(val_stats[3]))

    # Evaluate model on test set
    if testflag and not rng_test:
        print('\nTesting the model...\n', flush=True)
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
                                   filters=filters, filtconv=filtconv)
        else:
            test_stats = S.rmse_r2(ftestpred, ftesttrue, y_mean,
                                   y_std, y_min, y_max, scalelims,
                                   olog, y_mean_delog,
                                   xvals, filters, filtconv)
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
            try:
                P.plot(''.join([plotdir, rmse_file, '_test_norm.png']),
                       xvals, test_stats[0], xlabel, 'RMSE')
            except:
                print("Normalized RMSE unable to be plotted.")
            np.savez(outputdir+rmse_file+'_test_norm.npz',
                     rmse=test_stats[0], rmse_mean=np.mean(test_stats[0]))
        if saveRMSEdenorm:
            try:
                P.plot(''.join([plotdir, rmse_file, '_test_denorm.png']),
                       xvals, test_stats[1], xlabel, 'RMSE')
            except:
                print("Denormalized RMSE unable to be plotted.")
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
            try:
                P.plot(''.join([plotdir, r2_file, '_test_norm.png']),
                       xvals, test_stats[2], xlabel, '$R^2$')
            except:
                print("Normalized R2 unable to be plotted.")
            np.savez(outputdir+r2_file+'_test_norm.npz',
                     r2=test_stats[2], r2_mean=np.mean(test_stats[2]))
        if saveR2denorm:
            try:
                P.plot(''.join([plotdir, r2_file, '_test_denorm.png']),
                       xvals, test_stats[3], xlabel, '$R^2$')
            except:
                print("Denormalized R2 unable to be plotted.")
            np.savez(outputdir+r2_file+'_test_denorm.npz',
                     r2=test_stats[3], r2_mean=np.mean(test_stats[3]))

    # Plot requested cases
    if not rng_test and plot_cases is not None:
        predfoo = sorted(glob.glob(preddir+'test'+os.sep+'pred*'))
        truefoo = sorted(glob.glob(preddir+'test'+os.sep+'true*'))
        if len(predfoo) > 0 and len(truefoo) > 0:
            print("\nPlotting the requested cases...", flush=True)
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
            raise Exception("No predictions found in " + preddir + "test.")

    return nn
