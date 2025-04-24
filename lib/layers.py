"""
Custom layers go here

ConcreteDropout: Implements the concrete dropout method of 
                 Gal, Hron, & Kendall (2017).

SpatialConcreteDropout: Like above, but for convolutional layers.
                        NOTE: Currently does not work with the latest version 
                              of Keras.  It will be fixed in a future update.
"""

import sys, os
import numpy as np

import tensorflow as tf
# Keras
import tensorflow.keras as keras
K = keras.backend
from tensorflow.keras.layers  import (InputSpec, Dense,
                           Conv1D,    Conv2D,    Conv3D,
                           Conv2DTranspose, Conv3DTranspose, 
                           SeparableConv1D, SeparableConv2D, 
                           DepthwiseConv1D, DepthwiseConv2D, 
                           Dropout, Layer, Wrapper, concatenate)
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras import activations as tfactivations
from tensorflow.keras import regularizers
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import nn_ops


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

  @classmethod
  def from_config(cls, config):
    return cls(**config)


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

