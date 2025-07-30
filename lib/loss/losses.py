import numpy as np
import tensorflow as tf
#import tensorflow.keras as keras
keras = tf.keras
K = keras.backend
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops


def mslse(y_true, y_pred, alpha=5.):
    """
    Mean-squared log-scaled error loss
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    first_log = math_ops.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = math_ops.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.mean(math_ops.squared_difference(y_pred, y_true), axis=-1) + alpha * K.mean(math_ops.squared_difference(first_log, second_log), axis=-1)


def maxmse(y_true, y_pred, maxax=1):
    sh = y_true.get_shape().as_list()
    ax = np.arange(len(sh), dtype=int)
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    return K.max(K.mean(tf.math.squared_difference(y_pred, y_true), axis=tuple(ax[maxax+1:])), axis=maxax)


def m3se(y_true, y_pred, maxax=1):
    sh = y_true.get_shape().as_list()
    ax = np.arange(len(sh), dtype=int)
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    maxmse = K.max (K.mean(tf.math.squared_difference(y_pred, y_true), axis=tuple(ax[maxax+1:])), axis=maxax)
    mse    = K.mean(K.mean(tf.math.squared_difference(y_pred, y_true), axis=tuple(ax[1:])))
    return mse + maxmse


def mse_per_ax(y_true, y_pred, mseax=1):
    sh = y_true.get_shape().as_list()
    ax = np.arange(len(sh), dtype=int)
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    return K.mean(K.mean(tf.math.squared_difference(y_pred, y_true), axis=tuple(ax[mseax+1:])), axis=mseax)

def maxse(y_true, y_pred):
    sh = y_true.get_shape().as_list()
    ax = np.arange(len(sh), dtype=int)
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    return K.max(tf.math.squared_difference(y_pred, y_true), axis=tuple(ax[1:]))


def heteroscedastic_loss(true, pred, D, N):
    mean = pred[..., :D]
    L    = pred[..., D:]
    #N    = true.get_shape().as_list()[0] # tf.shape(true)[0]
    # Slow:
    k    = 1
    inc  = 0
    Z    = []
    diag = []
    for d in range(D):
        if k == 1:
            Z.append(tf.concat([tf.exp(tf.reshape(L[:, inc:inc+k], 
                                                  [N, k])), 
                                tf.zeros((N, D-k))],1))
        else:
            Z.append(tf.concat([tf.reshape(L[:, inc:inc+k-1], 
                                           [N, k-1]), 
                                tf.exp(tf.reshape(L[:, inc+k-1],
                                                  [N, 1])), 
                                tf.zeros((N, D-k))],1))
        diag.append(K.exp(L[:,inc+k-1]))
        inc += k
        k   += 1
    # Shape magic for matrix multiplication
    diag  = tf.concat(tf.expand_dims(diag,-1), -1)
    lower = tf.reshape(tf.concat(Z, -1), [N, D, D])
    S_inv = tf.matmul(lower, tf.transpose(lower, perm=[0,2,1]))
    x     = tf.expand_dims((true - mean), -1)
    quad  = tf.matmul(tf.matmul(tf.transpose(x, perm=[0,2,1]), S_inv), x)
    quad  = tf.squeeze(quad, -1)
    # Ensure you do not take the log of 0! +epsilon
    log_det = -2 * K.sum(K.log(diag+K.epsilon()),0)
    # - 0.5 * [log det + quadratic term] = log likelihood 
    # remove minus sign as we want to minimise NLL

    return K.mean(quad + log_det, 0)

def smape(y_true, y_pred):
    epsilon = 1e-8  # Small constant to prevent division by zero
    numerator = tf.abs(y_pred - y_true)
    denominator = tf.abs(y_true) + tf.abs(y_pred) + epsilon
    return K.mean(numerator / denominator) * 200

def maxsape(y_true, y_pred):
    epsilon = 1e-8  # Small constant to prevent division by zero
    numerator = tf.abs(y_pred - y_true)
    denominator = tf.abs(y_true) + tf.abs(y_pred) + epsilon
    return K.max(numerator / denominator) * 200
