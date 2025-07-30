"""
Module that contains functions related to loading/reading files.

load_data_file: Loads a given .NPY file.

load_activations: Takes the user-specified parameters and loads the
                  activation object.

"""

import sys, os
import logging
logging.setLoggerClass(logging.Logger)
import types
import numpy as np

import tensorflow.keras as keras
K = keras.backend
from tensorflow.keras.layers import ReLU, LeakyReLU, ELU, Softmax

import custom_logger as CL
logging.setLoggerClass(CL.MARGE_Logger)

logger = logging.getLogger('MARGE.'+__name__)


def load_data_file(data_file, ilog=False, olog=False):
    """
    Loads a given .NPY file.

    Inputs
    ------
    data_file: string. .NPY file to load
    ilog     : bool.   Determines whether to take the log of inputs  or not.
    olog     : bool.   Determines whether to take the log of outputs or not.

    Outputs
    -------
    x: array. Input data features.
    y: array. Output parameters to predict.

    """
    # Load file
    data = np.load(data_file)
    x = data['x']
    y = data['y']

    # Ensure at least 2D
    if x.ndim == 1:
        x = x[None, :]
    if y.ndim == 1:
        y = y[None, :]

    # Take log10 as needed
    if ilog:
        x[..., ilog] = np.log10(x[..., ilog])
    if olog:
        y[..., olog] = np.log10(y[..., olog])

    return x, y


def load_activation(act_str, act_par):
    """
    Loads an activation object.

    Inputs
    ------
    act_str: string. Desired acivation function.
    act_par: string. Parameter for activation function.

    Outputs
    -------
    activation: Activation object, or string that maps to it.
    """
    if act_str in ['exponential', 'exp']:
        activation = 'exponential'
    elif act_str in ['sigmoid', 'sig']:
        activation = 'sigmoid'
    elif act_str == 'relu':
        if act_par in [None, 'None']:
            logger.debug("No ReLU activation parameter provided; using the default behavior (no max)")
            activation = ReLU() #default: no max
        else:
            activation = ReLU(float(act_par))
        activation.str = 'relu'
        if activation.max_value is not None:
            activation.str += str(activation.max_value)[:min(len(str(activation.max_value)), 5)]
    elif act_str in ['leaky_relu', 'leakyrelu', 'lrelu']:
        if act_par in [None, 'None']:
            logger.debug("No Leaky ReLU activation parameter provided; using the default behavior (0.3)")
            activation = LeakyReLU(0.3) #default: 0.3
        else:
            activation = LeakyReLU(float(act_par))
        activation.str = 'lrelu' + str(activation.negative_slope)[:min(len(str(activation.negative_slope)), 5)]
    elif act_str == 'elu':
        if act_par in [None, 'None']:
            logger.debug("No ELU activation parameter provided; using the default behavior (0.1)")
            activation = ELU(0.1) #default: 0.1
        else:
            activation = ELU(float(act_par))
        activation.str = 'elu' + str(activation.alpha)[:min(len(str(activation.alpha)), 5)]
    elif act_str in ['linear', 'lin', None, 'None', 'identity']:
        activation = 'linear'
    elif act_str in ['hard_sigmoid', 'hardsigmoid', 'hardsig']:
        activation = 'hard_sigmoid'
    elif act_str in ['hard_silu', 'hardsilu', 'hard_swish', 'hardswish']:
        activation = 'hard_silu'
    elif act_str in ['hard_tanh', 'hardtanh']:
        activation = 'hard_tanh'
    elif act_str in ['log_sigmoid', 'logsigmoid', 'logsig']:
        activation = 'log_sigmoid'
    elif act_str in ['log_softmax', 'logsoftmax']:
        activation = 'log_softmax'
    elif act_str in ['silu', 'swish']:
        activation = 'silu'
    elif act_str in ['sparse_plus', 'sparseplus']:
        activation = 'sparse_plus'
    elif act_str in ['sparsemax', 'sparse_max']:
        activation = 'sparsemax'
    elif act_str in ['tanh_shrink', 'tanhshrink']:
        activation = 'tanh_shrink'
    elif act_str in ['gelu', 'glu', 'mish', 'relu6', 'selu', \
                     'softmax', 'softplus', 'softsign', 'tanh']:
        activation = act_str
    else:
        raise Exception('Activation function not understood: ' + act_str)

    return activation
