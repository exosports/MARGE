"""
Module that contains functions related to loading/reading files.

load_data_file: Loads a given .NPY file.

load_activations: Takes the user-specified parameters and loads the 
                  activation object.

"""

import sys, os
import numpy as np

import keras
from keras import backend as K
from keras.layers import ReLU, LeakyReLU, ELU, Softmax


def load_data_file(data_file, inD, ilog=False, olog=False):
    """
    Loads a given .NPY file.

    Inputs
    ------
    data_file: string. .NPY file to load
    inD      : int.    Dimension of inputs.
    ilog     : bool.   Determines whether to take the log of inputs  or not.
    olog     : bool.   Determines whether to take the log of outputs or not.

    Outputs
    -------
    x: array. Input data features.
    y: array. Output parameters to predict.

    """
    # Load file
    data = np.load(data_file)
    
    # Ensure 2D
    if data.ndim == 1:
        data = data[None, :]

    # Slice inputs/outputs
    x = data[:, :inD]
    y = data[:, inD:]

    # Take log10 as needed
    if ilog:
        x = np.log10(x)
    if olog:
        y = np.log10(y)

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
    if act_str == 'exp':
        activation = 'exponential'
    elif act_str == 'sig':
        activation = 'sigmoid'
    elif act_str == 'relu':
        if act_par == 'None':
            activation = ReLU() #default: no max
        else:
            activation = ReLU(float(act_par))
    elif act_str == 'leakyrelu':
        if act_par == 'None':
            activation = LeakyReLU() #default: 0.3
        else:
            activation = LeakyReLU(float(act_par))
    elif act_str == 'elu':
        if act_par == 'None':
            activation = ELU() #default: 0.1
        else:
            activation = ELU(float(act_par))
    elif act_str == 'softmax':
        activation = Softmax()
    elif act_str == 'None'     \
      or act_str == 'identity' \
      or act_str == 'linear':
        activation = 'linear'
    elif act_str == 'tanh':
        activation = 'tanh'
    else:
        raise Exception('Activation function not ' \
                      + 'understood: ' + act_str)

    return activation
