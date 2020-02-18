"""
Module that contains functions related to loading/reading files.

load_data_file: Loads a given .NPY file.

"""

import sys, os
import numpy as np


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


