"""
Module containing utilities functions like finding sigmoid of the matrix or vector or value.
"""

import numpy as np


def sigmoid(z):
    """
    Implements sigmoid function
    :param z: array or matrix or a number
    :return: sigmoid of the array or matrix or a number applied elementwise
    """
    return 1 / (1 + np.exp(-z))


def OneHotEncoder(data):
    """
    Returns one-hot encoded data
    :param data: numpy matrix/vector where each row represents each value to be encoded
    :return: one-hot encoded matrix
    """
    data = np.array(data).reshape(-1, 1)
    encoding = np.zeros((data.shape[0], 10))
    for i, j in enumerate(data[:, 0]):
        encoding[i, j] = 1
    return encoding