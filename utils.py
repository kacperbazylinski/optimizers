"""
This file contains all functions and classes
to perform Linear Regression in numpy
"""
import numpy as np


def mse_loss(y_pred, y_true):
    """
    Implementation of mean squared error i numpy
    :param y_pred: array with predicted values
    :type y_pred: np.array
    :param y_true: array with ground truth values
    :type y_true: np.array
    :return: loss
    :rtype: float
    """
    return 0.5 * np.mean(
        np.power(
            (y_pred - y_true),
            2
        )
    )


def calc_gradient_descent(y_pred, y_true, X):
    """
    Gradient for L2 loss
    :param y_pred: array with predicted values
    :type y_pred: np.array
    :param y_true: array with ground truth values
    :type y_true: np.array
    :param X: features of data points
    :type X: np.nd.array()
    :return:
    """
    return np.mean(
        (y_pred - y_true) * X
    )
