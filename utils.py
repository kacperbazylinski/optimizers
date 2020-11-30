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


def init_weights(X):
    """weight initialization"""
    return np.random.rand(
        X.shape[1]
    ) * np.sqrt(2)


def init_bias():
    """bias initialization"""
    return np.random.rand(1) * np.sqrt(2)


def calc_gradient_descent(y_pred, y_true, X):
    """
    Gradient for L2 loss
    :param y_pred: array with predicted values
    :type y_pred: np.array
    :param b: bias
    :type b: int, float
    :param y_true: array with ground truth values
    :type y_true: np.array
    :param X: features of data points
    :type X: np.nd.array()
    :return:
    """
    weight_gradient = np.matmul(
        y_pred - y_true,
        X
    ) / y_true.shape[1]

    bias_gradient = np.mean(
        (y_pred - y_true)
    )
    return weight_gradient, bias_gradient


def update_w_and_b(weigth, bias,
                   learning_rate,
                   weight_gradient,
                   bias_gradient):
    """
    updates weights, biases for next iteration
    :param weigth: machine's weights
    :param bias: bias value
    :param learning_rate: learning rate
    :param weight_gradient: gradient calculated from previous iteration
    :param bias_gradient: gradient calculated from last iteration
    :return: weight_new, bias_new
    :rtype: np.array, float
    """
    weight_new = weigth - learning_rate * weight_gradient
    bias_new = bias - learning_rate * bias_gradient
    return weight_new, bias_new


class LinearRegression:
    LEARNING_RATE = 0.01
    GRADIENTS_HISTORY = list()
    LOSS_HISTORY = list()
    W = 0
    b = 0

    def __init__(self, learning_rate=LEARNING_RATE):
        self.learning_rate = learning_rate

    def predict(self, X):
        """
        Makes prediction for regression
        :param W: Weights
        :param bias: bias value fo machine
        :type bias: int, float
        :param X: feature matrix of samples
        :type X: np.nd.array
        :return: np.dot(W, X.T) + bias
        :rtype: float
        """
        return np.dot(self.W, X.T) + self.b

    def fit(self, X, y_true, iteration=100, method='bach_gd'):
        """
        fits machine
        :param y_true: ground truth values
        :param method:
        :param iteration: number of iterations
        :param X: features
        :param y: ground truth values
        :return: weight, bias
        """
        self.W = init_weights(X)
        self.b = init_bias(X)

        if method == 'bach_gd':
            for _ in range(0, iteration):
                y_pred = self.predict(X)
                loss = mse_loss(y_pred, y_true)
                self.LOSS_HISTORY.append(loss)

                W_gr, b_gr = calc_gradient_descent(
                    y_pred,
                    y_true,
                    X
                )
                self.GRADIENTS_HISTORY.append([W_gr, b_gr])

                self.W, self.b = update_w_and_b(
                    self.W,
                    self.b,
                    self.learning_rate,
                    W_gr,
                    b_gr
                )
        else:
            print('No more optimizers yet implemented')
