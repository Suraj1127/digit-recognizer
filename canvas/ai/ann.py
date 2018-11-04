#!/usr/bin/env python3

"""
Author: Suraj Regmi
Date: 8th June, 2018
Description: This module implements ANN(Artificial Neural Networks) from scratch.
"""

import random

import pickle
import csv

import numpy as np

from .utilities.util_functions import sigmoid, OneHotEncoder


class ANN:
    """
    Class to describe the artificial deep neural network.
    Stochastic Gradient Descent(SGD) is used here for training.

    ----------------------------------------------------

    Class variables:
    epsilon: used for preventing errors during numerical computation like ZeroDivideError, nan problems, etc.

    """

    epsilon = 10**(-6)

    def __init__(self, input_variables, output_variables, nodes_array):
        """
        :param input_variables: no of variables in an input training data
        :param output_variables: no of variables in an output label
        :param nodes_array: array of nodes in the neural network layers(hidden layers and output layer)

        ------------------------------------------------

        Instance variables:
        n_x, n_y and nodes_array as described above
        L: depth of deep neural network
        w: weight dictionary
        b: bias dictionary
        z: dictionary of z values which are fed into sigmoid function
        a: dictionary of activation values
        nabla: dictionary of gradients
        """
        self.n_x = input_variables
        self.n_y = output_variables

        self.nodes_array = nodes_array

        self.L = len(nodes_array)

        self.w = self.b = None

        self._initialize_parameters()

        self.z = dict()
        self.a = dict()

        self.nabla = {}


    def _initialize_parameters(self):
        """
        Initializes parameters to some random values.  Concept of Xavier initialization used.
        """
        self.w = dict()
        self.b = dict()

        for i, j in enumerate(self.nodes_array):
            if i == 0:
                self.w[i+1] = np.random.randn(j, self.n_x) / np.sqrt(j)
            else:
                self.w[i+1] = np.random.randn(j, self.nodes_array[i-1]) / np.sqrt(j)
            self.b[i+1] = np.random.randn(j, 1)


    def _forward_propagate(self, x):
        """
        Propagates forward and sets activation values for the training example.
        :param x: a training example.
        """

        self.a[0] = x
        for i, j in enumerate(self.nodes_array):
            self.z[i+1] = np.matmul(self.w[i+1], self.a[i]) + self.b[i+1]
            self.a[i+1] = sigmoid(self.z[i+1])


    def _compute_gradients(self, lambd, y):
        """
        Computes gradients by using the training example and sets to dictionary, nabla.
        :param lambd: regularization parameter
        :param y: labels of the training example.
        """
        self.nabla['a' + '(' + str(self.L) + ')'] = \
            (self.a[self.L] - y)/((self.a[self.L] + self.epsilon) * (1 - self.a[self.L] + self.epsilon))

        for k in range(self.L, 0, -1):
            self.nabla['a' + '(' + str(k) + ')'] = self.nabla['a' + '(' + str(k) + ')'] * (self.a[k] * (1 - self.a[k]))
            self.nabla['b' + '(' + str(k) + ')'] = self.nabla['a' + '(' + str(k) + ')'] + 2 * lambd * self.b[k]
            self.nabla['w' + '(' + str(k) + ')'] = np.matmul(self.nabla['a' + '(' + str(k) + ')'], self.a[k-1].T) \
                                                   + 2 * lambd * self.w[k]
            self.nabla['a' + '(' + str(k-1) + ')'] = np.matmul(self.w[k].T, self.nabla['a' + '(' + str(k) + ')'])

    def _learn(self, learning_rate):
        """
        Learns the parameters using the computed gradients.
        """
        for i in range(1, self.L + 1):
            self.w[i] = self.w[i] - learning_rate * self.nabla['w' + '(' + str(i) + ')']
            self.b[i] = self.b[i] - learning_rate * self.nabla['b' + '(' + str(i) + ')']

    def sgd_train(self, x, y, learning_rate, lambd):
        """
        A pass of training.
        :param x: input of a training data
        :param y: output/label of a training data
        :param learning_rate: learning rate of the SGD
        :param lambd: regularization parameter
        """
        self._forward_propagate(x)
        self._compute_gradients(lambd, y)
        self._learn(learning_rate)

    def train(self, filename, no_of_epochs):
        """
        Trains the ANN model from the input data.
        :param filename: relative path of training data.
        :param no_of_epochs: no of epochs for the training.
        """

        # Reading the file to find out the size of given data.
        with open(filename) as file:
            self.m = sum(True for _ in file) - 1

        # Setting 10% of the training data as test data.
        self.test_indices = set(random.sample(range(self.m), int(0.1 * self.m)))
        self.test_data = None

        # Training the data
        for epoch in range(1, no_of_epochs + 1):

            # Training by using csv reader.  That loads each training data and learns using SGD.
            # It would not require large memory for the training.
            with open(filename, 'r', newline='\n') as csvfile:
                reader = csv.reader(csvfile)
                for i, line in enumerate(reader):

                    if line[0] == 'label':
                        continue

                    training_example = list(map(int, line))

                    # Ignoring the data to be tested later while training.
                    if i in self.test_indices:
                        if epoch == 1:
                            if self.test_data is None:
                                self.test_data = np.array(training_example).reshape(1, -1)
                            else:
                                self.test_data = np.concatenate((self.test_data, np.array(training_example).reshape(1, -1)))
                        else:
                            pass
                        continue

                    x = np.array(training_example[1:]).reshape(-1, 1) / 255
                    y = OneHotEncoder(training_example[0]).reshape(-1, 1)

                    self.sgd_train(x, y, 0.01, 0)

            # Setting the test data to the class instance
            if epoch == 1:
                self.x_test = self.test_data[:, 1:] / 255
                self.y_test = OneHotEncoder(self.test_data[:, 0])

            # Completion of a epoch and printing the evaluation parameters (cross entropy loss and accuracy percentage).
            print('Epoch {} completed.'.format(epoch))
            self.evaluate()


    def predict(self, X):
        """
        Given the array X, returns the predicted values of Y.
        :param X: dimension => no of samples by no of input variables
        :return: predicted values of Y from given data X, dimension => no of samples by no of output variables
        """
        self.a[0] = X
        for i, j in enumerate(self.nodes_array):
            self.z[i + 1] = np.matmul(self.a[i], self.w[i + 1].T) + self.b[i + 1].T
            self.a[i + 1] = sigmoid(self.z[i + 1])
        return self.a[i + 1]

    def get_cost(self, y, y_pred):
        """
        Returns cross-entropy loss from labels(y) and predicted values(y_pred).
        """
        return - np.sum(np.sum(y * np.log(y_pred + 10**(-2)) + (1 - y)*np.log(1-y_pred + 10**(-2)), axis=1)) / y.shape[0]

    def evaluate(self):
        """
        Calculates cross-entropy cost and accuracy in the 10% test data set in the instance.
        """
        y_actual = self.y_test
        y_pred = self.predict(self.x_test)
        print('Cost: {}'.format(self.get_cost(y_actual, y_pred)))

        y_actual = np.argmax(y_actual, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        print('Accuracy: {}%'.format(100 * sum(y_actual == y_pred)/self.test_data.shape[0]))



def main():

    # Creates instance of ANN by using the no of input parameters,
    # no of output parameters and architecture of hidden nodes.
    ocr_ann_model = ANN(784, 10, (800, 10))

    # Train the model from the file train/train.csv by using 10 epochs.
    ocr_ann_model.train('train/train.csv', 10)

    # save weights and bias as pickle files
    with open('parameters/weights.pickle', 'wb') as f:
        pickle.dump(ocr_ann_model.w, f, pickle.HIGHEST_PROTOCOL)

    with open('parameters/bias.pickle', 'wb') as f:
        pickle.dump(ocr_ann_model.b, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()