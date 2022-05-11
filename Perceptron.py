# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 11:59:13 2021

@author: asafd
"""
import numpy as np


class Perceptron:

    # Constructor
    def __init__(self, numOfInputs):
        self.numOfInputs = numOfInputs
        # define learning rate
        self.lr = 0.01
        # initialize the weights
        self.weights = np.random.rand(numOfInputs, 1) * 2 - 1

    # Activation function - sigmoid, sign etc..
    def activate(self, y):
        return np.sign(y)

    # Receives inputs and returns the sign of the dot product with the weights
    def predict(self, inputs):
        y = np.dot(self.weights.T, inputs)
        return self.activate(y)

    # Training algorithm
    def train(self, inputs, target):
        guess = self.predict(inputs)
        # print("Guess is:", guess)
        # print("Target is:", target)
        error = np.subtract(target, guess)
        # print("error is:", error)
        self.weights = np.add(self.weights, np.multiply(error, inputs) * self.lr)
        return error
