# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 11:59:13 2021

@author: asafd
"""
import numpy as np;

class Perceptron:
    
    #Constructor
    def __init__(self, numOfInputs):
        self.numOfInputs = numOfInputs;
        #define learning rate
        self.lr = 0.01;        
        #initialize the weights
        self.weights = np.random.rand(numOfInputs,1) *2 -1;
    
    # Activation function - sigmoid, sign etc..    
    def activate(self, y):
        return np.sign(y)
    
    # Recieves inputs and returns outputs    
    def predict(self, inputs):
        self.inputs = inputs;
        y  = np.dot(self.weights.T,self.inputs);
        return self.activate(y);
    
    # Training algorithm
    def train(self, inputs, target):
        guess = self.predict(inputs);
        guess = int(guess);
        print("Guess is:", guess)
        print("Target is:", target)
        error = np.subtract(target, guess)
        print("error is:", error)
        self.weights = np.add(self.weights, np.multiply(error,inputs)*self.lr) #Need to add learning rate
        return error;
########### TEST

#p = Perceptron(2)
# print("Weights are:", p.weights)

# #Test prediction:
# inputs = np.array([[-1,-1]], float);
# print("prediction: ",  p.predict(inputs), "Should be: -1") 

# inputs = np.array([[1,1]], float);
# print("prediction: ",  p.predict(inputs), "Should be: 1")

# a = np.ones([3,1]);
# print("this is a", a)
# print(a.shape)
# b = np.zeros([1,3]);
# print("this is b", b);
# b = b.reshape(a.shape);
# print();
# print(a*b)

#p.train(np.array([0.5,0]),1)