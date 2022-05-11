# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 18:55:46 2021

@author: AsafD

THIS IS A FIRST PERCEPTRON PROJECT, TO CLASSIFY POINTS ABOVE OR BELOW A LINE\
    
KNOBS ARE:
    Learning rate
    number of iteration
    number of training data 
 
"""
import numpy as np;
from Perceptron import Perceptron;
from TrainingPoints import Point;
from matplotlib import pyplot as plt;
import time

brain = Perceptron(2)
points = []  # List of Points()

fig = plt.figure()  # The plotting figure

# Create training data and plot them. Range is number of points for training
for i in range(10):
    points.append(Point())


# Training the perceptron showing ERROR convergence.
numberOFIterations = 30
for i in range(numberOFIterations):
    for j in range(len(points)):
        pnt = points[j]
        error = brain.train(pnt.pt, pnt.label)
        plt.scatter(j+(i*len(points)), np.abs(error), marker="o", color='green')

# SHOW plot
# plt.plot(np.array([-1, 1]), np.array([-1, 1]))
plt.show()
time.sleep(1)
