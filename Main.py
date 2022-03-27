# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 18:55:46 2021

@author: asafd

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

brain = Perceptron(2);
points = []  # List of Points()

fig = plt.figure()  # The plotting figure

# Create training data and plot them. Range is number of points for training
for i in range(10):
    points.append(Point())
    # print(point[i].pt,point[i].label)

    if points[i].label == 1:
        plt.scatter(points[i].pt[0], points[i].pt[1], marker="o", edgecolors="green", color='green')  # x>y
    else:
        plt.scatter(points[i].pt[0], points[i].pt[1], marker="o", edgecolor="black", color='black')  # x<y

# Plot line of X=Y
plt.plot(np.array([-1, 1]), np.array([-1, 1]))

# GUESS: You can set a point and let the untrained machine guess. black means wrong, green means right.
# for pnt in points:
#     x = pnt.pt[0]
#     y = pnt.pt[1]
#     prediction = brain.predict(pnt.pt)
#     if prediction == pnt.label:
#         plt.scatter(x, y, marker="o", color="green")
#     else:
#         plt.scatter(x, y, marker="o", color="black")
#     print(brain.predict(pnt.pt) == pnt.label)

# Training the perceptron
numberOFItrations = 10
for i in range(numberOFItrations):
    numOfErrors = 0
    for pnt in points:
        error = brain.train(pnt.pt, pnt.label);
        if error == 0:
            plt.scatter(pnt.pt[0], pnt.pt[1], marker="o", color='green')
        else:
            numOfErrors += 1
            plt.scatter(pnt.pt[0], pnt.pt[1], marker="o", color='black')
    print("Iteration #" + str(i) + "\n" + "Error Points:" + str(numOfErrors))

    # SHOW plot
    plt.plot(np.array([-1, 1]), np.array([-1, 1]))
    plt.show()
    time.sleep(1)
