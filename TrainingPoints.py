# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 18:47:14 2021

@author: asafd
"""
import numpy as np;
class Point:
    
    def __init__(self):
       self.pt = np.random.rand(2,1) *2 -1;
       
       if(self.pt[0]>=self.pt[1]):
           self.label = 1;
       else:
           self.label = -1;
           
