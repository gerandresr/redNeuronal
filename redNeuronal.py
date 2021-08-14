# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 09:26:42 2021

@author: gerar
"""

import numpy as np

def sigmoid(x):
    # activation function
    return 1/(1+np.e**(-x))


class Layer:
    def __init__(self, con, neuron):
        # .rand(rows, cols)
        self.b = np.random.rand(1, neuron) * 2 - 1  # norm [0,1] to [-1, 1]
        self.W = np.random.rand(con, neuron) * 2 - 1
        
        
class NeutralNetwork:
    def __init__(self, top=[], actFunc=sigmoid):
        self.top = top  # topology  [entradas, capa oculta de x layer, outputs]
        self.actFunc = actFunc  # activation function
        self.model = self.defineModel()  # aca guardo la red neuronal
        
    def defineModel(self):
        NeutralNetwork = []  # lista de Layer
        
        for i in range(len(self.top[:-1])):
            NeutralNetwork.append(Layer(self.top[i], self.top[i+1]))

        return NeutralNetwork

    def predict(self, X=[]):
        out = X
        
        for i in range(len(self.model)):
            z = self.actFunc(out @ self.model[i].W + self.model[i].b)
            out = z
        return out



brain = NeutralNetwork(top=[2, 4, 6])  # 
print(brain.predict(X=[0,0]))
print(brain.predict(X=[0,1]))
print(brain.predict(X=[1,0]))
print(brain.predict(X=[1,1]))