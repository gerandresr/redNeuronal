# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 09:26:42 2021

@author: gerar
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles


def sigmoid(x, deriv=False):
    if deriv:
        return x * (1-x)  # para back propagation
    return 1/(1+np.e**(-x))

def tanh(x, deriv=False):
    if deriv:
        return (1 - (np.tanh(x))**2)
    return np.tanh(x)

def mse(Yp, Yr, deriv=False):
    # error cuadratico medio
    '''
    Yp = valor calculado
    Yr = valor real
    '''
    if deriv:
        return (Yp - Yr)
    return np.mean((Yp - Yr)**2)


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

    def fit(self, X=[], Y=[], epochs=100, learnRate=0.5):

        for k in range(epochs):
            out = [(None, X)]
            
            for i in range(len(self.model)):
                z = out[-1][1] @ self.model[i].W + self.model[i].b
                a = sigmoid(z, deriv=False)
                out.append((z, a))  # append salida sin activar y activacion
            
            deltas = []  # guardamos el error en la funcion de coste
            
            for i in reversed(range(len(self.model))):
                z = out[i+1][0]
                a = out[i+1][1]
                
                if i == len(self.model)-1:
                    deltas.insert(0, mse(a, Y, deriv=True)*sigmoid(a, deriv=True))  #que tanto se equivoco
                else:
                    # error de cada capa... Back Propagation
                    deltas.insert(0, deltas[0] @ _W.T * sigmoid(a, deriv=True))  # que tanto varia la salida con respecto al error

                _W = self.model[i].W
            
                # descenso de gradiente
                self.model[i].b = self.model[i].b - np.mean(deltas[0], axis=0, keepdims=True) * learnRate
                self.model[i].W = self.model[i].W - out[i][1].T @ deltas[0] * learnRate

        print('aprendio!!')


def randomPoints(n=100):
    x = np.random.uniform(0.0, 1.0, n)
    y = np.random.uniform(0.0, 1.0, n)
    
    return np.array([x, y]).T


def main():
    brainXor = NeutralNetwork(top=[2, 4, 1], actFunc=sigmoid)

    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        ])
    Y = np.array([
        [0],
        [1],
        [1],
        [0],
        ])

    brainXor.fit(X=X, Y=Y, epochs=10000, learnRate=0.08)
    
    xTest = randomPoints(n=3000)
    yTest = brainXor.predict(xTest)
    
    plt.scatter(xTest[:,0], xTest[:,1], c=yTest, s=25, cmap='GnBu')
    plt.show()

    # circles
    brainCirc = NeutralNetwork(top=[2, 4, 8, 1], actFunc=sigmoid)
    X, Y = make_circles(n_samples=500, noise=0.05, factor=0.5)
    Y = Y.reshape(len(X), 1)
    
    brainCirc.fit(X=X, Y=Y, epochs=10000, learnRate=0.05)
    yTest = brainCirc.predict(X)

    plt.scatter(X[:,0], X[:,1], c=yTest, cmap='winter', s=25)
    plt.show()

if __name__ == '__main__':
    main()
