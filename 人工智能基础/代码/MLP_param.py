#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 10:16:39 2018

@author: kongxiangyi
"""

from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pickle

if __name__ == '__main__':
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    labels = digits.target

    clf = MLPClassifier(learning_rate_init=0.001)

    hidden_layer_sizes = list(range(100, 601, 20))

    learning_rate_init = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005,
                          0.0006, 0.0007, 0.0008, 0.0009,
                          0.001, 0.002, 0.003, 0.004, 0.005,
                          0.006, 0.007, 0.008, 0.009,
                          0.01, 0.02, 0.03, 0.04, 0.05,
                          0.06, 0.07, 0.08, 0.09, 0.1]
    
    param_grid = [{
        #'solver': ['sgd', 'adam'],
        'hidden_layer_sizes': hidden_layer_sizes,
        'activation': ['identity', 'tanh', 'relu']
    }]
    grid = GridSearchCV(clf, param_grid, scoring='f1_micro', cv=5, n_jobs=4,
                        pre_dispatch='2*n_jobs', verbose=3, refit=False)
    grid.fit(data, digits.target)

    with open('hidden_layer.pkl', 'wb') as f:
        pickle.dump(grid.cv_results_, f)
        
with open('hidden_layer.pkl', 'rb') as f:
    result = pickle.load(f)

params = result['params']
mean_test_score = result['mean_test_score']

hidden_layer_sizes = []
identity_score = []
relu_score = []
tanh_score = []
for param, score in zip(params, mean_test_score):
    if param['activation'] == 'identity':
        hidden_layer_sizes.append(param['hidden_layer_sizes'])
        identity_score.append(score)
    elif param['activation'] == 'tanh':
        tanh_score.append(score)
    elif param['activation'] == 'relu':
        relu_score.append(score)

plt.subplots(figsize=(10, 5))
plt.plot(hidden_layer_sizes, identity_score, label='identity', color='b')
plt.plot(hidden_layer_sizes, tanh_score, label='tanh', color='r')
plt.plot(hidden_layer_sizes, relu_score, label='relu', color='g')

plt.xlabel('Hidden layer sizes')
plt.ylabel('Micro-f1 score')
plt.title('The micro-f1 score of different hidden layer sizes')
plt.legend()
plt.savefig('hidden_layer')
plt.show()