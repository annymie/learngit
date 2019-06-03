#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 09:02:50 2018

@author: kongxiangyi
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.neural_network import MLPClassifier  # 批量导入要实现的回归算法
from sklearn import tree
from sklearn import neighbors
from sklearn import ensemble
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from time import time

model_svr = svm.SVC(gamma = 0.001)  # 建立支持向量机回归模型对象
model_gbr = GradientBoostingClassifier()  # 建立梯度增强回归模型对象
model_dt = tree.DecisionTreeClassifier() # 决策树
model_knn = neighbors.KNeighborsClassifier() # KNN
model_rf = ensemble.RandomForestClassifier() # 随机森林
model_ab = ensemble.AdaBoostClassifier() #AdaBoost
model_br = ensemble.BaggingClassifier() #Bagging
model_et = tree.ExtraTreeClassifier() #极端随机树


# The digits dataset
digits = datasets.load_digits()


# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(data[:n_samples], digits.target[:n_samples], test_size = 0.2, random_state = 0)

# Create a classifier: a support vector classifier
model_nn = MLPClassifier(solver='adam', hidden_layer_sizes=350, learning_rate_init=0.001)

model_dic = [model_svr, model_gbr, model_dt, model_knn, model_rf, model_ab, 
             model_br, model_et, model_nn]  # 不同回归模型对象的集合
model_names = ['SVC', 'GBC', 'DecisionTree', 'KNN', 'RandomForest', 'AdaBoost', 
               'Bagging', 'ExtraTree','MLP']  # 不同模型的名称列表
print(82 * '_')
print('init\t\ttime\acc\tprecision\trecall\tf1-score')
i = 0 
for classifier in model_dic:
    t0 = time()
    # We learn the digits on the first half of the digits
    classifier.fit(X_train, Y_train)
    expected = Y_test
    predicted = classifier.predict(X_test)
    print('%-9s\t%.2fs\t%.3f\t%.3f\t%.3f\t%.3f'
          % (model_names[i], (time() - t0), 
             metrics.accuracy_score(expected, predicted),
             metrics.precision_score(expected, predicted, average='micro'),
             metrics.recall_score(expected, predicted, average='micro'),
             metrics.f1_score(expected, predicted, average='micro')))
    i+=1
    # Now predict the value of the digit on the second half:
    #expected = digits.target[n_samples // 2:]
    #predicted = classifier.predict(data[n_samples // 2:])
    
    #print("Classification report for classifier %s:\n%s\n"
          #% (classifier, metrics.classification_report(expected, predicted)))
    #print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))