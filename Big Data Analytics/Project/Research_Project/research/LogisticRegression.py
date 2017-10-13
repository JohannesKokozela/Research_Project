#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 20:17:05 2017

@author: johannestebalokokozela
"""

import numpy as np
#import pandas as pd
import Clean_data as cd
#import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, classification_report ,accuracy_score


np.random.seed(1000)


def hypothesis_function(X,theta):
    """
    input : theta and X
    return the sigmoid function
    """
    mu = X.dot(theta)
    return 1.0/(1.0 + np.exp(-1*mu))


def convergence(theta, theta_p, epsilon=1e-8):
    """
    checks convergence
    theta: theta at iteration t
    thetap: theta at iteration t-1
    epsilon: a small number used for thresholdin
    """
    L = theta.shape[0]
    d = np.max(np.abs(theta[0]-theta_p[0]))
    for l in np.arange(L):
      d_temp = np.max(np.abs(theta[l]-theta_p[l]))
      if d_temp > d:
        d = d_temp
        if  d <= epsilon:
          return True
        else:
          return False

def logistic_regression_train(theta,X,Y):
        """
        return new theta weights
        """
        boolValue = False
        alpha = 0.1
        theta_temp = np.zeros((1,X.shape[1]))
        n = 0
        while n<=10000:
            Y_hat =  hypothesis_function(X,theta)
            theta_temp = theta_temp - alpha*(1.0/X.shape[0])* np.dot((Y_hat-Y).T,X)
            cost = 1.0/float(2*Y.shape[0])*(Y_hat-Y).T.dot((Y_hat-Y))
            #boolValue = convergence(theta_temp.T,theta)
            theta = theta_temp.T
            if n%100==0:
              print('Cost Function Value  = %2.5f \t interation = %2.1f' %(cost,n))
            #alpha -=0.0001
            n+=1
        return theta

#Start Working here

X,X_adj,Y = cd.clean_data()
X_train , X_validation , X_test = cd.data_split(X_adj)
Y_train , Y_validation ,Y_test = cd.data_split(Y)

theta = np.random.rand(X_train.shape[1],1)  #initializing theata
theta_update= logistic_regression_train(theta,X_train,Y_train)


Y_hat = hypothesis_function(X_train,theta_update)
Y_array_train =  Y_train.astype(np.int64).ravel()
Y_hat_array = Y_hat.astype(np.int64).ravel()


print('\n')
print('accuracy : %2.2f' %(accuracy_score(Y_array_train,Y_hat_array)))
print('auc score : %2.2f' %roc_auc_score(Y_array_train,Y_hat_array))

print(classification_report(Y_array_train,Y_hat_array))
theta_update.tofile('Logistic_weights5.csv',sep = ',')



#print(Y_hat)
#print(Y_train)
