#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 16:15:24 2017

@author: johannestebalokokozela
"""

import numpy as np
import pandas as pd
import Clean_data as cd
from scipy.linalg import blas
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import roc_auc_score, accuracy_score ,classification_report

#import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

X,X_adj,Y = cd.clean_data()

forty = np.int64(X.shape[0]*0.2)

X_train = X_adj[:forty,:]
Y_train = Y[0:forty,:]



def A(X,Y):
    model = LogisticRegression()
    model.fit(X_train,Y_train.ravel())
    w =  model.predict(X_train)
    w =  w[:,np.newaxis]
    w = np.ascontiguousarray(w)
    wt = np.ascontiguousarray(w.T)
    wwt = w.dot(wt)
    I = np.identity(wwt.shape[0])
    A = I+wwt
    return A


def knn(X,new_x,Y,A):
    P = np.ascontiguousarray(X-new_x)
    AP = A.dot(P)
    PAP = np.sqrt(P.T.dot(AP))
    i = np.argmin(PAP)
    return Y[i]


def predict(X,new_x,Y,A):
  n = 100
  for i in np.arange(n):
      if i%10==0:
        print(i)
      Y_hat = np.zeros(n)
      Y_hat[i] = knn(X,new_x[i,:][np.newaxis,:],Y,A)
  return Y_hat

A = A(X_train,Y_train[:50])
Y_hat = predict(X_train,X_train,Y_train,A)


print(accuracy_score(Y_train[:100].ravel(),Y_hat.ravel()))
print(roc_auc_score(Y_train[:100].ravel(),Y_hat.ravel()))
print(classification_report(Y_train[:100].ravel(),Y_hat.ravel()))



