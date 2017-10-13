import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import Clean_data as cd
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
#import LogisticRegression as lg

X,X_adj,Y = cd.clean_data()
X_train , X_validation , X_test = cd.data_split(X_adj)
Y_train , Y_validation ,Y_test = cd.data_split(Y)

theta = np.fromfile('Logistic_weights1.csv',sep = ',')
theta = theta[:,np.newaxis]

def hypothesis_function(X,theta):
    """
    input : theta and X
    return the sigmoid function
    """
    mu = X.dot(theta)
    return 1.0/(1.0 + np.exp(-1*mu))

Y_hat_test = (hypothesis_function(X_test,theta)).astype(np.int64)
Y_hat_train = (hypothesis_function(X_train,theta)).astype(np.int64)

print('\n')
print('Training accuracy-score = ', accuracy_score(Y_train.ravel(),Y_hat_train.ravel()))
print('Testing accuracy-score = ',accuracy_score(Y_test.ravel(),Y_hat_test.ravel()))
print('\n')
print('Training roc-auc-score = ',roc_auc_score(Y_train.ravel(),Y_hat_train.ravel()))
print('Testing roc-auc-score = ',roc_auc_score(Y_test.ravel(),Y_hat_test.ravel()))
print('\n')
print('Training classification_report','\n',classification_report(Y_train.ravel(),Y_hat_train.ravel()))
print('Testing classification_report','\n',classification_report(Y_test.ravel(),Y_hat_test.ravel()))
