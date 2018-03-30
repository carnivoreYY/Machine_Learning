
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import mean_squared_error
import sys


# In[2]:
def load_data(path):
    columns = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 
               'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'cnt']

    dataset = pd.read_csv(path, sep=",", header=0, index_col=False, usecols=columns, skipinitialspace=True)

    preditors = dataset.iloc[:,:-1]
    responses = dataset.iloc[:,-1]
    scaler = preprocessing.MinMaxScaler()
    preditors = scaler.fit_transform(preditors)
    responses = responses.values

    return preditors, responses


# In[3]:

train_file_path, test_file_path = sys.argv[1], sys.argv[2]
x_train, y_train = load_data(train_file_path)
x_test, y_test = load_data(test_file_path)
y_train_reshape = y_train.reshape(len(y_train), 1)
dataset = np.concatenate((x_train, y_train_reshape), axis=1)


# In[4]:


from sklearn.model_selection import KFold
numberSplits = 5
kf = KFold(n_splits=numberSplits, shuffle=True)


# In[31]:


from sklearn.neighbors import KNeighborsRegressor
K, minMSE = 0, float("infinity")
for k in range(1,20):
    MSE = 0
    for train_index, test_index in kf.split(dataset):
        KNNRegr = KNeighborsRegressor(n_neighbors=k)
        KNNRegr.fit(dataset[train_index, :-1], dataset[train_index, -1])
        predictions = KNNRegr.predict(dataset[test_index, :-1])
        MSE += mean_squared_error(dataset[test_index, -1], predictions)
    if MSE/5.0 < minMSE:
        minMSE = MSE/5.0
        K = k

KNNRegr = KNeighborsRegressor(n_neighbors=K)
KNNRegr.fit(x_train, y_train)
predictions_train = KNNRegr.predict(x_train)
predictions_test = KNNRegr.predict(x_test)
MSE_train = mean_squared_error(y_train, predictions_train)
MSE_test = mean_squared_error(y_test, predictions_test)
print( "===== Training Result For KNN =====")
print("MSE :: %0.2f" % (MSE_train))
print( "===== Testing Result For KNN =====")
print("MSE :: %0.2f" % (MSE_test))


# In[8]:


from sklearn.neural_network import MLPRegressor
MSE = 0.0
for train_index, test_index in kf.split(dataset):
    MLPregr = MLPRegressor(hidden_layer_sizes=(10,20,10), max_iter=100000)
    MLPregr.fit(dataset[train_index, :-1], dataset[train_index, -1])
    predictions = MLPregr.predict(dataset[test_index, :-1])
    MSE += mean_squared_error(dataset[test_index, -1], predictions)
MSE_train = MSE/5.0
MLPregr = MLPRegressor(hidden_layer_sizes=(10,30,30,10), max_iter=100000)
MLPregr.fit(x_train, y_train)
predictions_test = MLPregr.predict(x_test)
MSE_test = mean_squared_error(y_test, predictions_test)
print( "===== Training Result For Neural Network =====")
print("MSE :: %0.2f" % (MSE_train))
print( "===== Testing Result For Neural Network =====")
print("MSE :: %0.2f" % (MSE_test))


# In[33]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
MSE_Lasso,MSE_Ridge = 0,0
for train_index, test_index in kf.split(dataset):
    poly = PolynomialFeatures(degree=3)
    poly_x_train = poly.fit_transform(dataset[train_index, :-1])
    poly_x_test = poly.fit_transform(dataset[test_index, :-1])
    regrLasso = linear_model.Lasso(alpha=1.0)
    regrRidge = linear_model.Ridge(alpha=1.0)
    regrLasso.fit(poly_x_train, dataset[train_index, -1])
    regrRidge.fit(poly_x_train, dataset[train_index, -1])
    predictions_Lasso = regrLasso.predict(poly_x_test)
    predictions_Ridge = regrRidge.predict(poly_x_test)
    MSE_Lasso += mean_squared_error(dataset[test_index, -1], predictions_Lasso)
    MSE_Ridge += mean_squared_error(dataset[test_index, -1], predictions_Ridge)

print( "===== Training Result For Linear Regression =====")
print("MSE for Lasso :: %0.2f" % (MSE_Lasso/5.0))
print("MSE for Ridge :: %0.2f" % (MSE_Ridge/5.0))
poly = PolynomialFeatures(degree=3)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.fit_transform(x_test)
regrRidge = linear_model.Ridge(alpha=1.0)
regrLasso = linear_model.Lasso(alpha=1.0)
regrRidge.fit(x_train_poly, y_train)
regrLasso.fit(x_train_poly, y_train)
predictions_Ridge = regrRidge.predict(x_test_poly)
predictions_Lasso = regrLasso.predict(x_test_poly)
MSE_Ridge = mean_squared_error(y_test, predictions_Ridge)
MSE_Lasso = mean_squared_error(y_test, predictions_Lasso)
print( "===== Testing Result For Linear Regression =====")
print("MSE for Lasso :: %0.2f" % (MSE_Lasso))
print("MSE for Ridge :: %0.2f" % (MSE_Ridge))

