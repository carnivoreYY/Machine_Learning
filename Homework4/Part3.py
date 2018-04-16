
# coding: utf-8

# In[15]:


import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import mean_squared_error
import sys
from sklearn.model_selection import train_test_split


# In[21]:


def load_data(path):
    xl = pd.ExcelFile(path)
    data = xl.parse("Sheet1")
    predictors, responses = data.iloc[:,:-1], data.iloc[:,-1]
    scaler = preprocessing.MinMaxScaler()
    predictors = scaler.fit_transform(predictors)
    return predictors, responses


# In[22]:

file_path = sys.argv[1]
predictors, responses = load_data(file_path)
x_train, x_test, y_train, y_test = train_test_split(predictors, responses, test_size=0.25)


# In[32]:


from sklearn.neural_network import MLPRegressor
MLPregr = MLPRegressor(hidden_layer_sizes=(10,30,10), max_iter=100000)
MLPregr.fit(x_train, y_train)
predictions_test = MLPregr.predict(x_test)
MSE_test = mean_squared_error(y_test, predictions_test)
print( "===== Testing Result For Neural Network =====")
print("test MSE :: %0.2f" % (MSE_test))

