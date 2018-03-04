
# coding: utf-8

# In[6]:


import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import sys


# In[7]:

def load_data(path):
    
    columns = ["age", "workclass", "fnlwgt", "education", "education-num", 
           "marital-status", "occupation", "relationship", "race", "sex", 
           "capital-gain", "capital-loss", "hours-per-week", "native-country", 
           "income"]
    data = pd.read_csv(path, sep=",", header=None, index_col=False, names=columns, skipinitialspace=True)
    # Drop the rows where all of the elements are '?'
    data.replace({'?': np.nan}).dropna(axis=0, how='any')
    le_workclass = preprocessing.LabelEncoder()
    le_education = preprocessing.LabelEncoder()
    le_marital_status = preprocessing.LabelEncoder()
    le_occupation = preprocessing.LabelEncoder()
    le_relationship = preprocessing.LabelEncoder()
    le_race = preprocessing.LabelEncoder()
    le_sex = preprocessing.LabelEncoder()
    le_native_country = preprocessing.LabelEncoder()
    le_income = preprocessing.LabelEncoder()

    data['workclass'] = le_workclass.fit_transform(data['workclass'])
    data['education'] = le_education.fit_transform(data['education'])
    data['marital-status'] = le_marital_status.fit_transform(data['marital-status'])
    data['occupation'] = le_occupation.fit_transform(data['occupation'])
    data['relationship'] = le_relationship.fit_transform(data['relationship'])
    data['race'] = le_sex.fit_transform(data['race'])
    data['sex'] = le_sex.fit_transform(data['sex'])
    data['native-country'] = le_native_country.fit_transform(data['native-country'])
    data['income'] = le_income.fit_transform(data['income'])
    
    predictors, responses = data.iloc[:,:-1], data.iloc[:,-1]
    scaler = preprocessing.MinMaxScaler()
    predictors = scaler.fit_transform(predictors)
    responses = responses.values
    
    return predictors, responses


# In[8]:

train_file_path, test_file_path = sys.argv[1], sys.argv[2]
x_train, y_train = load_data(train_file_path)
x_test, y_test = load_data(test_file_path)

# In[9]:


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(30,), max_iter=1000)
mlp.fit(x_train,y_train)
predictions_train = mlp.predict(x_train)
predictions_test = mlp.predict(x_test)
accuracy_train = accuracy_score(y_train,predictions_train)
accuracy_test = accuracy_score(y_test,predictions_test)
cnm_train = confusion_matrix(y_train,predictions_train)
cnm_test = confusion_matrix(y_test,predictions_test)

print( "===== Training Result for Neural Network =====")
print( "Accuracy :: %0.2f" % (accuracy_train))
print( "Confusion matrix \n", cnm_train)
print( "===== Testing Result for Neural Network =====")
print( "Accuracy :: %0.2f" % (accuracy_test))
print( "Confusion matrix \n", cnm_test)


# In[13]:


from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=27)
KNN.fit(x_train,y_train)
predictions_train = KNN.predict(x_train)
predictions_test = KNN.predict(x_test)
accuracy_train = accuracy_score(y_train,predictions_train)
accuracy_test = accuracy_score(y_test,predictions_test)
cnm_train = confusion_matrix(y_train,predictions_train)
cnm_test = confusion_matrix(y_test,predictions_test)

print( "===== Training Result for KNN =====")
print( "Accuracy :: %0.2f" % (accuracy_train))
print( "Confusion matrix \n", cnm_train)
print( "===== Testing Result for KNN =====")
print( "Accuracy :: %0.2f " % (accuracy_test))
print( "Confusion matrix \n", cnm_test)

# In[15]:


from sklearn import svm
svc = svm.SVC(kernel='rbf', C=1)
svc.fit(x_train, y_train)
predictions_train = svc.predict(x_train)
predictions_test = svc.predict(x_test)
accuracy_train = accuracy_score(y_train,predictions_train)
accuracy_test = accuracy_score(y_test,predictions_test)
cnm_train = confusion_matrix(y_train,predictions_train)
cnm_test = confusion_matrix(y_test,predictions_test)

print( "===== Training Result for SVM =====")
print( "Accuracy :: %0.2f" % (accuracy_train))
print( "Confusion matrix \n", cnm_train)
print( "===== Testing Result for SVM =====")
print( "Accuracy :: %0.2f" % (accuracy_test))
print( "Confusion matrix \n", cnm_test)
