
# coding: utf-8

# In[1]:


from sklearn.preprocessing import LabelEncoder  
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import tree
import sys


def load_data(path):
    """
    Load and process dataset
    """
    dataset = pd.read_csv(path)
    le = LabelEncoder()
    dataset = dataset.apply(le.fit_transform)
    
    # Fetching predict variables at columns 'V1' 'V2' 'V3' 'V4' 'V5' 'V6'
    preditors = dataset.iloc[:,:-1]
    
    # Fetching response variable at column 'V7'
    responses = dataset.iloc[:,-1]
    return preditors, responses
    
train_file_path = sys.argv[1]
test_file_path = sys.argv[2]

# Load train dataset
train_x, train_y = load_data(train_file_path)

# Load test dataset
test_x, test_y = load_data(test_file_path)

# Train decition tree model
clf = tree.DecisionTreeClassifier()
trained_model = clf.fit(train_x, train_y)


# Demostrate traning results
predictions = trained_model.predict(train_x)
actual = train_y
print( "===== Training Result =====")
print( "Train Accuracy : ", accuracy_score(actual, predictions))
print( "Confusion matrix \n", confusion_matrix(actual, predictions))

# Demostrate test results
predictions = trained_model.predict(test_x)
actual = test_y
print( "===== Test Result =====")
print( "Test Accuracy : ", accuracy_score(actual, predictions))
print( "Confusion matrix \n", confusion_matrix(actual, predictions))





