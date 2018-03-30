
# coding: utf-8

# In[28]:


import pandas as pd
from sklearn import preprocessing
import numpy as np
import sys
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture


# In[29]:


def load_data(path):
    data = pd.read_csv(path, sep=",", header=0, index_col=False, usecols=[*range(1, 22)], skipinitialspace=True)
    return data

def total_Sum_Of_Squares(data):
    centroid = data.mean()
    total = 0.0
    for index, row in data.iterrows():
        for i in range(data.shape[1]):
            total += (row.iloc[i] - centroid.iloc[i])**2
    return total

def within_total_Sum_Of_Squares(data, y_labels, k):
    clusters = {}
    for index, row in data.iterrows():
        if y_labels[index] not in clusters:
            clusters[y_labels[index]] = []
        clusters[y_labels[index]].append(row)
    withinTotal = 0.0
    for i in range(k):
        cluster = np.asarray(clusters[i])
        withinTotal += cluster.var() * len(cluster) 
    return withinTotal


# In[30]:

file_path = sys.argv[1]
data = load_data(file_path)
total = total_Sum_Of_Squares(data)
print("Total sum of squares :: %0.2f" % (total))


# In[35]:


for k in range(1,11):
    print("===== Clustering Result for K equals", k , "=====")
    #K-Means
    kmeans = KMeans(k)
    kmeans.fit(data)
    labels = kmeans.predict(data)
    withinTotal = within_total_Sum_Of_Squares(data, labels, k)
    print("The ratio for K-Means :: %0.5f" % (withinTotal/total))
    #H-Clustering
    model = AgglomerativeClustering(n_clusters=k)
    labels = model.fit_predict(data)
    withinTotal = within_total_Sum_Of_Squares(data, labels, k)
    print("The ratio for H-Clustering :: %0.5f" % (withinTotal/total))
    #Gaussian Mixture Models
    gmm = GaussianMixture(n_components=k).fit(data)
    labels = gmm.predict(data)
    withinTotal = within_total_Sum_Of_Squares(data, labels, k)
    print("The ratio for Gaussian Mixture Models :: %0.5f" % (withinTotal/total))

