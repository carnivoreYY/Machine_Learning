

from sklearn.preprocessing import LabelEncoder  
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
import sys

train_file_path = sys.argv[1]
test_file_path = sys.argv[2]

train_data = pd.read_csv(train_file_path, sep='\t', encoding='latin1')
test_data = pd.read_csv(test_file_path, sep='\t', encoding='latin1')
stop_words = set(stopwords.words('english'))


def get_text(reviews, score):
  # Join together the text in the reviews for a particular sentiment.
  # We lowercase to avoid "Not" and "not" being seen as different words, for example.

    s = ""
    for index,row in reviews.iterrows():
        if row['Sentiment'] == score:
            s = s + row['Sentence'].lower()
    
    return s

def count_text(text):
    lemma = nltk.wordnet.WordNetLemmatizer()
    words = re.compile('\w+').findall(text)
    s = []
    for word in words:
        if word not in stop_words:
            s.append(lemma.lemmatize(word))
    result = Counter(s)
    return result


negative_train_text = get_text(train_data, 0)
positive_train_text = get_text(train_data, 1)
# Here we generate the word counts for each sentiment
negative_counts = count_text(negative_train_text)
# Generate word counts for positive tone.
positive_counts = count_text(positive_train_text)


# In[ ]:


def get_y_count(train_data, score):
  # Compute the count of each classification occuring in the data.
  # return len([r for r in reviews if r[1] == str(score)])
    c = 0
    for index,row in train_data.iterrows():
        if row['Sentiment'] == score:
            c = c + 1
    
    return c


# In[ ]:


positive_review_count = get_y_count(train_data, 1)
negative_review_count = get_y_count(train_data, 0)


# In[ ]:


prob_positive = positive_review_count / len(train_data)
prob_negative = negative_review_count / len(train_data)


# In[ ]:


def make_class_prediction(text, counts, class_prob, class_count):
  prediction = 1
  text_counts = Counter(re.split("\s+", text))
  for word in text_counts:
      # For every word in the text, we get the number of times that word occured in the reviews for a given class, add 1 to smooth the value, and divide by the total number of words in the class (plus the class_count to also smooth the denominator).
      # Smoothing ensures that we don't multiply the prediction by 0 if the word didn't exist in the training data.
      # We also smooth the denominator counts to keep things even.
      prediction *=  text_counts.get(word) * ((counts.get(word, 0) + 1) / (sum(counts.values()) + class_count))
  # Now we multiply by the probability of the class existing in the documents.
  return prediction * class_prob

# In[ ]:


def make_decision(text, make_class_prediction):
    # Compute the negative and positive probabilities.
    negative_prediction = make_class_prediction(text, negative_counts, prob_negative, negative_review_count)
    positive_prediction = make_class_prediction(text, positive_counts, prob_positive, positive_review_count)

    # We assign a classification based on which probability is greater.
    if negative_prediction > positive_prediction:
      return 0
    return 1


# Now we make predictions on the test data. Since it is a large set, we will simply select 200 movies.
train_predictions = [make_decision(row['Sentence'], make_class_prediction) for index,row in train_data.iterrows()]
train_actual = train_data['Sentiment'].tolist()
train_accuracy = sum(1 for i in range(len(train_predictions)) if train_predictions[i] == train_actual[i]) / float(len(train_predictions))

print( "===== Training Result =====")
print( "Train Accuracy  :: %0.2f" % (train_accuracy))
print( "Confusion matrix \n", confusion_matrix(train_actual, train_predictions))

# Test results

# Now we make predictions on the test data. Since it is a large set, we will simply select 200 movies.
test_predictions = [make_decision(row['Sentence'], make_class_prediction) for index,row in test_data.iterrows()]
test_actual = test_data['Sentiment'].tolist()
test_accuracy = sum(1 for i in range(len(test_predictions)) if test_predictions[i] == test_actual[i]) / float(len(test_predictions))
print( "===== Test Result =====")
print( "Test Accuracy  :: %0.2f" % (test_accuracy))
print( "Confusion matrix \n", confusion_matrix(test_actual, test_predictions))