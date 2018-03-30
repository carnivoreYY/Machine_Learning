-----------------------------------------------------
===   Homework 1 CS 6375.501: Machine Learning    ===
-----------------------------------------------------
Name: Ying Yi
Email: yxy160830@utdallas.edu
NetID: yxy160830

Name: Zeqing Li
Email: zxl165030@utdallas.edu
NetID: yxy160830

=== RUNTIME AND DEPENDENCIES ===
Requires Anaconda 5.0.1 or above with Python 3.6
Requires nltk
Requires nltk stopwords
Requires nltk wordnet


There are 3 text files available:

1. README.txt: This file contains instructions on how to run the program and where to place your test data.
2. Homework1/textTrainData.txt contains the training data for sentiment prediction
3. Homework1/carTrainData.csv contain the training data for car evaluation

-----------------------------------------------------
== USAGE: HOW TO RUN THE PROGRAM / INSTALLATION ==
-----------------------------------------------------
(NOTE: PLEASE ENSURE THAT THE LOCATION OF THE DATASET IS ACCESSIBLE TO THE PROGRAM.)

To run the program the user must navigate to "Homework1" folder where all the python files are available.

please place test data under 'Homework1' directory, otherwise the program won't work without testing data.

  1. Required test data:
    a) textTestData.txt 
    b) carTestData.csv

----------------------
===== SENTIMENT PREDICTION =====
----------------------
(NOTE THE MANUAL COMPILATIONS AND CODE EXECUTION MUST FOLLOW THE BELOW STEPS IN ORDER)
    1. Please navigate to the 'Homework1' folder and execute the commands below

    2.
      # Sentiment Prediction
      python TextClassifier.py <train-data-file> <test-data-file>
      For example, $ python TextClassifier.py textTrainData.txt textTestData.txt


----------------------
===== CAR EVALUATION =====
----------------------
(NOTE THE MANUAL COMPILATIONS AND CODE EXECUTION MUST FOLLOW THE BELOW STEPS IN ORDER)
    1. Please navigate to the 'Homework1' folder and execute the commands below

    2.
      # Car Evaluation
      python CarClassifier.py <train-data-file> <test-data-file>
      For example, $ python CarClassifier.py carTrainData.csv carTestData.csv


----------------------
===== SAMPLE PROGRAM OUTPUT =====
----------------------
$ python TextClassifier.py textTrainData.txt textTrainData.txt
Negative prediction: 2.6809960173611194e-15
Positive prediction: 8.647440316728755e-15
===== Training Result =====
Train Accuracy  :: 0.83
Confusion matrix
 [[1141  127]
 [ 293  974]]
===== Test Result =====
Test Accuracy  :: 0.83
Confusion matrix
 [[1141  127]
 [ 293  974]]

$ python CarClassifier.py carTrainData.csv carTrainData.csv
===== Training Result =====
Train Accuracy :  1.0
Confusion matrix
 [[ 307    0    0    0]
 [   0   57    0    0]
 [   0    0 1013    0]
 [   0    0    0   51]]
===== Test Result =====
Test Accuracy :  1.0
Confusion matrix
 [[ 307    0    0    0]
 [   0   57    0    0]
 [   0    0 1013    0]
 [   0    0    0   51]]

===== NLTK CORPUS INSTALLATION =====
# Install nltk via anaconda
  conda install -c anaconda nltk
# Launch Python Interactive Shell
  python
>>> import nltk
>>> nltk.download("stopwords")
>>> nltk.download("wordnet")