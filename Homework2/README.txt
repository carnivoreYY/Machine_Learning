-----------------------------------------------------
===   Homework 2 CS 6375.501: Machine Learning    ===
-----------------------------------------------------
Name: Ying Yi
Email: yxy160830@utdallas.edu
NetID: yxy160830

Name: Zeqing Li
Email: zxl165030@utdallas.edu
NetID: yxy160830

=== RUNTIME AND DEPENDENCIES ===
Requires Anaconda 5.0.1 or above with Python 3.6
Requires pandas
Requires sklearn
Requires numpy


There are 5 text files available:

1. README.txt: This file contains instructions on how to run the program and where to place your test data.
2. Homework2/adultTrain.data contains the training data for census income prediction
3. Homework2/adultTest.data contains the testing data for census income prediction
4. Homework2/bikeRentalHourlyTrain.data contain the training data for bike rental prediction 
5. Homework2/bikeRentalHourlyTest.data contain the testing data for bike rental prediction 

-----------------------------------------------------
== USAGE: HOW TO RUN THE PROGRAM / INSTALLATION ==
-----------------------------------------------------
(NOTE: PLEASE ENSURE THAT THE LOCATION OF THE DATASET IS ACCESSIBLE TO THE PROGRAM.)

To run the program the user must navigate to "Homework2" folder where all the python files are available.

please place test data under 'Homework2' directory, otherwise the program won't work without testing data.

  1. Required test data:
    a) adultTest.data
    b) bikeRentalHourlyTest.csv

----------------------
===== Census Income Prediction - Neural Network, SVM, KNN =====
----------------------
(NOTE THE MANUAL COMPILATIONS AND CODE EXECUTION MUST FOLLOW THE BELOW STEPS IN ORDER)
    1. Please navigate to the 'Homework2' folder and execute the commands below

    2.
      # Census Income Prediction
      python HW2CensusIncome.py adultTrain.data adultTest.data


----------------------
===== Bike Rental Prediction - Neural Networks, Linear Regression (lasso and ridge), KNN =====
----------------------
(NOTE THE MANUAL COMPILATIONS AND CODE EXECUTION MUST FOLLOW THE BELOW STEPS IN ORDER)
    1. Please navigate to the 'Homework2' folder and execute the commands below

    2.
      # Bike Rental Prediction
      python HW2BikeRental.py bikeRentalHourlyTrain.csv bikeRentalHourlyTest.csv


----------------------
===== SAMPLE PROGRAM OUTPUT =====
----------------------

---------------------------------- Census Income Prediction -----------------------------
===== Training Result for Neural Network =====
Accuracy :: 0.85
Confusion matrix
 [[23120  1600]
 [ 3325  4516]]
===== Testing Result for Neural Network =====
Accuracy :: 0.85
Confusion matrix
 [[11628   807]
 [ 1647  2199]]
===== Training Result for KNN =====
Accuracy :: 0.85
Confusion matrix
 [[22886  1834]
 [ 3169  4672]]
===== Testing Result for KNN =====
Accuracy :: 0.84
Confusion matrix
 [[11434  1001]
 [ 1683  2163]]
===== Training Result for SVM =====
Accuracy :: 0.83
Confusion matrix
 [[23569  1151]
 [ 4292  3549]]
===== Testing Result for SVM =====
Accuracy :: 0.83
Confusion matrix
 [[11827   608]
 [ 2116  1730]]

---------------------------------- Bike Rental Prediction -----------------------------
===== Training Result For KNN =====
MSE :: 7278.33
===== Testing Result For KNN =====
MSE :: 9020.33
===== Training Result For Neural Network =====
MSE :: 5902.30
===== Testing Result For Neural Network =====
MSE :: 3155.57
===== Training Result For Linear Regression =====
MSE for Lasso :: 15514.78
MSE for Rige :: 12750.91
===== Testing Result For Linear Regression =====
MSE for Lasso :: 15547.92
MSE for Rige :: 12278.03
