-----------------------------------------------------
===   Homework 4 CS 6375.501: Machine Learning    ===
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
Requires sys
Requires matplotlib
Requires itertools


There are 4 files available:

1. README.txt: This file contains instructions on how to run the program and where to place your test data.
2. Homework4/ThoraricSurgery.arff contains the data set Thoracic Surgery Data
3. Homework4/Sales_Transactions_Dataset_Weekly.csv contains the Sales_Transactions_Dataset_Weekly Data Set
4. Homework4/Concrete_Data.xls contain Concrete Compressive Strength Data Set

-----------------------------------------------------
== USAGE: HOW TO RUN THE PROGRAM / INSTALLATION ==
-----------------------------------------------------
(NOTE: PLEASE ENSURE THAT THE LOCATION OF THE DATASET IS ACCESSIBLE TO THE PROGRAM.)

To run the program the user must navigate to "Homework4" folder where all the python files are available.

----------------------
===== Part 1 : Thoracic Surgery Data Classification =====
----------------------
(NOTE THE MANUAL COMPILATIONS AND CODE EXECUTION MUST FOLLOW THE BELOW STEPS IN ORDER)
    1. Please navigate to the 'Homework4' folder and execute the commands below

    2. python Part1.py ThoraricSurgery.arff


----------------------
===== Part 2 : Sales Transactions Dataset Weekly Data =====
----------------------
(NOTE THE MANUAL COMPILATIONS AND CODE EXECUTION MUST FOLLOW THE BELOW STEPS IN ORDER)
    1. Please navigate to the 'Homework4' folder and execute the commands below

    2. python Part2.py Sales_Transactions_Dataset_Weekly.csv

----------------------
===== Part 3 : Concrete Compressive Strength Data =====
----------------------
(NOTE THE MANUAL COMPILATIONS AND CODE EXECUTION MUST FOLLOW THE BELOW STEPS IN ORDER)
    1. Please navigate to the 'Homework4' folder and execute the commands below

    2. python Part3.py Concrete_Data.xls

----------------------
===== SAMPLE PROGRAM OUTPUT =====
----------------------

---------------------------------- Sales Transactions Dataset Weekly Data -----------------------------
===== Clustering Result for K equals 2 =====
The TWSS/TSS ratio :: 0.01514
===== Clustering Result for K equals 10 =====
The TWSS/TSS ratio :: 0.01429

---------------------------------- Concrete Compressive Strength Data -----------------------------
===== Testing Result For Neural Network =====
test MSE :: 48.47
