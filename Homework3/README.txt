-----------------------------------------------------
===   Homework 3 CS 6375.501: Machine Learning    ===
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

1. README.txt: This file contains instructions on how to run the program and where to place your train/cluster data.
2. Homework3/dataset_Facebook.csv contains the training/clustering data for Facebook Post metrics clustering
3. Homework3/Frogs_MFCCs.csv contain the training/clustering data Anuran Calls clustering 

-----------------------------------------------------
== USAGE: HOW TO RUN THE PROGRAM / INSTALLATION ==
-----------------------------------------------------
(NOTE: PLEASE ENSURE THAT THE LOCATION OF THE DATASET IS ACCESSIBLE TO THE PROGRAM.)

To run the program the user must navigate to "Homework3" folder where all the python files are available.

please place test data under 'Homework3' directory, otherwise the program won't work without testing data.

  1. Required train/cluster data:
    a) dataset_Facebook.csv
    b) Frogs_MFCCs.csv

----------------------
===== Facebook Post Metrics Clustering - K-Means, H-Clustering, Gaussian Mixture Models =====
----------------------
(NOTE THE MANUAL COMPILATIONS AND CODE EXECUTION MUST FOLLOW THE BELOW STEPS IN ORDER)
    1. Please navigate to the 'Homework3' folder and execute the commands

    2. $ cd Homework3/
       $ python FBPostMetrics.py dataset_Facebook.csv


----------------------
===== Anuran Calls Clustering - K-Means, H-Clustering, Gaussian Mixture Models
----------------------
(NOTE THE MANUAL COMPILATIONS AND CODE EXECUTION MUST FOLLOW THE BELOW STEPS IN ORDER)
    1. Please navigate to the 'Homework3' folder and execute the commands

    2. $ cd Homework3/
       $ python AnuranCalls.py Frogs_MFCCs.csv


----------------------
===== SAMPLE PROGRAM OUTPUT =====
----------------------

---------------------------------- Facebook Post Metrics Clustering -----------------------------

Total sum of squares :: 410.68
===== Clustering Result for K equals 1 =====
The ratio for K-Means :: 0.10650
The ratio for H-Clustering :: 0.10650
The ratio for Gaussian Mixture Models :: 0.10650
===== Clustering Result for K equals 2 =====
The ratio for K-Means :: 0.10560
The ratio for H-Clustering :: 0.10572
The ratio for Gaussian Mixture Models :: 0.10614
===== Clustering Result for K equals 3 =====
The ratio for K-Means :: 0.10554
The ratio for H-Clustering :: 0.10495
The ratio for Gaussian Mixture Models :: 0.10544
===== Clustering Result for K equals 4 =====
The ratio for K-Means :: 0.10509
The ratio for H-Clustering :: 0.10479
The ratio for Gaussian Mixture Models :: 0.10555
===== Clustering Result for K equals 5 =====
The ratio for K-Means :: 0.10507
The ratio for H-Clustering :: 0.10478
The ratio for Gaussian Mixture Models :: 0.10498
===== Clustering Result for K equals 6 =====
The ratio for K-Means :: 0.10506
The ratio for H-Clustering :: 0.10441
The ratio for Gaussian Mixture Models :: 0.10500
===== Clustering Result for K equals 7 =====
The ratio for K-Means :: 0.10477
The ratio for H-Clustering :: 0.10440
The ratio for Gaussian Mixture Models :: 0.10459
===== Clustering Result for K equals 8 =====
The ratio for K-Means :: 0.10473
The ratio for H-Clustering :: 0.10398
The ratio for Gaussian Mixture Models :: 0.10528
===== Clustering Result for K equals 9 =====
The ratio for K-Means :: 0.10400
The ratio for H-Clustering :: 0.10384
The ratio for Gaussian Mixture Models :: 0.10486
===== Clustering Result for K equals 10 =====
The ratio for K-Means :: 0.10451
The ratio for H-Clustering :: 0.10380
The ratio for Gaussian Mixture Models :: 0.10449

---------------------------------- Anuran Calls Clustering -----------------------------

Total sum of squares :: 3693.26
===== Clustering Result for K equals 1 =====
The ratio for K-Means :: 0.08472
The ratio for H-Clustering :: 0.08472
The ratio for Gaussian Mixture Models :: 0.08472
===== Clustering Result for K equals 2 =====
The ratio for K-Means :: 0.08472
The ratio for H-Clustering :: 0.08472
The ratio for Gaussian Mixture Models :: 0.08472
===== Clustering Result for K equals 3 =====
The ratio for K-Means :: 0.08430
The ratio for H-Clustering :: 0.08456
The ratio for Gaussian Mixture Models :: 0.08443
===== Clustering Result for K equals 4 =====
The ratio for K-Means :: 0.08430
The ratio for H-Clustering :: 0.08446
The ratio for Gaussian Mixture Models :: 0.08447
===== Clustering Result for K equals 5 =====
The ratio for K-Means :: 0.08424
The ratio for H-Clustering :: 0.08441
The ratio for Gaussian Mixture Models :: 0.08428
===== Clustering Result for K equals 6 =====
The ratio for K-Means :: 0.08424
The ratio for H-Clustering :: 0.08431
The ratio for Gaussian Mixture Models :: 0.08427
===== Clustering Result for K equals 7 =====
The ratio for K-Means :: 0.08425
The ratio for H-Clustering :: 0.08430
The ratio for Gaussian Mixture Models :: 0.08422
===== Clustering Result for K equals 8 =====
The ratio for K-Means :: 0.08424
The ratio for H-Clustering :: 0.08420
The ratio for Gaussian Mixture Models :: 0.08427
===== Clustering Result for K equals 9 =====
The ratio for K-Means :: 0.08412
The ratio for H-Clustering :: 0.08415
The ratio for Gaussian Mixture Models :: 0.08419
===== Clustering Result for K equals 10 =====
The ratio for K-Means :: 0.08412
The ratio for H-Clustering :: 0.08411
The ratio for Gaussian Mixture Models :: 0.08413
