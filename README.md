# Assignment2

In this folder there are three seperate files all relating back to the way to manage and handle anomaly detection. 

Python3 is required
Required Libraries: Numpy, SickitLearn, and Matplotlib
Other required files are all present in the repository. 

A2Materials folder contains the testing data as well as assignment parameters used to make this program. 

DBscan.py has the functions relating directly to DBscan itself. It is not executed. 

k_means.py is a nearly self contained file using some functions present from Loading.py to assist in loading the data sets and identifying the metrics. Executing this file will show a graph of hyperparamers as well as the best accuracy hyperparameter. 

Loading.py is a core file, where it had the ability to load data into the programs, as well as handling metrics calculation functions. executing this file runs the dbscan functions, creates a graph to demonstrate the dbscan clustering, then creates four graphs to demonstrate the metrics present. 

To execute: go into the folder, and use a python3 execution method (python3 (filename)) and it will generate the results. 