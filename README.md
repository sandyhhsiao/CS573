# Purde CS573

This repo consists of the classifiers I implemented from scratch using Python with mainly `pandas` and `numpy` libraries for this course. Assignment 1 is a recap on statistic and linear algebra. Thus, it is not included here. 
<hr> 

## Assignment 2 - Naive Bayes Classifier 

Implemented a Naive Bayes classifier (NBC) and tested it with a speed dating dataset `dating-full.csv`. The dataset was splitted into training (80% of the dataset) and testing set (20% of the dataset), and the model reached 77% and 76% of accuracy for training set and testing set respectively. In addition to implementing the classifier, there were several analysis regarding the dataset, comparisons about the performances when the size of dataset varies and when the bin size (for discretization) changes. These can be found in `Naive_Bayes_Classifier/report.pdf`.  

## Assignment 3 - Support Vector Machine & Logistic Regression 

Implemented linear Support Vector Machine (SVM) and Logistic Regression (LR) and tested them with the same dataset as Naive Bayes Classifier `dating-full.csv`. The dataset was also splitted into training set (80%) and testing set (20%). The accuracy for training and testing data with SVM were 56% and 58% respectively. The accuracy for training and testing data with LR were 66% and 67% respectively. There was a comparison among NBC, SVM, and LR. The result can be found in `SVM_LR/report.pdf`.  

## Assignment 4 - Trees (Decision Trees, Bagging, Random Forests) 
Implemented Decision Trees, Bagging, and Random Forests and tested them with `dating-full.csv`. The dataset was splitted into training (80%) and testing (20%) as before. The trainging and testing accuracy for Decision Trees were 77% and 74% respectively. The training and testing accuracy for Bagging were 79% and 75% respectively. The training and testing accuracy for Random Forests were 77% and 73% respectively. There was a in-depth analysis on the influence of tree depth for all three models and an analysis on the influence of the number of trees used in Bagging and Random Forests. These can be found in `Trees/report.pdf`.  

## Assignment 5 - Clustering (K-Means)
Implemented K-Means clustering and tested it with `digits-embegging.csv`. The evaluation was done using WC_SSD (within-cluster sum of squared distances) and SC (silhouette coefficient) to choose the proper number of cluster and see if it was close to the actual number of the classes. This assignment also imported hierarchical clustering from `scipy` and evaluated the dataset with different linkage methods of hierarchical clustering. The evaluation can be found in `Clustering/report.pdf`. 