import sys 
import pandas as pd
import numpy as np
import trees
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel 

depth = [3, 5, 7, 9]
accuracy_dt = {d: [] for d in depth}
accuracy_bt = {d: [] for d in depth}
accuracy_rf = {d: [] for d in depth}

def crossValidate(df_train, df_test): 
	for d in depth: 
		for idx in range(10):
			train_accuracy, test_accuracy = trees.decisionTree(df_train[idx], df_test[idx], d, 50) 
			accuracy_dt[d].append(test_accuracy) 
			train_accuracy, test_accuracy = trees.bagging(df_train[idx], df_test[idx], d, 50, 30) 
			accuracy_bt[d].append(test_accuracy) 
			train_accuracy, test_accuracy = trees.randomForests(df_train[idx], df_test[idx], d, 50, 30) 
			accuracy_rf[d].append(test_accuracy) 

def plot(): 
	mean_dt = []
	stderr_dt = []
	mean_bt = []
	stderr_bt = []
	mean_rf = []
	stderr_rf = []
	for d in depth:
		mean_dt.append(np.mean(accuracy_dt[d]))
		stderr_dt.append(np.std(accuracy_dt[d])/np.sqrt(10)) 
		mean_bt.append(np.mean(accuracy_bt[d]))
		stderr_bt.append(np.std(accuracy_bt[d])/np.sqrt(10))
		mean_rf.append(np.mean(accuracy_rf[d]))
		stderr_rf.append(np.std(accuracy_rf[d])/np.sqrt(10))
		
	plt.errorbar(depth, mean_dt, stderr_dt, label="Decision Tree")
	plt.errorbar(depth, mean_bt, stderr_bt, label="Bagging")
	plt.errorbar(depth, mean_rf, stderr_rf, label="Random Forests")
	plt.xlabel("Depth") 
	plt.ylabel("Accuracy") 
	plt.legend() 
	plt.savefig("cv_depth.jpg")

def ttest(): 
	print("BT vs RF") 
	for d in depth: 
		print("Depth " + str(d))
		print(ttest_rel(accuracy_bt[d], accuracy_rf[d]))
	


if __name__ == "__main__": 
	train = pd.read_csv("trainingSet.csv")
	train = train.sample(random_state=18, frac=1)
	train = train.sample(random_state=32, frac=0.5)
	df_test = np.array_split(train, 10) 
	df_train = [] 
	for i in range(10): 
		df_train.append(train.drop(df_test[i].index))
	crossValidate(df_train, df_test)
	plot()
	ttest()

