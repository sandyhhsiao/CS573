import sys 
import pandas as pd
import numpy as np
import trees
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel 

tree = [10, 20, 40, 50]
accuracy_bt = {t: [] for t in tree}
accuracy_rf = {t: [] for t in tree}

def crossValidate(df_train, df_test): 
	for t in tree: 
		print("Tree " + str(t))
		for idx in range(10):
			train_accuracy, test_accuracy = trees.bagging(df_train[idx], df_test[idx], 8, 50, t) 
			accuracy_bt[t].append(test_accuracy) 
			train_accuracy, test_accuracy = trees.randomForests(df_train[idx], df_test[idx], 8, 50, t) 
			accuracy_rf[t].append(test_accuracy) 
	

def plot(): 
	mean_bt = []
	stderr_bt = []
	mean_rf = []
	stderr_rf = []
	for t in tree:
		mean_bt.append(np.mean(accuracy_bt[t]))
		stderr_bt.append(np.std(accuracy_bt[t])/np.sqrt(10))
		mean_rf.append(np.mean(accuracy_rf[t]))
		stderr_rf.append(np.std(accuracy_rf[t])/np.sqrt(10))
		
	plt.errorbar(tree, mean_bt, stderr_bt, label="Bagging")
	plt.errorbar(tree, mean_rf, stderr_rf, label="Random Forests")
	plt.xlabel("Number of Trees") 
	plt.ylabel("Accuracy") 
	plt.legend() 
	plt.savefig("cv_tree.jpg")

def ttest(): 
	print("BT vs RF") 
	for t in tree: 
		print("Tree " + str(t))
		print(ttest_rel(accuracy_bt[t], accuracy_rf[t]))



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

