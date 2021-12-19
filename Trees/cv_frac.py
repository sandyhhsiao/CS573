import sys 
import pandas as pd
import numpy as np
import trees
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel 

frac = [0.05, 0.075, 0.1, 0.15, 0.2]
accuracy_dt = {f: [] for f in frac}
accuracy_bt = {f: [] for f in frac}
accuracy_rf = {f: [] for f in frac}

def crossValidate(df_train, df_test): 
	for f in frac: 
		print("frac " + str(f))
		for idx in range(10):
			df = df_train[i].sample(random_state=32, frac=f)
			train_accuracy, test_accuracy = trees.decisionTree(df, df_test[idx], 8, 50) 
			accuracy_dt[f].append(test_accuracy) 
			train_accuracy, test_accuracy = trees.bagging(df, df_test[idx], 8, 50, 30) 
			accuracy_bt[f].append(test_accuracy) 
			train_accuracy, test_accuracy = trees.randomForests(df, df_test[idx], 8, 50, 30) 
			accuracy_rf[f].append(test_accuracy) 
	

def plot(): 
	mean_dt = []
	stderr_dt = []
	mean_bt = []
	stderr_bt = []
	mean_rf = []
	stderr_rf = [] 

	for f in frac:
		mean_dt.append(np.mean(accuracy_dt[f]))
		stderr_dt.append(np.std(accuracy_dt[f])/np.sqrt(10)) 
		mean_bt.append(np.mean(accuracy_bt[f]))
		stderr_bt.append(np.std(accuracy_bt[f])/np.sqrt(10))
		mean_rf.append(np.mean(accuracy_rf[f]))
		stderr_rf.append(np.std(accuracy_rf[f])/np.sqrt(10))
		
	plt.errorbar(frac, mean_dt, stderr_dt, label="Decision Tree")
	plt.errorbar(frac, mean_bt, stderr_bt, label="Bagging")
	plt.errorbar(frac, mean_rf, stderr_rf, label="Random Forests")
	plt.xlabel("t_frac") 
	plt.ylabel("Accuracy") 
	plt.legend() 
	plt.savefig("cv_frac.jpg")
	
def ttest(): 
	print("BT vs RF") 
	for f in frac: 
		print("t_frac " + str(f))
		print(ttest_rel(accuracy_bt[f], accuracy_rf[f]))


if __name__ == "__main__": 
	train = pd.read_csv("trainingSet.csv")
	train = train.sample(random_state=18, frac=1)
	df_test = np.array_split(train, 10) 
	df_train = [] 
	for i in range(10): 
		df_train.append(train.drop(df_test[i].index))

	crossValidate(df_train, df_test)
	plot() 
	ttest()

