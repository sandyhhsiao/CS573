import sys 
import pandas as pd
import numpy as np
import model 
import lr_svm
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel 


t_frac = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2] 
accuracy_nbc = {frac: [] for frac in t_frac}
accuracy_lr = {frac: [] for frac in t_frac}
accuracy_svm = {frac: [] for frac in t_frac}

def learn_lr_svm(train, trainset): 
	for frac_size in t_frac: 
		for idx in range(10): 
			df_test = trainset[idx].copy(deep=True) 
			df_test_dec = df_test.pop("decision")
			df = train.drop(df_test.index).sample(random_state=32, frac=frac_size) 
			df_dec = df.pop("decision")
			accuracy_lr[frac_size].append(lr_svm.lr(df, df_dec, df_test, df_test_dec))	
			accuracy_svm[frac_size].append(lr_svm.svm(df, df_dec, df_test, df_test_dec))			

def learn_nbc(train_nbc, trainset_nbc): 
	for idx in range(10): 
		test = trainset_nbc[idx] 
		train = train_nbc.drop(test.index)
		for frac_size in t_frac: 
			df = train.sample(random_state=32, frac=frac_size) 	
			nbcObj = model.NBC(df)
			nbcObj.learnProb(5) 
			nbcObj.evaluate(test, "Testing")
			accuracy_nbc[frac_size].append(nbcObj.accuracy["Testing"])			

def plot(): 
	mean_nbc = []
	stderr_nbc = []
	mean_lr = []
	stderr_lr = []
	mean_svm = []
	stderr_svm = []
	for frac in t_frac:
		mean_nbc.append(np.mean(accuracy_nbc[frac]))
		stderr_nbc.append(np.std(accuracy_nbc[frac])/np.sqrt(10))
		mean_lr.append(np.mean(accuracy_lr[frac]))
		stderr_lr.append(np.std(accuracy_lr[frac])/np.sqrt(10))
		mean_svm.append(np.mean(accuracy_svm[frac]))
		stderr_svm.append(np.std(accuracy_svm[frac])/np.sqrt(10))
	
	'''
	plt.errorbar(t_frac, mean_nbc, stderr_nbc, label="Naive Bayesian")
	plt.errorbar(t_frac, mean_lr, stderr_lr, label="Logistic Regression")
	plt.errorbar(t_frac, mean_svm, stderr_svm, label="Linear SVM")
	plt.xlabel("t_frac") 
	plt.ylabel("Accuracy") 
	plt.legend() 
	plt.savefig("compare.jpg")
	'''
	print("===============================") 
	#print("NBC vs LR") 
	#print(ttest_rel(mean_nbc, mean_lr))
	#print("NBC vs SVM") 
	#print(ttest_rel(mean_nbc, mean_svm))
	print("LR vs SVM") 
	print(ttest_rel(mean_lr, mean_svm))


if __name__ == "__main__": 
	train = pd.read_csv("trainingSet.csv")
	train = train.sample(random_state=18, frac=1)
	train.insert(len(train.columns), "bias", 1)
	trainset = np.array_split(train, 10)

	train_nbc = pd.read_csv("trainingSet_NBC.csv")
	train_nbc = train_nbc.sample(random_state=18, frac=1)
	trainset_nbc = np.array_split(train_nbc, 10)

	learn_lr_svm(train, trainset)
	learn_nbc(train_nbc, trainset_nbc)
	plot()

