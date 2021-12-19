import sys 
import pandas as pd
import numpy as np

def sigmoid(x): 
	return 1/(1+ np.exp(-x))

def evaluate_lr(df, decision, weight): 
	correct_predict = 0
	for i, row in df.iterrows():
		predict = np.dot(weight, row.T)
		predict = sigmoid(predict)
			 
		if predict >= 0.5 and decision[i] == 1:
			correct_predict += 1
		if predict < 0.5 and decision[i] == 0:
			correct_predict += 1
	return correct_predict/df.index.size

def learn_lr(df, decision):  
	weight = np.zeros(len(df.columns)) 
	regu_param = 0.01  
	max_iteration = 500 
	tol = 1e-6
	norm_diff = 1
	step_size = 0.01 
	iteration = 0 
	norm_prev = 0
	while iteration < max_iteration and norm_diff > tol:
		trans = np.dot(weight, df.T)
		#logistic 
		result = sigmoid(trans) 
		#gradient 
		result_diff = result-decision 
		dW = np.dot(result_diff.T, df) + regu_param*weight
		weight -= step_size*dW 
		
		norm = 0.5*regu_param*np.sum(weight**2)
		if iteration != 0: 
			norm_diff = abs(norm - norm_prev)  
		norm_prev = norm 		
		iteration += 1 
	return weight

def lr(train, train_decision, test, test_decision):  
	#learn model 
	weight = learn_lr(train, train_decision)
	
	#evaluate
	print("Training Accuracy LR: {:.2f}".format(evaluate_lr(train, train_decision, weight))) 
	accuracy = evaluate_lr(test, test_decision, weight)
	print("Testing Accuracy LR: {:.2f}".format(accuracy))
	return accuracy


def sign(output): 
	for i in range(len(output)): 
		if(output[i] > 0):  
			output[i] = 1 
		else: 
			output[i] = -1 
	return output

def evaluate_svm(df, decision, weight): 
	decision.replace(to_replace=0, value=-1, inplace=True)
	correct_predict = 0
	for i, row in df.iterrows():
		predict = np.dot(weight, row.T)
		predict = sigmoid(predict)
		if predict > 0 and decision[i] == 1:
			correct_predict += 1
		if predict <= 0 and decision[i] == -1:
			correct_predict += 1
	return correct_predict/df.index.size

def learn_svm(df, decision):  
	weight = np.zeros(len(df.columns)) 
	regu_param = 0.01  
	max_iteration = 500 
	tol = 1e-6
	norm_diff = 1
	step_size = 0.01
	iteration = 0 
	norm_prev = 0
	decision.replace(to_replace=0, value=-1, inplace=True)
	while iteration < max_iteration and norm_diff > tol:
		result = np.dot(weight, df.T)
		result = sign(result) 
		#gradient  
		tmp = np.zeros(len(weight))
		idx = 0
		for index, row in df.iterrows():
			if result[idx]*decision[index] < 1:
				tmp -= row*decision[index] 
			idx += 1
		tmp /= df.index.size
		dW = regu_param*weight - tmp 
		weight -= step_size*dW 
		norm = 0.5*regu_param*np.sum(weight**2)
		if iteration != 0: 
			norm_diff = abs(norm - norm_prev)  
		norm_prev = norm 		
		iteration += 1 
	return weight 

def svm(train, train_decision, test, test_decision):  
	#learn model 
	weight = learn_svm(train, train_decision)
	
	#evaluate
	print("Training Accuracy SVM: {:.2f}".format(evaluate_svm(train, train_decision, weight)))
	accuracy = evaluate_svm(test, test_decision, weight)
	print("Testing Accuracy SVM: {:.2f}".format(accuracy))
	return accuracy 
	
if __name__ == "__main__": 
	if(len(sys.argv) < 4): 
		print("Missing Argument") 
	else: 
		df_train = pd.read_csv(sys.argv[1])
		df_test = pd.read_csv(sys.argv[2])
		df_train.insert(len(df_train.columns), "bias", 1)
		df_test.insert(len(df_test.columns), "bias", 1)
		#split data into features and decision 
		df_train_decision = df_train.pop('decision')
		df_test_decision = df_test.pop('decision')

		model = int(sys.argv[3])
		if model == 1: 
			lr(df_train, df_test) 
		else:
			svm(df_train, df_test)
