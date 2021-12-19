import sys 
import pandas as pd
import numpy as np
import random

class Node: 
	def __init__(self, df, attrs, depth): 
		self.df = df
		self.size = len(df.index) 
		self.attrs = attrs
		self.split_attr = None
		self.gini = -1
		self.left = None 
		self.right = None
		self.label = None #leaf node only
		self.depth = depth
		
	def setLabel(self): 
		bin_result = np.bincount(self.df['decision']) 
		if len(bin_result) == 1: 
			self.label = self.df['decision'].iloc[0]
		else:
			self.label = 0 if bin_result[0] > bin_result[1] else 1

	def computeGini(self, indices): 
		bin_result = np.bincount(self.df['decision'].iloc[indices])/len(indices) 
		return 1 - np.inner(bin_result, bin_result)

	def computeGiniGain(self, model): 
		index_array = np.arange(self.size)
		decision_gini = self.computeGini(index_array)	
		
		if model == 3: 
			attrs = random.choices(self.attrs, k=int(np.sqrt(len(self.attrs))))
		else: 
			attrs = self.attrs
		for attr in attrs:
			attr_0 = np.where(self.df[attr] == 0)[0]
			attr_1 = np.setdiff1d(index_array, attr_0)
			attr_gini = [self.computeGini(attr_0), self.computeGini(attr_1)]
			attr_prob = [len(attr_0)/self.size, len(attr_1)/self.size]
			gini = decision_gini - np.inner(attr_gini, attr_prob)
			if gini > self.gini: 
				self.gini = gini 
				self.split_attr = attr 

	def split(self): 
		left = np.where(self.df[self.split_attr] == 0)[0]
		right = np.setdiff1d(np.arange(self.size), left)
		attrs = self.attrs.copy() 
		attrs.remove(self.split_attr)
			
		if len(left) > 0: 
			self.left = Node(self.df.iloc[left], attrs, self.depth+1) 
		if len(right) > 0: 
			self.right = Node(self.df.iloc[right], attrs, self.depth+1)  

	def predict(self, data): 
		if self.label != None: 
			return self.label 
		if self.split_attr: 
			if self.left and data[self.split_attr] == 0: 
				return self.left.predict(data)
			elif self.right and data[self.split_attr] == 1: 
				return self.right.predict(data)
			else:
				return random.choice([0, 1]) 

	def printTree(self): 
		if self.label != None:
			print("Leaf size {} label {} attr {} depth {}".format(self.size, self.label, len(self.attrs), self.depth))
		print(self.depth, self.split_attr, self.gini)
		if self.left: 
			self.left.printTree() 
		if self.right: 
			self.right.printTree()

def evaluate(tree, df):
	correct = 0
	for index, row in df.iterrows(): 
		predict = {0:0, 1:0} 
		for node in tree:  
			result = node.predict(row) 
			predict[result] += 1
		
		max_predict = 0 if predict[0] > predict[1] else 1 	
		correct = correct + 1 if max_predict == row['decision'] else correct 
	return correct/len(df.index) 

def buildTree(node, max_depth, min_size, model): 
	if node == None: 
		return 
	if node.attrs == None or len(node.attrs) == 0: 
		node.setLabel()
	elif len(node.attrs) == 1: 
		node.split_attr = node.attrs[0] 
		node.setLabel() 
	elif len(np.bincount(node.df['decision'])) == 1: 
		node.label = node.df['decision'].iloc[0] 
	elif node.depth < max_depth and node.size >= min_size: 
		node.computeGiniGain(model) 
		node.split() 
		buildTree(node.left, max_depth, min_size, model)
		buildTree(node.right, max_depth, min_size, model)
	else: 
		node.setLabel() 
	return node 


def decisionTree(train, test, max_depth, min_size):
	attributes = train.columns.tolist()[:-1]
	root = buildTree(Node(train, attributes, 0), max_depth, min_size, 1)
	tree = [root]
	return evaluate(tree, train), evaluate(tree, test)   	

def bagging(train, test, max_depth, min_size, num_tree):
	attributes = train.columns.tolist()[:-1]
	tree = [] 
	for i in range(num_tree): 
		bag_train = train.sample(replace=True, frac=1) 
		tree.append(buildTree(Node(bag_train, attributes, 0), max_depth, min_size, 2)) 
	return evaluate(tree, train), evaluate(tree, test)   	

def randomForests(train, test, max_depth, min_size, num_tree):
	attributes = train.columns.tolist()[:-1]
	tree = [] 
	for i in range(num_tree): 
		rf_train = train.sample(replace=True, frac=1) 
		tree.append(buildTree(Node(rf_train, attributes, 0), max_depth, min_size, 3)) 
	return evaluate(tree, train), evaluate(tree, test)   

if __name__ == "__main__": 
	if(len(sys.argv) < 4): 
		print("Missing Argument") 
	else: 
		df_train = pd.read_csv(sys.argv[1])
		df_test = pd.read_csv(sys.argv[2])
		model = int(sys.argv[3])
		if model == 1: 
			train_accuracy, test_accuracy = decisionTree(df_train, df_test, 8, 50) 
			print("Training Accuracy DT: {:.2f}".format(train_accuracy))
			print("Testing Accuracy DT: {:.2f}".format(test_accuracy))
		elif model == 2: 
			train_accuracy, test_accuracy = bagging(df_train, df_test, 8, 50, 30) 
			print("Training Accuracy BT: {:.2f}".format(train_accuracy))
			print("Testing Accuracy BT: {:.2f}".format(test_accuracy))
		else:
			train_accuracy, test_accuracy = randomForests(df_train, df_test, 8, 50, 30) 
			print("Training Accuracy RF: {:.2f}".format(train_accuracy))
			print("Testing Accuracy RF: {:.2f}".format(test_accuracy))
