import sys 
import pandas as pd 
import matplotlib.pyplot as plt

class NBC: 
	def __init__(self, df): 
		self.df = df 
		self.decision = self.df.groupby('decision').size()
		self.decision_probs = self.decision/len(self.df.index)
		self.accuracy = {}

	def learnProb(self, discretBin):	
		self.prob = {}
		for col in self.df.columns: 
			if col != "decision":
				count = self.df.groupby(['decision', col]).size().unstack(fill_value=0).stack()
				prob = count/self.decision 	
					
				# zero count smoothing 
				if col == "field":
					binsize = 210 
				elif col == "gender" or col == "samerace": 
					binsize = 2 
				elif col == "race" or col == "race_o": 
					binsize = 5
				else: 
					binsize = discretBin
				for dec in range(2): 
					for bin in range(binsize):
						try: 
							condition = prob[dec][bin]
							if condition == 0: 
								prob[dec][bin] = 1/(self.decision[dec]+binsize)
						except KeyError: 
							prob.loc[(dec, bin)] = 1/(self.decision[dec]+binsize)
				self.prob[col] = prob
				#print(self.prob[col])
	def evaluate(self, dataFrame, type):
		# calculate p(0|x), p(1|x)
		# predict 0 if p(0|x) > p(1|x) 

		correct = 0  
		for index, row in dataFrame.iterrows():
			answer = row['decision']
			predict_0 = self.decision_probs[0] 
			predict_1 = self.decision_probs[1] 
			for col in dataFrame.columns: 
				if col != "decision":
					value = row[col] 
					predict_0 *= self.prob[col][0][value] 
					predict_1 *= self.prob[col][1][value] 
			
			predict = 0 if predict_0 > predict_1 else 1 
			correct = correct+1 if predict == answer else correct
		#print("{} Accuracy : {:.2f} ".format(type, correct/len(dataFrame.index)))
		self.accuracy[type] = correct/len(dataFrame.index)
		
def nbc(df, t_frac): 
	return NBC(df.sample(random_state=32, frac = t_frac)) 

def plotAccuracy(xLabel, xData, train, test):
	fig, ax = plt.subplots() 
	ax.set_title(xLabel+" vs Accuracy") 
	ax.set_xlabel(xLabel) 
	ax.set_ylabel("Accuracy")
	ax.plot(xData, train, label="Training")
	ax.plot(xData, test, label="Testing")
	ax.legend()
	if "fraction" in xLabel.lower():
		name = "fraction" 
	else: 
		name = "bin"
	plt.savefig(name+"_accuracy.jpg") 
