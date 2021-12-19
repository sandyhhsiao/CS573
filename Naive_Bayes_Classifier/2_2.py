import sys 
import pandas as pd 
import re
import matplotlib.pyplot as plt
import numpy as np

class Visualization: 
	def __init__(self, dataFrame): 
		self.dataFrame = dataFrame

	def visualize(self): 
		partner_rating = [attr for attr in self.dataFrame.columns if re.match(r".*_part.*", attr)]
		for col in partner_rating:
			self.computeSuccessRate(col)

	def computeSuccessRate(self, colname): 
		uniqueDict = self.dataFrame[colname].value_counts().to_dict() 
		for k, v in uniqueDict.items(): 
			success =((self.dataFrame[colname] == k) & (self.dataFrame['decision'] == 1)).sum()
			uniqueDict[k] = success/v
		self.plot(colname, uniqueDict)
				
	def plot(self, col, successRate): 
		plt.figure()
		plt.scatter(successRate.keys(), successRate.values())		
		plt.title(col + " success rate") 
		plt.xlabel("Value") 
		plt.ylabel("Success Rate")
		plt.savefig(col+"_success_rate.jpg")

def main(argv): 
	if(len(argv) < 2): 
		print('Missing arguments\n') 
	else: 
		visualObj = Visualization(pd.read_csv(argv[1]))
		visualObj.visualize()

if __name__ == "__main__": 
	main(sys.argv)
