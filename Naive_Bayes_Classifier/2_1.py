import sys 
import pandas as pd 
import re
import matplotlib.pyplot as plt
import numpy as np

class Visualization: 
	def __init__(self, dataFrame): 
		self.dataFrame = dataFrame
		self.participant_score = [attr for attr in dataFrame.columns if re.match(r".*_important", attr)]

	def visualize(self): 
		self.meanScores = []
		self.getMean(0)
		self.getMean(1) 
		self.plot() 

	def getMean(self, gender): 
		df = self.dataFrame.loc[self.dataFrame['gender'] == gender]
		score = []
		for col in self.participant_score: 
			score.append(df[col].mean())
		self.meanScores.append(score)		

	def plot(self): 
		label = [] 
		for col in self.participant_score: 
			label.append(col[:-10])
		
		fig, ax = plt.subplots() 
		index = np.arange(len(self.participant_score))
		ax.bar(index + 0.0, self.meanScores[0] , color = 'lightpink', width = 0.4, label = 'female')
		ax.bar(index + 0.4, self.meanScores[1] , color =
 'royalblue', width = 0.4, label = 'male')
		ax.set_title('Mean Scores by Gender')
		ax.set_ylabel('Scores')
		ax.set_xlabel('Preference Score of Participant')
		ax.set_xticks(index+0.2)
		ax.set_xticklabels(label)
		ax.legend()
		plt.tick_params(labelsize=8)
		plt.savefig('mean_gender.jpg')

def main(argv): 
	if(len(argv) < 2): 
		print('Missing arguments\n') 
	else: 
		visualObj = Visualization(pd.read_csv(argv[1]))
		visualObj.visualize()

if __name__ == "__main__": 
	main(sys.argv)
