import sys 
import pandas as pd 
import re

class ContinuousAttr: 
	def __init__(self, df): 
		self.dataFrame = df
		self.process_col = []
		for col in self.dataFrame.columns: 
			if col not in ['gender', 'race', 'race_o', 'samerace', 'field', 'decision']:
				self.process_col.append(col)

	def processBin(self, binsize): 
		self.adjustMax() 
		self.divideBin(binsize)

	def adjustMax(self):
		for row in range(0, len(self.dataFrame.index)):
			for col in self.process_col:
				if col == 'age' or col == 'age_o': 
					if self.dataFrame.at[row, col] > 58: 
						self.dataFrame.at[rol, col] = 58
				elif self.dataFrame.at[row, col] > 10: 
					self.dataFrame.at[row, col] = 10
	
	def divideBin(self, binsize):
		for col in self.process_col:
			if re.match(r".*age.*", col):
				min = 18
				max = 58 
			elif re.match(r"pref_o.*", col) or re.match(r".*_important", col):
				min = 0 
				max = 1
			elif col == 'interests_correlate': 
				min = -1 
				max = 1 
			else: 
				min = 0
				max = 10 
			self.divide(min, max, col, binsize)
	
	def divide(self, min, max, column, binsize):
		width = (max-min)/binsize 
		binCount = [0]*binsize
		binMaxValue = [] 
		for i in range(1, binsize+1):
			binMaxValue.append(min + width*i)
		
		for row in range(0, len(self.dataFrame.index)): 
			value = self.dataFrame.at[row, column]
			for bin in range(binsize): 
				if value <= binMaxValue[bin]: 
					break 
						
			binCount[bin] += 1 
			self.dataFrame.at[row, column] = int(bin)
		self.dataFrame[column] = self.dataFrame[column].astype(int)
#			if value == max:
#				print(value, self.dataFrame.at[row, column])
		#print(column + ": " + str(binCount)) 


def main(argv):
	if(len(argv) < 3): 
		print("Missing input arguments") 
	else: 
		'''
		if len(argv) == 4: 
			binsize = int(argv[3]) 
		else: 
			binsize = 5 
		'''
		conObj = ContinuousAttr(pd.read_csv(argv[1]))
		conObj.processBin(5) 
		conObj.dataFrame.to_csv( argv[2], index=False)



if __name__ == "__main__": 
	main(sys.argv)
