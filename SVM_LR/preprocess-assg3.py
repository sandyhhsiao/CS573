import sys 
import pandas as pd
import re 

class Preprocess: 
	def __init__(self, dataFrame): 
		self.dataFrame = dataFrame.head(6500)  
	
	def process(self):  
		attributes = ['gender', 'race', 'race_o', 'field']
		participant_score = [attr for attr in self.dataFrame.columns if re.match(r".*_important", attr)]		
		partner_score = [attr for attr in self.dataFrame.columns if re.match(r"pref_o_.*", attr)]
		self.stripQuotes(attributes[1:]) 
		self.toLower(attributes[-1])
		self.encodeValues(attributes)
		self.normalize(participant_score) 
		self.normalize(partner_score) 
		decision_col = self.dataFrame.pop('decision') 
		self.dataFrame.insert(len(self.dataFrame.columns), 'decision', decision_col)

	def stripQuotes(self, columns):
		self.stripCount = 0
		for col in columns: 
			self.dataFrame[col] = self.dataFrame[col].apply(self.strip)

	def strip(self, data): 
		if re.match(r"^\'.*\'$", data):
			data = data.replace("'", "") 
			self.stripCount += 1
		return data

	def toLower(self, colname): 
		self.lowerCount = 0
		self.dataFrame[colname] = self.dataFrame[colname].apply(self.lower)

	def lower(self, data): 
		if re.match(r".*[A-Z].*", data):
			data = data.lower() 
			self.lowerCount += 1
		return data 

	def encodeValues(self, columns): 
		self.dataFrame = self.dataFrame.sort_values(columns) 
		typeDict = {'gender': 'female', 'race' : 'Black/African American', 'race_o' : 'Other', 'field' : 'economics' }
		dropCol = []
		for col in columns: 
			colValues = self.dataFrame[col].astype('category').cat.categories.tolist() 
			dropColVal = col+'_'+colValues[-1]
			dropCol.append(dropColVal)
			targetIndex = colValues.index(typeDict[col])
			targetVec = [0] * (len(colValues)-1) 
			if(targetIndex != len(targetVec)): 
				targetVec[targetIndex] = 1	 
			print("Mapped vector for {} in column {}: {}".format(typeDict[col], col, targetVec))
		self.dataFrame = pd.get_dummies(self.dataFrame).drop(columns=dropCol)

	def normalize(self, columns): 
		col_sum = self.dataFrame[columns].sum(axis=1)
		self.dataFrame[columns] = self.dataFrame[columns].div(col_sum, axis='index')		
	def split(self): 
		df_test = self.dataFrame.sample(random_state = 25, frac = 0.2) 
		df_test.to_csv("testSet.csv", index=False) 
		df_train = self.dataFrame.drop(df_test.index) 
		df_train.to_csv("trainingSet.csv", index=False)

def main(argv): 
	processObj = Preprocess(pd.read_csv("dating-full.csv")) 
	processObj.process()
	processObj.split()

if __name__ == "__main__": 
	main(sys.argv)


