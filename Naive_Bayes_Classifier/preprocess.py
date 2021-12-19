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

	def stripQuotes(self, columns):
		self.stripCount = 0
		for col in columns: 
			self.dataFrame[col] = self.dataFrame[col].apply(self.strip)
		print("Quotes removed from {} cells.".format(self.stripCount))

	def strip(self, data): 
		if re.match(r"^\'.*\'$", data):
			data = data.replace("'", "") 
			self.stripCount += 1
		return data

	def toLower(self, colname): 
		self.lowerCount = 0
		self.dataFrame[colname] = self.dataFrame[colname].apply(self.lower)
		print("Standardized {} cells to lower case.".format(self.lowerCount))

	def lower(self, data): 
		if re.match(r".*[A-Z].*", data):
			data = data.lower() 
			self.lowerCount += 1
		return data 

	def encodeValues(self, columns): 
		self.dataFrame = self.dataFrame.sort_values(columns, ascending = True) 
		typeDict = {'gender': 'male', 'race' : 'European/Caucasian-American', 'race_o' : 'Latino/Hispanic American', 'field' : 'law' }
		self.encodeMap = {}
		for col in columns:
			self.encode(col, typeDict)

	def encode(self, colname, typeDict):
		labels = self.dataFrame[colname].astype('category').cat.categories.tolist() 
		print(len(labels))
		self.encodeMap[colname] = {k : v for k, v in zip(labels, list(range(0, len(labels))))}
		self.dataFrame[colname].replace(self.encodeMap[colname], inplace=True)
		print("Value assigned for {} in column {}: {}".format(typeDict[colname], colname, self.encodeMap[colname][typeDict[colname]]))

	def normalize(self, columns): 
		col_sum = self.dataFrame[columns].sum(axis=1)
		self.dataFrame[columns] = self.dataFrame[columns].div(col_sum, axis='index')		
		for col in columns: 
			print("Mean of {}: {:.2f}".format(col, self.dataFrame[col].mean()))

def main(argv): 
	if(len(argv) < 3): 
		print('Missing input arguments\n') 
	else: 
		processObj = Preprocess(pd.read_csv(argv[1])) 
		processObj.process()
		processObj.dataFrame.to_csv(argv[2], index=False)

if __name__ == "__main__": 
	main(sys.argv)


