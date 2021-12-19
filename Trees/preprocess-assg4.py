import sys 
import pandas as pd
import re 

class Preprocess: 
	def __init__(self, df): 
		self.df = df.head(6500).drop(columns=['race', 'race_o', 'field'])	
	
	def process(self):  
		participant_score = [attr for attr in self.df.columns if re.match(r".*_important", attr)]		
		partner_score = [attr for attr in self.df.columns if re.match(r"pref_o_.*", attr)]
		self.encodeValues("gender")
		self.normalize(participant_score) 
		self.normalize(partner_score) 
		self.discretize()

	def encodeValues(self, colname):
		self.df = self.df.sort_values(colname) 
		labels = self.df[colname].astype('category').cat.categories.tolist() 
		map = {k : v for k, v in zip(labels, list(range(0, len(labels))))}
		self.df[colname].replace(map, inplace=True)

	def normalize(self, columns): 
		col_sum = self.df[columns].sum(axis=1)
		self.df[columns] = self.df[columns].div(col_sum, axis='index')		

	def discretize(self): 
		column = self.df.columns.values.tolist()
		column.remove('gender') 
		column.remove('samerace')
		column.remove('decision')

		for col in column: 
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

			for row in range(len(self.df.index)): 
				if self.df.at[row, col] > max:  
					self.df.at[row, col] = max
			mean = (min+max)/2
			self.df[col] = pd.cut(x=self.df[col], bins=[min, mean, max], labels=[0,1], include_lowest=True) 
		print(self.df['gaming'].to_numpy())
	def split(self): 
		df_test = self.df.sample(random_state = 47, frac = 0.2) 
		df_test.to_csv("testSet.csv", index=False) 
		df_train = self.df.drop(df_test.index) 
		df_train.to_csv("trainingSet.csv", index=False)

def main(argv): 
	processObj = Preprocess(pd.read_csv(argv[1])) #"dating-full.csv")) 
	processObj.process()
	processObj.split()

if __name__ == "__main__": 
	main(sys.argv)


