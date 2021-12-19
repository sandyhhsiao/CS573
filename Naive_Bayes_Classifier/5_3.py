import sys 
import pandas as pd 
import matplotlib.pyplot as plt
import model 

def main(argv): 
	
	df_train = pd.read_csv("trainSet.csv")
	df_test = pd.read_csv("testSet.csv")
	fracSize = [0.01, 0.1, 0.2, 0.5, 0.6, 0.75, 0.9, 1] 
	training = []
	testing = []
	for frac in fracSize: 		
		print("sample fraction {}".format(frac))
		nbcObj = model.nbc(df_train, frac)
		nbcObj.learnProb(5) 
		nbcObj.evaluate(nbcObj.df, "Training")
		nbcObj.evaluate(df_test, "Testing")
		training.append(nbcObj.accuracy['Training'])
		testing.append(nbcObj.accuracy['Testing'])
	
	model.plotAccuracy("Sample Fraction", fracSize, training, testing)

if __name__ == "__main__": 
	main(sys.argv)
