import sys
import pandas as pd 
import discretize
import model


def main(argv):  
	df = pd.read_csv("dating.csv")
	bins = [2, 5, 10, 50, 100, 200]
	training = [] 
	testing = []
	for bin in bins:
		obj = discretize.ContinuousAttr(df.copy(deep=True)) 
		obj.processBin(bin)
		df_test = obj.dataFrame.sample(random_state=47, frac=0.2) 
		df_train = obj.dataFrame.drop(df_test.index) 
		nbcObj = model.NBC(df_train)
		nbcObj.learnProb(bin) 
		print("Bin size: {}".format(bin))
		nbcObj.evaluate(df_train, "Training")
		nbcObj.evaluate(df_test, "Testing")
		training.append(nbcObj.accuracy['Training'])
		testing.append(nbcObj.accuracy['Testing'])
	model.plotAccuracy("Bin Size", bins, training, testing)

if __name__ == "__main__": 
	main(sys.argv)
