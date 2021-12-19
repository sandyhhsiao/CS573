import sys 
import pandas as pd 
import model

def main(argv): 
	if(len(argv) < 3): 
		print('Missing arguments\n') 
	else: 
		df_train = pd.read_csv(argv[1])
		df_test = pd.read_csv(argv[2])
		frac = float(argv[3]) if len(argv) == 4 else 1
		nbcObj = model.nbc(df_train, frac)
		#nbcObj.learnProb(5) 
		print("Train age 0 decision 0 ")
		print(((df_train['age'] == 0) & (df_train['decision']== 0)).sum() )
		print("Test age 0 decision 0 ")
		print(((df_test['age'] == 0) & (df_test['decision']== 0)).sum() )
		#nbcObj.evaluate(df_train, "Training")
		#nbcObj.evaluate(df_test, "Testing")

if __name__ == "__main__": 
	main(sys.argv)
