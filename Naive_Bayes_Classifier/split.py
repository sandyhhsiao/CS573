import sys 
import pandas as pd 

def main(argv): 
	if(len(argv) < 4): 
		print('Missing arguments\n') 
	else: 
		df = pd.read_csv(argv[1])
		df_test = df.sample(random_state = 25, frac = 0.2)
		df_test.to_csv(argv[3], index=False)
		df_train = df.drop(df_test.index)
		df_train.to_csv(argv[2], index=False)

if __name__ == "__main__": 
	main(sys.argv)
