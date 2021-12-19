# Naive Bayes Classifier 
### Preprocess 
``` 
python3 preprocess.py dating-full.csv dating.csv
``` 

### Visualization 
```
python3 2_1.py dating.csv
python3 2_2.py dating.csv
```

### Convert Continuous Attributes 
```
python3 discretize.py dating.csv dating-binned.csv
```

### Training-Test Split 
```
python3 split.py dating-binned.csv trainSet.csv testSet.csv
```

### Naive Bayes Classifier 
```
(The model implementation is in model.py)
python3 5_1.py trainSet.csv testSet.csv
python3 5_2.py 
python3 5_3.py 
```
