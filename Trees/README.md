# Decision Trees, Bagging, Random Forests  
### Preprocess 
``` 
python3 preprocess-assg4.py dating-full.csv
``` 

### Implementation  
```
Decision Trees
python3 trees.py trainingSet.csv testSet.csv 1 
Bagging
python3 trees.py trainingSet.csv testSet.csv 2 
Random Forests
python3 trees.py trainingSet.csv testSet.csv 3 
```

### The Influence of Tree Depth on Classifier Performance
```
Import trees.py
python3 cv_depth.py 
```
### Compare Performance of Different Models  
```
Import trees.py
python3 cv_frac.py 
```
### The Influence of Number of Trees on Classifier Performance
```
Import trees.py
python3 cv_numtrees.py 
```
