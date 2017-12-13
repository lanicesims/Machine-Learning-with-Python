#The dataset used is from http://archive.ics.uci.edu/ml/datasets.html. We are using the original Wisconsin Breast cancer dataset for our classification example.

import numpy as np 
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd 

df = pd.read_csv('breast-cancer-wisconsin.data.txt') #loading the text file into a dataframe
df.replace('?', -99999, inplace=True) #We know that our dataset has missing values denoted by a question mark, so we want are going to replace these value.s

#Looking at the data in breast-cancer-wisconsin.data.txt, we want to make sure we are only including columns that are necessary for what we are evaluating.T he column 'ID' is not necessary information for determining what type of tumor is present, so we will omit this column. 

df.drop(['id'], 1, inplace=True)
#define X and y

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

#fit the data
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

#test the accuracy
accuracy = clf.score(X_test, y_test)

print(accuracy)

#now we are going to predict 

example_measures = np.array([[4,2,1,1,1,2,3,2,1], [4,2,1,1,2,2,3,2,1]])
example_measures = example_measures.reshape(len(example_measures),-1) #reshaping numpy array in order to use SciKit Learn

prediction = clf.predict(example_measures)
print(prediction)