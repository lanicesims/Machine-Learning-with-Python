#The dataset used is from http://archive.ics.uci.edu/ml/datasets.html. We are using the original Wisconsin Breast cancer dataset for our classification example.
 
import numpy as np 
from sklearn import preprocessing, cross_validation, neighbors, svm 
import pandas as pd 
 
df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True) 
df.drop(['id'], 1, inplace=True)
 
X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])
 
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

#fit the data
clf = svm.SVC()
clf.fit(X_train, y_train)
 
#test the accuracy
accuracy = clf.score(X_test, y_test)

print(accuracy)

# example_measures = np.array([[4,2,1,1,1,2,3,2,1], [4,2,1,1,2,2,3,2,1]])
# example_measures = example_measures.reshape(len(example_measures),-1)
 
# prediction = clf.predict(example_measures)
# print(prediction) 