import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

#dataframe equals df
df = quandl.get('WIKI/GOOGL')

#print(df.head) - this will call your columns/features. 

#We want to determine which features are useful/meaningful and which aren't. To make a list of the features that we want, we have the columns Open, High, Low, Close, and Volume:

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume',]]

# Let's create relationships between some of the different features.

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0 #This will give is the margin of change in percent form. 

df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0 #This will give the daily percent change.

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']] #This defines a new dataframe with only the columns/features we care about.

#print(df.head())

# We want to bring in information so that we can now predict the prices (Adj. Close). The only information that we have to go off of is past Adj. Close

forecast_col = 'Adj. Close' 

#This will be our forecasted column for stock prices (Adj. Close)

df.fillna(-99999, inplace=True) 

#NA refers to missing or not available data. With machine learning, we will replace any NaN values with a real number (even though it's an #outlier). This is a better choice than getting rid of columns that don't have all the data 
#
# We can't work with nan data, so the nan data has to be replaced with something. When you call 'df.fillna', this data will be included but #treated like an outlier in dataset.   

#To forecast out (predict future values): 

forecast_out = int(math.ceil(.01*len(df)))

print(forecast_out) 

#math.ceil rounds everything up. We are trying to predict out 1% out (depending on the number of days, that many days * .01 out).

df['label'] = df[forecast_col].shift(-forecast_out) 
df.dropna(inplace=True)

#.shift is pandas method that is used to shift the column in a direction by any number you decide. This new column will be trained against. Each row will be the Adj. Close in 10%

#print(df.head())

#define X and y (generally features is represented as a capital X and labels as a lowercase y)

X = np.array(df.drop(['label'],1)) #df.drop will return a new dataframe with all the features but the label column
y = np.array(df['label'])

#now we want to scale X. Need to scale before putting it through a classifier.

X = preprocessing.scale(X) #using it on real time on real data
y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

#take all features and labels (keeping Xs and ys together) and will output X and y training data and X and y testing data. We use X_train and y_train to fit our classifiers

#we are finding a classifier by choosing SKLearn's Linear Regression algorithnm for our ML example. We can change out clf = LinearRegression() to easily to use a different algorithm.

clf = LinearRegression()
#clf = svm.SVR()  this is support vector regression.

clf.fit(X_train, y_train) # using clf.fit to fit or train a classifier; you fit features and labels.

#now we have our classifier to use and predict data

accuracy = clf.score(X_test, y_test) #testing the accuracy of our classifier and see how well the training set compares to the testing data.

print(accuracy)