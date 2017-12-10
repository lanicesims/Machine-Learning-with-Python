import pandas as pd
import quandl
import datetime
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

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

#math.ceil rounds everything up. We are trying to predict out 1% out (depending on the number of days, that many days * .01 out).

df['label'] = df[forecast_col].shift(-forecast_out) 

#.shift is pandas method that is used to shift the column in a direction by any number you decide. This new column will be trained against. Each row will be the Adj. Close in 10%

#print(df.head())

#define X and y (generally features is represented as a capital X and labels as a lowercase y)

X = np.array(df.drop(['label'],1)) #df.drop will return a new dataframe with all the features but the label column

#now we want to scale X. Need to scale before putting it through a classifier.

X = preprocessing.scale(X) #using it on real time on real data
X_lately =X[-forecast_out:] #X_lately is the data we are going to predict against. (need to figure out m and b in y = mx + b)

#X_lately does not have a y value attached to it.

X = X[:-forecast_out] #This will slice the array from 0 to length - forecast_out

df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

#take all features and labels (keeping Xs and ys together) and will output X and y training data and X and y testing data. We use X_train and y_train to fit our classifiers

#we are finding a classifier by choosing SKLearn's Linear Regression algorithnm for our ML example. We can change out clf = LinearRegression() to easily to use a different algorithm.

clf = LinearRegression()
#clf = svm.SVR()  this is support vector regression.

clf.fit(X_train, y_train) # using clf.fit to fit or train a classifier; you fit features and labels.

#pickling is the process whereby a Python object hierarchy is converted into a byte stream, and “unpickling” is the inverse operation. We want to save our #classifier so that we dont have to keep training it every time we want to use it. This step will take the most time when you have a lot of data.

with open('linearregression.pickle', 'wb') as f:
    pickle.dump(clf, f)

#We are opening file and dumping the trained classfier in f

#To use classifier 
pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

# Once you have your classifier and the daa stored in the pickle, it might make more sense to comment that code out so you aren't running it everytime you open the file. 

# now we have our classifier to use and predict data

accuracy = clf.score(X_test, y_test) #testing the accuracy of our classifier and see how well the training set compares to the testing data.

#print(accuracy)

#Need to predict based on the X data. 

forecast_set = clf.predict(X_lately) #can pass in a single value or an array of values. You'll make a prediction per value in the array.

print(forecast_set, accuracy, forecast_out) #next 34 days of prices.

df['Forecast'] = np.nan #specifies the column consists of not a number data

#need to find out what the last day. The prediction does not know what date it is for, as we don't have date values. The following code will specify the dates for our x-axis on the graph. 

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

#Now we need to add the dates and forecast values to the dataframe.

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i] #We are taking each forecast and day, and setting those values in the dataframe to make the future features not a number. 

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
