import pandas as pd
import quandl
import math

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

#NA refers to missing or not available data. In the real world, one can miss data (say a column is missing). In machine learning, we can't work with nan data, so the nan data has to be replaced with something. When you call 'df.fillna', this data will be included but treated like an outlier in dataset.   

#To forecast out: 

forecast_out = int(math.ceil(.01*len(df))) 

#math.ceil rounds everything up. We are trying to predict out 1% out.

df['label'] = df[forecast_col].shift(-forecast_out) 

#.shift is pandas method that is used to shift the column in a direction by any number you decide. This new column will be trained against. Each row will be the Adj. Close in 10%

print(df.head())