import pandas as pd

import quandl

#dataframe equals df
df = quandl.get('WIKI/GOOGL')

#print(df.head) - this will call all the features. 

#But we want to determine which features are useful/meaningful and which aren't. To make a list of the features that we want.

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume',]]

#dataframe will now only have the columns Open, High, Low, Close, and Volume. Now let's create relationships between the different features.

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0 #This will give is the margin of change in percent form. 

df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0 

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

print(df.head())