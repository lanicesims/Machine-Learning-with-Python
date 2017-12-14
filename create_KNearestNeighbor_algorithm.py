import numpy as np 
from math import sqrt
import matplotlib.pyplot as plt 
import warnings
from matplotlib import style
from collections import Counter
style.use('fivethirtyeight')

#calculate euclidean distance

# plot1 =  [1,3]
# plot2 = [2,5]

# euclidean_distance = sqrt( (plot1[0]-plot2[0])**2 + (plot1[1]-plot2[1])**2 ) Unfortunately this isn't a fast calculation so isn't entiely useful.

dataset = {'k': [[1,2], [2,3], [3,1]], 'r':[[6,5], [7,7], [8,6]]} #Created two classes and their features.
new_features = [5,7]

#Loop that graphs the dataset and new_features.

# for i in dataset:
#     for ii in dataset[i]:
#         plt.scatter(ii[0],ii[1], s=100, color=i)

# This is how the loop could be written in one line: 
# 
# [[plt.scatter(ii[0],ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]

# plt.scatter(new_features[0], new_features[1], s=100)
# plt.show()

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')

# We want to create a list of distances. The goal is to calculate the K nearest neighbors, their class, and what is the class of whatever we are predicting. # We can see which data points are close to each other. 
    
    distances = []
    for group in data:
        for features in data[group]:
            #euclidean_distance - np.sqrt(np.sum((np.array(features) - np.array(predict))**2)) - numpy array's version of euclidean distance. 
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict)) #this is the fastest way to write euclidean distance with np
            distances.append([euclidean_distance, group]) 
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    print(Counter(votes).most_common(1))
    return vote_result

result = k_nearest_neighbors(dataset, new_features, k=3)
print(result)