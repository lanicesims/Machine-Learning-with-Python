#Using our y = mx + b equation:
#First we will calculate for slope

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

# xs = np.array([1,2,3,4,5,6], dtype=np.float64)
# ys = np.array([5,4,6,5,6,7], dtype=np.float64)

#We want to be able to test our algorithm and r-squared value. So we can create a dataset by randomly assign numbers in our dataset using the specific parameters.

def create_dataset(hm, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
        xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

#calculation of the slope (m) and y-int (b)

def best_fit_slope_and_yint(xs,ys):
    m = ( ((mean(xs) * mean(ys)) - mean(xs * ys)) /
    ((mean(xs)**2) - mean(xs * xs)))
    b = mean(ys) - m * mean(xs)
    return m, b

#How good is our best fit line? We can use the coefficient of determination to find out. Using a new function that squares error.

def squared_error(ys_orig, ys_line): #the distance from the y point and the line in question is the error, then we square it. 
    return sum((ys_line - ys_orig)**2)

def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)

xs, ys = create_dataset(40, 6, 2, correlation='neg')

m, b = best_fit_slope_and_yint(xs,ys)

#We can create a line that fits the data we have.

regression_line = [(m*x + b) for x in xs]

#To find out the r-squared value of our regression line:

r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)

#We can use the best fit line to predict values of y.

predict_x = 8
predict_y = (m * predict_x) + b

plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, color='green')
plt.plot(xs, regression_line)
plt.show()