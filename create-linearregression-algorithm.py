#Using our y = mx + b equation:
#First we will calculate for slope

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

xs = np.array([1,2,3,4,5,6], dtype=np.float64)
ys = np.array([5,4,6,5,6,7], dtype=np.float64)

#calculation of the slope (m) and y-int (b)

def best_fit_slope_and_yint(xs,ys):
    m = ( ((mean(xs) * mean(ys)) - mean(xs * ys)) /
    ((mean(xs)**2) - mean(xs * xs)))
    b = mean(ys) - m * mean(xs)
    return m, b

m, b = best_fit_slope_and_yint(xs,ys)

print(m, b)

#We can create a line that fits the data we have.

regression_line = [(m*x + b) for x in xs]

plt.scatter(xs, ys)
plt.plot(xs, regression_line)
plt.show()