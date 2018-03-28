import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.stats

# Predict a bias value using your prefered method and keep it constant.
# Read in a time series with L layers from csv. At each year we can extract
# L+1 f(u,v) values, from each we get a P(u) = m * P(1-u) relationship. For
# each, store the m in an array indexed by min(u, 1-u)^1. Resolution can vary,
# example of an index value => (round(1000*u) or round(1000*(1-u))), in this
# case the array indices span 0 to 500, size = 501. No matter the resolution,
# colissions can't be fully eliminated, handle colissions with a list.
# If there is high variation within one list, there is not a consitent
# homophily function => hypothesis might be wrong?


# 1. If 1-u is chosen for the index, store 1/m.
