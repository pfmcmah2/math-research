import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.stats
import csv
import matplotlib

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

################
### Get data ###
################
name_string = 'Academia_Engineering.csv'
directory = 'Data/'

# open file
with open(directory + name_string, newline='') as myFile:
    reader = csv.reader(myFile)
    reader = list(reader)

# get layer names
layer_names = []
for j in range(len(reader[0])):
    layer_names.append(reader[0][j])

# get data
data = []
for i in range(1, len(reader)):
    data.append([])
    for j in range(len(reader[0])):
        if(reader[i][j] != ''):
            data[i-1].append(float(reader[i][j]))

data = data[5:]

IC = data[0]
# compute number of layers
num_layers = len(data[0])
# number of years
years = len(data)
# arrange data in "time series" format
data = np.transpose(data)
print(data)



############################
### Initialize Variables ###
############################

# These values are a guess, but they might not matter?
# Try varying them later
#R = [1/4,1/8,2/6,2/7,2/9,2/15]
R = [1/4,1/5,1/6,1/7,1/9,1/15]      # Retirement rate at each level
N = [13,8,5,3,2,1]                  # Number(Ratio) of people at each level
mu = .7
sigma = .3
L = num_layers - 1
max_len = 0

r = np.zeros(num_layers, dtype = np.float64)
for i in range(num_layers):
    for j in range(i + 1, num_layers):
        r[i] += R[j]*N[j]
    r[i] /= (R[i]*N[i])

bias = []
for i in range(num_layers):
    bias.append([])

def Normal(x):
    return scipy.stats.norm(mu, sigma).pdf(x)
    #return x
    #return 1

P = np.zeros(1001, dtype = np.float64)   # gaussian List
for i in range(0, 1001):
    P[i] = Normal(i/1000)

############################
### Function Definitions ###
############################

def GetB(t):
    dx = np.zeros(num_layers, dtype = np.float64)
    X = np.zeros(num_layers, dtype = np.float64)
    for i in range(num_layers):
        X[i] = data[i][t]
        dx[i] = data[i][t+1] - data[i][t]
    f = X[L] + dx[L]/R[L]              # f(X[L],X[L-1]) = X[L] + dx[L]/R[L]
    u = X[L]
    v = X[L-1]
    b = f*(1-v)*P[round(1000*(1-u))]/(f*P[round(1000*(1-u))]*(1-v) + v*P[round(1000*u)] - f*v*P[round(1000*u)])
    bias[L].append(b)
    fprev = f
    for i in range(L-1, -1, -1):
        f = (X[i] + r[i]*fprev + dx[i]/R[i])*(1 + r[i])
        u = X[i]
        if(i > 0):
            v = X[i-1]
        else:
            v = .5
        b = f*(1-v)*P[round(1000*(1-u))]/(f*P[round(1000*(1-u))]*(1-v) + v*P[round(1000*u)] - f*v*P[round(1000*u)])
        bias[i].append(b)
        fprev = f

def normalize(exb):
    max_val = 0
    min_val = 1
    for i in range(num_layers):
        for j in range(years - 1):
            max_val = max(max_val, bias[i][j])
            min_val = min(min_val, bias[i][j])

    if(min_val < 0):
        max_val -= min_val
    for i in range(num_layers):
        for j in range(years - 1):
            if(min_val < 0):
                bias[i][j] -= min_val
            bias[i][j] = exb*bias[i][j]/max_val

def removeOutliers():
    std = []
    mean = []
    for i in range(num_layers):
        std.append(np.std(bias[i]))
        mean.append(np.mean(bias[i]))

    rem = 0
    for i in range(num_layers):
        for j in range(years - 1):
            if((bias[i][j] < mean[i] - std[i] or bias[i][j] > mean[i] + std[i]) and j > 0):
                bias[i][j] = bias[i][j - 1]
                rem += 1
    print('rem =', rem)




for i in range(years - 1):
    GetB(i)


normalize(.4)
removeOutliers()

T = np.arange(0, years - 1, 1)
for i in range(num_layers):
    plt.plot(T, bias[i], label = layer_names[i])
#plt.ylim(0,1)
plt.legend()
plt.show()
