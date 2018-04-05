######## !!!!!!!!!!!!
# Assuming x <= .5 for all values for testing


import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.stats
import csv
import matplotlib

# Stores u and m as a tuple, sped up data structure in later versions

# 1. If 1-u is chosen for the index, store 1/m.

################
### Get data ###
################
name_string = 'Academia_Psychology.csv'
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

IC = data[0]
# compute number of layers
num_layers = len(data[0])
# number of years
years = len(data)
# arrange data in "time series" format
data = np.transpose(data)
print(IC)



############################
### Initialize Variables ###
############################

# These values are a guess, but they might not matter?
# Try varying them later
# R = [15, 9, 7, 6, 5, 4]
R = [1/4,1/5,1/6,1/7,1/9,1/15]      # Retirement rate at each level
N = [13,8,5,3,2,1]                  # Number(Ratio) of people at each level
b = .7
L = num_layers - 1
max_len = 0

r = np.zeros(num_layers, dtype = np.float64)
for i in range(num_layers):
    for j in range(i + 1, num_layers):
        r[i] += R[j]*N[j]
    r[i] /= (R[i]*N[i])

homophily = []



############################
### Function Definitions ###
############################

# predicted shape of first half of homophily
def P(x):
    return x

def GetH(t):
    dx = np.zeros(num_layers, dtype = np.float64)
    X = np.zeros(num_layers, dtype = np.float64)
    for i in range(num_layers):
        X[i] = data[i][t]
        dx[i] = data[i][t+1] - data[i][t]
    f = X[L] + dx[L]/R[L]              # f(X[L],X[L-1]) = X[L] + dx[L]/R[L]
    u = X[L]
    v = X[L-1]
    index = int(round(u*1000))
    # P(u) = m*P(1-u)
    # m = f*(1-b)(1-v)/(b*v*(1-f))
    m = f*(1-b)*(1-v)/(b*v*(1-f))
    if(index <= 500):
        homophily.append([index,m])
    else:
        homophily.append([1000-index,1/m])
    fprev = f
    for i in range(L-1, -1, -1):
        # dx[i] = R[i]*((1 + r[i])*f(X[i],X[i-1]) - XX[i] - r[i]*f(X[i+1],X[i]))
        # f(X[i],X[i-1]) = (dx[i]/R[i] + X[i] + r[i]*f(X[i+1],X[i]))(1 + r[i])
        f = (X[i] + r[i]*fprev + dx[i]/R[i])*(1 + r[i])
        u = X[i]
        if(i > 0):
            v = X[i-1]
        else:
            v = .5
        index = int(round(u*1000))
        m = f*(1-b)*(1-v)/(b*v*(1-f))
        if(index <= 500):
            homophily.append([index,m])
        else:
            homophily.append([1000-index,1/m])
        fprev = f



for i in range(years - 1):
    GetH(i)

homophily = np.transpose(homophily)

std = np.std(homophily[1])
mean = np.mean(homophily[1])

minh = mean - 3*std
maxh = mean + 3*std

print(minh)
print(maxh)

ghomophily = [[],[]]    # homophily with outliers removed
rem = 0

for i in range(len(homophily[1])):
    if((homophily[1][i] < maxh) and (homophily[1][i] > minh)):
        ghomophily[0].append(homophily[0][i]/1000)
        ghomophily[1].append(homophily[1][i])
    else:
        rem += 1

print(rem)
#print(ghomophily)




matplotlib.pyplot.scatter(ghomophily[0], ghomophily[1])
plt.show()
