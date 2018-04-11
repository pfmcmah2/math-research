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

# Looking at homophily seperately for each layer

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

mu = .9
sigma = .3

r = np.zeros(num_layers, dtype = np.float64)
for i in range(num_layers):
    for j in range(i + 1, num_layers):
        r[i] += R[j]*N[j]
    r[i] /= (R[i]*N[i])

homophily = []
for i in range(num_layers):
    homophily.append([])
    for j in range(501):
        homophily[i].append([])



############################
### Function Definitions ###
############################

# predicted shape of homophily
def P(x):
    #return scipy.stats.norm(mu, sigma).pdf(x)
    return (x-.5)**2
    #return x

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
        homophily[L][index].append(m)
    else:
        homophily[L][1000-index].append(1/m)
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
            homophily[i][index].append(m)
        else:
            homophily[i][1000-index].append(1/m)
        fprev = f



for i in range(years - 1):
    GetH(i)

#print(homophily)

# each element of ahomophily is a
# list of observed fractions and associated average m values
# by nature of its construction this list is sorted by fraction value
# this is needed for building the final funciton
# taking the average deals with overlapping m values but having very different
# overlapping m values implies that this is not a function so keep that in mind

ahomophily = []
for i in range(num_layers):
    ahomophily.append([[],[]])

for j in range(num_layers):
    for i in range(len(homophily[j])):
        if(len(homophily[j][i]) != 0):
            ahomophily[j][0].append(i/1000)
            ahomophily[j][1].append(np.mean(homophily[j][i]))

print(ahomophily)

'''rhomophily = []         # ahomophily with outliers removed
rem = 0

    mean = np.mean(ahomophily[1])
    std = np.std(ahomophily[1])
    min_val = mean - 3*std
    max_val = mean + 3*std

    rhomophily = [[],[]]


    for i in range(len(ahomophily[0])):
        if(ahomophily[1][i] > min_val and ahomophily[1][i] < max_val):
            rhomophily[0].append(ahomophily[0][i])
            rhomophily[1].append(ahomophily[1][i])
        else:
            rem += 1

print(rem)
'''
# final function, stored in "scatter plot" format
# "fills out" function in back and forth format
# lots of room for creativity here
# last holds last value filled in
fhomophily = []
for i in range(num_layers):
    fhomophily.append([[],[]])

### Assigns pairs based on min difference from a predicted homophily function, P(x)
for j in range(num_layers):
    for i in range(len(ahomophily[j][0])):
        m = ahomophily[j][1][i]
        u = ahomophily[j][0][i]
        l = (m*P(1-u) + P(u))/(m**2 + 1)
        r = l * m
        fhomophily[j][0].append(ahomophily[j][0][i])
        fhomophily[j][1].append(l)
        fhomophily[j][0].append(1 - ahomophily[j][0][i])
        fhomophily[j][1].append(r)

plt.xlim(0,1)
for i in range(num_layers):
    plt.title(layer_names[i])
    matplotlib.pyplot.scatter(fhomophily[i][0], fhomophily[i][1])
    plt.show()
