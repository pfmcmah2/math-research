import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.stats
import csv
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

# Predict a bias value using your prefered method and keep it constant.
# Read in a time series with L layers from csv.
# Can't store values in an array becasue it's to space inefficient.
# Store tuples of the form (u, v, m)


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
#R = [15, 9, 7, 6, 5, 4]
R = [1/4,1/5,1/6,1/7,1/9,1/15]      # Retirement rate at each level
N = [13,8,5,3,2,1]                  # Number(Ratio) of people at each level
b = .4
L = num_layers - 1

mu = .9
sigma = .3

r = np.zeros(num_layers, dtype = np.float64)
for i in range(num_layers):
    for j in range(i + 1, num_layers):
        r[i] += R[j]*N[j]
    r[i] /= (R[i]*N[i])

λ = []



############################
### Function Definitions ###
############################

# predicted shape of homophily
def P(x):
    return 1/(1 + 2.71828**(-lam*(u - v)))
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
    m = f*(1-b)*(1-v)/(b*v*(1-f))

    #test = np.log(m)/(v-u)
    λ.append(m)

    fprev = f
    for i in range(L-1, -1, -1):
        f = (X[i] + r[i]*fprev + dx[i]/R[i])*(1 + r[i])
        u = X[i]
        if(i > 0):
            v = X[i-1]
        else:
            v = .5
        m = f*(1-b)*(1-v)/(b*v*(1-f))

        #test = np.log(m)/(v-u)
        λ.append(m)

        fprev = f



for i in range(years - 1):
    GetH(i)

λ1 = []
rem = 0
for i in range(len(λ)):
    if(λ[i] > 0 and λ[i] < 10000):
        λ1.append(λ[i])
    else:
        rem += 1


print(λ1)
print(np.std(λ1))
print(rem)
