import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.stats
import csv

name_string = 'Film_Industry.csv'
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



'''
T = np.arange(0, years, 1)
plt.xlabel("Years")
plt.ylabel("Fraction of Women")
plt.title("MATH 492")
for i in range(num_layers):
    plt.plot(T, data[i],label = layer_names[i])
plt.legend()
plt.show()
'''

R = [1/4,1/5,1/6,1/7,1/9,1/15]
N = [13,8,5,3,2,1]

b = .1
mu = .6
sigma = .3

L = num_layers - 1
# initialize r, the ratio parameter
r = np.zeros(num_layers, dtype = np.float64)
for i in range(num_layers):
    for j in range(i + 1, num_layers):
        r[i] += R[j]*N[j]
    r[i] /= (R[i]*N[i])
### Initialize gaussian array
Normal = []
Normal = (np.zeros(1000, dtype = np.float64))   # gaussian DP
for j in range(0, 1000):
        Normal[j] = scipy.stats.norm(mu, sigma).pdf(j/1000)
### Fraction of women promoted to layer u from layer v ###
def f(u, v):
    #return b*v*P(u)/(b*v*P(u) + (1 - b)*(1 - v)*P(1 - u))
    return b*v*Normal[math.floor(u*999)]/(b*v*Normal[math.floor(u*999)] + (1 - b)*(1 - v)*Normal[math.floor((1 - u)*999)])
### Rate of change of fraction of women at each layer ###
def dx(RR, rr, XX):
    out = np.zeros(num_layers, dtype = np.float64)
    out[0] = RR[0]*((1 + rr[0])*f(XX[0],.5) - XX[0] - rr[0]*f(XX[1],XX[0]))
    for i in range(1, L):
        out[i] = RR[i]*((1 + rr[i])*f(XX[i],XX[i-1]) - XX[i] - rr[i]*f(XX[i+1],XX[i]))
    out[L] = RR[L]*(f(XX[L], XX[L-1]) - XX[L])
    return out
### Integration over time t = years/100 ###
def intode(RR, rr, XX, t):
    out = []
    for i in range(num_layers):
        out.append([])

    for i in range(t*100):
        if(i % 100 == 0):
            for j in range(num_layers):
                out[j].append(XX[j])
        XX += .01*dx(RR, rr, XX)
    return out

test = intode(R,r,IC,years)

diff = 0.0
for j in range(num_layers):
    for i in range(years):
        diff += abs(test[j][i] - data[j][i])

print(diff)
