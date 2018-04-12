import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.stats
import csv


### Takes in time series data from one industry and runs our model over varying
### bias and homophily values, computing the square of the difference at each
### layer at each year. This is a brute force method which assumes a normal
### homophily with sigma = .3

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



R = [1/4,1/5,1/6,1/7,1/9,1/15]
N = [13,8,5,3,2,1]

b = .5
lam = 10

L = num_layers - 1
# initialize r, the ratio parameter
r = np.zeros(num_layers, dtype = np.float64)
for i in range(num_layers):
    for j in range(i + 1, num_layers):
        r[i] += R[j]*N[j]
    r[i] /= (R[i]*N[i])


### Fraction of women promoted to layer u from layer v ###
def P(u, v):
    #return b*v*P(u)/(b*v*P(u) + (1 - b)*(1 - v)*P(1 - u))
    return 1/(1 + 2.71828**(-lam*(u - v)))

def f(u, v):
    return b*v*P(u,v)/(b*v*P(u,v) + (1 - b)*(1 - v)*P(1 - u,1 - v))

### Rate of change of fraction of women at each layer ###
def dx(RR, rr, XX):
    out = np.zeros(num_layers, dtype = np.float64)
    out[0] = RR[0]*((1 + rr[0])*f(XX[0],.5) - XX[0] - rr[0]*f(XX[1],XX[0]))
    for i in range(1, L):
        out[i] = RR[i]*((1 + rr[i])*f(XX[i],XX[i-1]) - XX[i] - rr[i]*f(XX[i+1],XX[i]))
    out[L] = RR[L]*(f(XX[L], XX[L-1]) - XX[L])
    return out

### Integration over time t = years/100 ###
def intodediff(RR, rr, XX, t, data):
    out = 0.0
    for i in range(t*100):
        if(i % 100 == 0):
            for j in range(num_layers):
                out += (XX[j] - data[j][round(i/100)])**2
        XX += .01*dx(RR, rr, XX)
    return out



min_val = 1000000
min_param = [0,0,0]
lam = 1

### Turn this into an iterative method finding multiple minimums and running
### another search centered at those found minimums with higher resolution
### hold the temporary found minimums in a prioity queue, select the n smallest
### for the next iteration, n could increase at each layer of iteration
### more iterations -> higher precision, higher n -> more likely to find true
### global min, less likely to fall into local min trap


for i in range(9):
    b = .5
    for j in range(20):
        #out.append([])
        temp = intodediff(R,r,IC,years,data)
        #out[i].append(temp)
        if(temp < min_val):
            min_val = temp
            min_param = [b,lam]
        b += .01
    print(lam, min_val)
    lam += 1


#print(out)
print(min_val, min_param)
