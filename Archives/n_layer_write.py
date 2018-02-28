import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.stats
import csv


### initialize variables ###
num_layers = 6
years = 500
R = [1/4,1/5,1/6,1/7,1/9,1/15]    # Retirement rate at each level
N = [13,8,5,3,2,1]   # Number of people at each level
X = [0.4,0.3,0.2,0.1,0.05,0.01]   # Fraction of women at each level
layer_names = ['undergrad','grad','postdoc','tenure track','tenured','full']

b = .5      # Bias
mu = .5    # Mean for gaussian homophily distribution
sigma = .3  # STD for gaussian homophily distribution

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
        if(i % 100 == 0 and i > 45000):
            for j in range(num_layers):
                out[j].append(XX[j])
        XX += .01*dx(RR, rr, XX)
    return out



### Compute sum of standard devation for time series of each level
### in each hierarchy with varying bias and homophily
### Store values in out[mu index][bias index]
out = []
std_sum = 0
for i in range(0, 6):   ## range of gaussian
    ### create gaussian for this mu
    Normal = (np.zeros(1000, dtype = np.float64))   # gaussian DP
    for j in range(0, 1000):
            Normal[j] = scipy.stats.norm(mu, sigma).pdf(j/1000)

    out.append([])      ## create subarray for this mu
    b = .5              ## set bias

    for j in range(0, 6):   ## range of bias
        std_sum = 0
        test = intode(R, r, X, years)   ## run simulation, store values in test
        for j in range(num_layers):     ## sum std of all layers
            std_sum += np.std(test[j])
        out[i].append(std_sum)
        b += .1
    mu += .1



### Write to csv
myData = np.array(out)
myFile = open('6_hig_50-65_65-80_late.csv', 'w')
with myFile:
   writer = csv.writer(myFile)
   writer.writerows(myData)
