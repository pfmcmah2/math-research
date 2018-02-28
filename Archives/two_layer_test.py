import matplotlib.pyplot as plt
import numpy as np
import math
import scipy
import scipy.stats
from scipy.integrate import odeint
import csv


### initialize variables ###
num_layers = 2
years = 500
R = [1/4,1/5] #,1/6,1/7,1/9,1/15]   # Retirement rate at each level
N = [13,8] #,5,3,2,1]   # Number of people at each level
X = [0.4,0.3] #,0.2,0.1,0.05,0.01]   # Fraction of women at each level
layer_names = ['undergrad','grad']

b = .5      # Bias
mu = .75     # Mean for gaussian homophily distribution
sigma = .3  # STD for gaussian homophily distribution

Normal = []
# initialize r, the ratio parameter
r = np.zeros(num_layers, dtype = np.float64)
for i in range(num_layers):
    for j in range(i + 1, num_layers):
        r[i] += R[j]*N[j]
    r[i] /= (R[i]*N[i])


### Homophily, either linear or gaussian ###
def P(x):
    # return x
    return scipy.stats.norm(mu, sigma).pdf(x)


Normal = (np.zeros(1000, dtype = np.float64))   # gaussian DP
for j in range(0, 1000):
        Normal[j] = scipy.stats.norm(mu, sigma).pdf(j/1000)

### Fraction of women promoted to layer u from layer v ###
def f(u, v):
    #return b*v*P(u)/(b*v*P(u) + (1 - b)*(1 - v)*P(1 - u))
    return b*v*Normal[math.floor(u*1000)]/(b*v*Normal[math.floor(u*1000)] + (1 - b)*(1 - v)*Normal[math.floor((1 - u)*1000)])



### Rate of change of fraction of women at each layer ###
def dx(XX, t):
    out = np.zeros(num_layers, dtype = np.float64)
    out[0] = R[0]*((1 + r[0])*f(XX[0],.5) - XX[0] - r[0]*f(XX[1],XX[0]))
    out[1] = R[1]*(f(XX[1], XX[0]) - XX[1])
    return out



### Integration over time t = years/100 ###
'''
def intode(XX, t):
    out = [[],[]]
    for i in range(t*100):
        #print(XX)
        #print(dx(RR, rr, XX))
        if(i % 100 == 0):
            out[0].append(XX[0])
            out[1].append(XX[1])
        XX += .01*dx(XX)
    return out
'''



#test = intode(X, years)


T = np.arange(0, years, 1)

test = odeint(dx, X, T)
#test = np.transpose(test)
print(test)

'''
plt.xlabel("Years")
plt.ylabel("Fraction of Women")
plt.title("MATH 492")
for i in range(num_layers):
    plt.plot(T, test[i],label = layer_names[i])
plt.legend()
plt.show()
'''
