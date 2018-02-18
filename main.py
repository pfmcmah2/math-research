import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.stats


############################
### initialize variables ###
############################
num_layers = 6                      # Number of layers in hierarchy
years = 500                         # Number of years to simulate
R = [1/4,1/5,1/6,1/7,1/9,1/15]      # Retirement rate at each level
N = [13,8,5,3,2,1]                  # Number(Ratio) of people at each level
X = [0.4,0.3,0.2,0.1,0.05,0.01]     # Fraction of women at each level (Initial Condition)

# Label for each layer
layer_names = ['undergrad','grad','postdoc','tenure track','tenured','full']

b = .5      # Bias, <.5 -> favors men, >.5 -> favors women
mu = .7     # Mean for gaussian homophily distribution
sigma = .3  # STD for gaussian homophily distribution

# Index of highest layer
L = num_layers - 1

# initialize r, the ratio parameter
r = np.zeros(num_layers, dtype = np.float64)
for i in range(num_layers):
    for j in range(i + 1, num_layers):
        r[i] += R[j]*N[j]
    r[i] /= (R[i]*N[i])




############################################
### Homophily, either linear or gaussian ###
############################################

def P(x):
    #return x
    return scipy.stats.norm(mu, sigma).pdf(x) #/scipy.stats.norm(mu, sigma).pdf(mu)

# Store homophily parameter in lookup table to decrease runtime
Normal = np.zeros(1000, dtype = np.float64)   # gaussian List
for i in range(0, 1000):
    Normal[i] = P(i/1000)



############################
### Function Definitions ###
############################

### Fraction of women promoted to layer u from layer v ###
def f(u, v):
    # return b*v*P(u)/(b*v*P(u) + (1 - b)*(1 - v)*P(1 - u))
    return b*v*Normal[math.floor(u*1000)]/(b*v*Normal[math.floor(u*1000)] + (1 - b)*(1 - v)*Normal[math.floor((1 - u)*1000)])


### Rate of change of fraction of women at each layer ###
def dx(RR, rr, XX):
    out = np.zeros(num_layers, dtype = np.float64)
    out[0] = RR[0]*((1 + rr[0])*f(XX[0],.5) - XX[0] - rr[0]*f(XX[1],XX[0]))
    for i in range(1, num_layers-1):
        out[i] = RR[i]*((1 + rr[i])*f(XX[i],XX[i-1]) - XX[i] - rr[i]*f(XX[i+1],XX[i]))
    out[L] = RR[L]*(f(XX[L], XX[L-1]) - XX[L])
    return out


### Integration over time t = years/100 ###
def intode(RR, rr, XX, t):
    out = []
    for i in range(num_layers):
        out.append([])
    for i in range(t*100):
        #print(XX)
        #print(dx(RR, rr, XX))
        if(i % 100 == 0):
            for i in range(num_layers):
                out[i].append(XX[i])
        XX += .01*dx(RR, rr, XX)
    return out



#####################
### Graph Results ###
#####################

test = intode(R, r, X, years)

std = 0.
for i in range(num_layers):
    print(np.std(test[i]))

T = np.arange(0, years, 1)
plt.xlabel("Years")
plt.ylabel("Fraction of Women")
plt.title("MATH 492")
for i in range(num_layers):
    plt.plot(T, test[i],label = layer_names[i])
plt.legend()
plt.show()


'''

G = np.arange(0, 1000, 1)
plt.plot(G/1000, Normal)
plt.legend()
plt.show()'''
