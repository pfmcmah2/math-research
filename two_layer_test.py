import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.stats
import csv


### initialize variables ###
num_layers = 2
years = 500
R = [1/4,1/5] #,1/6,1/7,1/9,1/15]   # Retirement rate at each level
N = [13,8] #,5,3,2,1]   # Number of people at each level
X = [0.4,0.3] #,0.2,0.1,0.05,0.01]   # Fraction of women at each level
layer_names = ['undergrad','grad']

b = .5      # Bias
mu = .5     # Mean for gaussian homophily distribution
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
def dx(RR, rr, XX):
    out = np.zeros(num_layers, dtype = np.float64)
    out[0] = RR[0]*((1 + rr[0])*f(XX[0],.5) - XX[0] - rr[0]*f(XX[1],XX[0]))
    out[1] = RR[1]*(f(XX[1], XX[0]) - XX[1])
    return out



### Integration over time t = years/100 ###
def intode(RR, rr, XX, t):
    out = [[],[]]
    for i in range(t*100):
        #print(XX)
        #print(dx(RR, rr, XX))
        if(i % 100 == 0):
            out[0].append(XX[0])
            out[1].append(XX[1])
        XX += .01*dx(RR, rr, XX)
    return out



out = []


for i in range(0, 11):
    Normal = (np.zeros(1000, dtype = np.float64))   # gaussian DP
    for j in range(0, 1000):
            Normal[j] = scipy.stats.norm(mu, sigma).pdf(j/1000)
    out.append([])
    b = .5
    for j in range(0, 11):
        test = intode(R, r, X, years)
        out[i].append(np.std(test[0])+np.std(test[1]))
        b += .05
    mu += .05

#print(out)

x = np.arange(.5, 1.05, .05)
y = np.arange(.5, 1.05, .05)
x, y = np.meshgrid(x, y)

out = np.array(out)
print(out)


plt.pcolormesh(x, y, out)
plt.xlabel("bias")
plt.ylabel("homophily")
plt.colorbar() #need a colorbar to show the intensity scale
plt.show() #boom

myData = out

myFile = open('two_layer_heatmap.csv', 'w')
with myFile:
   writer = csv.writer(myFile)
   writer.writerows(myData)


'''
test = intode(R, r, X, years)
print(np.std(test[0]))
print(np.std(test[1]))
#print(test)


T = np.arange(0, years, 1)
plt.xlabel("Years")
plt.ylabel("Fraction of Women")
plt.title("MATH 492")
for i in range(num_layers):
    plt.plot(T, test[i],label = layer_names[i])
plt.legend()
plt.show()
'''
