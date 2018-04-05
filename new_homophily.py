import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.stats


# Homophily is given by P(u,v) where v is the fraction at the lower level
# and u is the fraction at the higher level.
# P(u) = pdf(u) of the normal distribution with mean v and var sigma (arbitrary)

############################
### initialize variables ###
############################
num_layers = 6                      # Number of layers in hierarchy
years = 500                         # Number of years to simulate
R = [1/4,1/5,1/6,1/7,1/9,1/15]      # Retirement rate at each level
N = [13,8,5,3,2,1]                  # Number(Ratio) of people at each level
#X = [0.4,0.3,0.2,0.1,0.05,0.01]     # Fraction of women at each level (Initial Condition)
X = [0.6,0.7,0.8,0.9,0.95,0.99]
# Label for each layer
layer_names = ['undergrad','grad','postdoc','tenure track','tenured','full']

b = .3     # Bias, <.5 -> favors men, >.5 -> favors women
sigma = .3  # STD for gaussian homophily distribution

sigma2 = sigma**2
sqt = np.sqrt(6.28318*sigma2)

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

def P(u,v):
    #return 1 - abs(u-v)
    return (2.71828**(((u-v)**2)/(-2*sigma2)))/sqt




############################
### Function Definitions ###
############################

### Fraction of women promoted to layer u from layer v ###
def f(u, v):
    # return b*v*P(u)/(b*v*P(u) + (1 - b)*(1 - v)*P(1 - u))
    return b*v*P(u, v)/(b*v*P(u, v) + (1 - b)*(1 - v)*P(1 - u,1 - v))


### Rate of change of fraction of women at each layer ###
def dx(XX):
    out = np.zeros(num_layers, dtype = np.float64)
    out[0] = R[0]*((1 + r[0])*f(XX[0],.5) - XX[0] - r[0]*f(XX[1],XX[0]))
    for i in range(1, L):
        out[i] = R[i]*((1 + r[i])*f(XX[i],XX[i-1]) - XX[i] - r[i]*f(XX[i+1],XX[i]))
    out[L] = R[L]*(f(XX[L], XX[L-1]) - XX[L])
    return out


### Integration over time t = years/100 ###
def intode(XX, t):
    out = []
    for i in range(num_layers):
        out.append([])
    for i in range(t*100):
        #print(XX)
        #print(dx(RR, rr, XX))
        if(i % 100 == 0):
            for i in range(num_layers):
                out[i].append(XX[i])
        XX += .01*dx(XX)
    return out



#####################
### Graph Results ###
#####################



test = intode(X, years)
print(X)
std = 0.
for i in range(num_layers):
    print(np.std(test[i]))

T = np.arange(0, years, 1)
plt.xlabel("Years")
plt.ylabel("Fraction of Women")
plt.ylim(0,1)
plt.title("bias = " "." + str(round(b*10)) + "     " " sigma = " + "." + str(round(sigma*10)))
for i in range(num_layers):
    plt.plot(T, test[i],label = layer_names[i])
plt.legend()
plt.show()
