import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.stats
import csv
import matplotlib


b = .47
lam = 4.7


name_string = 'Academia_Engineering.csv'
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


# Index of highest layer
L = num_layers - 1

R = [1/4,1/5,1/6,1/7,1/9,1/15]  # Retirement rate at each level
N = [13,8,5,3,2,1]              # Number(Ratio) of people at each level


# initialize r, the ratio parameter
r = np.zeros(num_layers, dtype = np.float64)
for i in range(num_layers):
    for j in range(i + 1, num_layers):
        r[i] += R[j]*N[j]
    r[i] /= (R[i]*N[i])




############################################
### Homophily, either linear or gaussian ###
############################################

def P(u, v):
    #return x
    return 1/(1 + 2.71828**(-lam*(u - v)))




############################
### Function Definitions ###
############################

### Fraction of women promoted to layer u from layer v ###
def f(u, v):
    return b*v*P(u,v)/(b*v*P(u,v) + (1 - b)*(1 - v)*P(1 - u,1 - v))


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
    global error
    out = []
    for i in range(num_layers):
        out.append([])
    for i in range(t*100):
        #print(XX)
        #print(dx(RR, rr, XX))
        if(i % 100 == 0):
            for j in range(num_layers):
                out[j].append(XX[j])
                error += (XX[j] - data[j][round(i/100)])**2
        XX += .01*dx(XX)
    return out



#####################
### Graph Results ###
#####################
#b_string = "." + str(round(b*100))
#h_string = "." + str(round(mu*100))
color = ['bo', 'ro', 'bo']
error = 0.0

test = intode(IC, years)
std = 0.

T = np.arange(0, years, 1)
plt.xlabel("Years")
plt.ylabel("Fraction of Women")
#plt.ylim(0,1)
#plt.title(name_string[:(len(name_string)-4)] + "   error = " + str(error))
#plt.title(name_string[:(len(name_string)-4)])
plt.title("Academia Engineering")
for i in range(num_layers):
    plt.plot(T, test[i],label = layer_names[i])
    #plt.plot(T, data[i])
    matplotlib.pyplot.scatter(T, data[i])
plt.legend()
plt.show()
