import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.stats
import random


### Selecting randomly with bias from a set of X binary elements
# Starting with a fraction of f 0's and 1-f 1's
# Generate a random number from a uniform distribution:
# (0, a) for 0's, (0, b) for 1's
# Expected fraction of 0's selected is f*b/(f*b + (1-f)*a)
Pool = []
Select = []
PoolSize = 10000
SelectSize = 1000
startFraction = .5  # starting fraction of 0's
P0 = .7
P1 = .5
count = 0

print("Target:", startFraction*P0/(startFraction*P0 + (1-startFraction)*P1))

for i in range(PoolSize):
    rand = random.uniform(0, 1)
    if(rand > startFraction):
        Pool.append(1)
    else:
        Pool.append(0)
        count += 1

print("Start:", count/PoolSize)

for i in range(PoolSize):
    if(Pool[i] == 0):
        rand = random.uniform(0, P1)
    else:
        rand = random.uniform(0, P0)
    Select.append([rand, Pool[i]])

Select.sort()

count = 0
for i in range(SelectSize):
    if(Select[i][1] == 0):
        count += 1

print("End:", count/SelectSize)
