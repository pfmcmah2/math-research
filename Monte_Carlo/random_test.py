import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.stats
import random


# want select fraction to be sF*P0/(sF*P0 + (1-sF)*P1)
Pool = []
Select = []
PoolSize = 10000
SelectSize = 1000
startFraction = .5  # starting fraction of 0's
P0 = .9
P1 = .5
count = 0

for i in range(PoolSize):
    rand = random.uniform(0, 1)
    if(rand > startFraction):
        Pool.append(1)
    else:
        Pool.append(0)
        count += 1

print(count/PoolSize)

for i in range(PoolSize):
    if(Pool[i] == 0):
        rand = random.uniform(0, 1 - P0)
    else:
        rand = random.uniform(0, 1 - P1)
    Select.append([rand, Pool[i]])

Select.sort()

count = 0
for i in range(SelectSize):
    if(Select[i][1] == 0):
        count += 1

print(count/SelectSize)
