import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.stats
import random

### WARNING!!! The situation in which x people retire from layer y, and there
#   are <x people in layer y-1 is not "yet" accounted for.
#   Simple fix is running to full layer promote repeatedly until all layers are full

# PERSON CLASS
#   1 int 1 bool, [years to retirement (yr), Gender (g)]
# LAYER CLASS
#   1 list of persons 1 float, [people in layer[], fraction of women]
#       By construction list is ordered by seniority
#           Acts a priority queue with no aditional overhead
# Layer has a preset size, cannot be changed
# At each year:
#   Decrement all persons yr by 1
#       If yr = 0: remove person
#   Starting with top layer fill all vacant spots using model
#       Promoting a person is done by copying them into the higher layer and
#       deleting them from the lower layer

# Notes:
#   Choosing an initial yr value for all people is tricky
#       Could be based on layer
#   Choosing yr for a new person is done randomly, exponential distribution?
#   Only ratio of layer size matters, larger layers => higher "resolution"

λ = 1
b = .5
Layers = []
LayerSize = [13,8,5,3,2,1]
### Homophily Function ###
def P(u, v):
    # sigmoid funciton λ
    return 1/(1 + 2.71828**(-λ*(u - v)))

### Fraction of women promoted to layer u from layer v ###
def f(u, v):
    return b*v*P(u,v)/(b*v*P(u,v) + (1 - b)*(1 - v)*P(1 - u,1 - v))


### create a new person ###
# inputs:   bool Gender;    0 = female, 1 = male
# outputs:  person struct
def newPerson(gender):
    lifeSpan = random.randint(10,20)
    #level = 0
    return [lifeSpan, gender]

### choose person to be promoted ###
# inputs:   int dest;  index of higher layer
#           int vacancies;  number of vacancies
# outputs:  int[vacancies] idx;  list of indexes of people to be promoted
def selectPromote(dest, vacancies):
    # compute likelyhood of a woman to apply
    # Layers[dest][1] holds fraction of women at layer dest
    homophily = P(Layers[dest][1], Layers[dest-1][1])

    count = 0   # number of people selected to be promoted
    idx = []    # indexes of people to be promoted
    # loop until enough people have been found
    while(count < vacancies):
        # look at everyone in lower layer
        # people in the front of the list have been in the layer longest
        # it makes sense to me that they would be looked at first
        # it should also fill higher layer with older people
        for i in range(len(Layers[dest-1][0])):
            if(!(i in idx)): # if the person at i hasn't been promoted
                # compute probability of being promoted
                if(Layers[dest-1][0][i][1] == 0): # if woman
                    pp = b*homophily
                else: # if man
                    pp = (1-b)*(1-homophily)
                # determine if the person should be promoted
                rand = random.uniform(0, 1) # generate random float from uniform distribution
                # Give a person a (100*pp)% chance of getting promoted
                if(pp > rand and count < vacancies): # promoted as long as there are still spots
                    idx.append(i)
                    count += 1

    idx.sort()      # sort idx
    return idx





test = random.randint(10,20)
print(test)
