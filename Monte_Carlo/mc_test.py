import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.stats
import random

### WARNING!!! The situation in which x people retire from layer y, and there
#   are <x people in layer y-1 is not "yet" accounted for.
#   Simple fix is running to full layer promote repeatedly until all layers are full
#   Or set each layer size >= the sum of all above layer sizes

# PERSON CLASS
#   1 int 1 bool, [years to retirement (yr), Gender (g)]
#   example: Person12 = [17, man]
#
# LAYER CLASS
#   1 list of persons 1 float, [people in layer[], number of women]
#       By construction list is ordered by seniority
#           Acts a priority queue with no aditional overhead
#   example: Layer3 = [[[3, woman], [11, man], [9, man], [23, woman]], 2]
#
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

λ = 1   # homophily
b = .5  # bias

min_yr = 5  # minimum years to retirement of a new person
max_yr = 50 # maximum years to retirement of a new person

num_layers = 2      # number of layers
L = num_layers - 1  # index of top layer
Layers = []         # list of LAYERs
LayerSize = [13,8,5,3,2,1]          # preset layer size
IC = [0.4,0.3,0.2,0.1,0.05,0.01]    # initial condition
### TODO: store number of women in each layer in a seperate array, not in layer struct
###     This removes the need for a layer class as it only needs to be an array of persons
###     TODO: refactor every function, look for Layers[i][1]
#numWomen = [0,0,0,0,0,0]

### Homophily Function ###
def P(u, v):
    # sigmoid funciton λ
    return 1/(1 + 2.71828**(-λ*(u - v)))

### Fraction of women promoted to layer u from layer v ###
def f(u, v):
    return b*v*P(u,v)/(b*v*P(u,v) + (1 - b)*(1 - v)*P(1 - u,1 - v))


### create a new person ###
# inputs:   bool Gender;    0 = female, 1 = male
#           int lls;        lower limit for lifeSpan
#           int uls;        upper limit for lifeSpan
# outputs:  person struct
# effects:  none
def newPerson(gender, lls, uls):
    lifeSpan = random.randint(lls,uls)
    #level = 0
    return [lifeSpan, gender]



### choose person to be promoted ###
# inputs:   int dest;  index of higher layer
#           int vacancies;  number of vacancies
# outputs:  int[vacancies] idx;  list of indexes of people to be promoted
#           int women;  number of women promoted
#           outputs stored in a tuple
# effects:  none
def selectPromote(dest, vacancies):
    # compute likelyhood of a woman to apply
    # Layers[dest][1] holds number of women at layer dest
    homophily = P(Layers[dest][1]/LayerSize[dest], Layers[dest-1][1]/LayerSize[dest-1])

    women = 0   # number of women selected to be promoted
    count = 0   # number of people selected to be promoted
    idx = []    # indexes of people to be promoted
    # loop until enough people have been found
    while(count < vacancies):
        # look at everyone in lower layer
        # people in the front of the list have been in the layer longest
        # it makes sense to me that they would be looked at first
        # it should also fill higher layer with older people
        for i in range(len(Layers[dest-1][0])):
            if(i in idx): # if the person at i hasn't been promoted
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
                    if(Layers[dest-1][0][i][1] == 0):   # if woman
                        women += 1

    idx.sort()      # sort idx
    for i in range(len(idx)):
        idx[i] -= i # adjust indexes to accommodate for removal of earlier elements

    return [idx, women]



### move to next year ###
# inputs:   none
# outputs:  int[L] vacancies;    array holding vacancies at each layer
# effects:  removes people from each layer, updates number of women in each layer
def nextYear():
    vacancies = []
    for i in range(num_layers):     # for every layer
        for j in range(LayerSize[i]):   # for every person
            women = 0       # number of retiring women
            count = 0
            Layers[i][0][j][0] -= 1     # decrement yr of this person
            if(Layers[i][0][j][0] == 0):    # check if at retirment
                if(Layers[i][0][j][1] == 0):    # if woman decrease woman count
                    women += 1
                del Layers[i][0][j]     # remove person
                count += 1
        Layers[i][1] -= women   # update number of women in layer
        vacancies.append([])
    return vacancies



### fill vacancies ###
# inputs:   int[L] vacancies;    array holding vacancies at each layer
# outputs:  none
# effects:  adds
def fillVacancies(vacancies):
    for i in range(L, 0, -1): # for all layers L down to 1
        promoted = selectPromote(i, vacancies[i])   # get promoted people for layer i
        Layers[i][1] += promoted[1]     # add number of women promoted to upper layer
        Layers[i-1][1] -= promoted[1]   # subtract number of women promoted out of lower layer
        for j in range(vacancies[i]):   # for each person promoted
            # add person to upper layer, promoted[0][j] is index of person to be promoted
            Layers[i][0].append(Layers[i-1][0][promoted[0][j]])
            # remove person from lower layer
            del Layers[i-1][0][promoted[0][j]]
        # number of vacancies at lower level is increases do to promotions
        vacancies[i-1] += vacancies[i]


    # fill lower layer with new people
    # number of women in bottom layer is Layers[0][1], fraction of women in entry pool is always .5
    homophily = P(Layers[0][1]/LayerSize[0], .5)
    pp = homophily*b    # probability that the new person promoted is a woman
    women = 0
    for j in range(vacancies[0]):
        rand = random.uniform(0, 1)
        if(pp >= rand): # add new woman
            Layers[i][0].append(newPerson(0, min_yr, max_yr))
            women += 1
        else    # add new man
            Layers[i][0].append(newPerson(1, min_yr, max_yr))
    Layers[0][1] += women





### INITIALIZE LAYERS, make this a funciton in the future?
# decrease max_yr as you move to higher levels, TESTING ONLY
min_yr = 5
max_yr = 50

for i in range(num_layers): # for all layers
    Layers.append([[],0])   # initialize list structure
    for j in range(LayerSize[i]):   # fill current layer
        women = 0
        rand = random.uniform(0, 1) # should produce fraction of women consistent with IC
        if(IC[i] >= rand):  # add woman
            Layers[i][0].append(newPerson(0, min_yr, max_yr))
            women += 1
        else:   # add man
            Layers[i][0].append(newPerson(1, min_yr, max_yr))
    Layers[i][1] = women   # set fraction of women
    max_yr -= 10 # temporary







'''test = 0
for i in range(1000):
    if(random.uniform(0, 1) < .6):
        test += 1
print(test)'''
