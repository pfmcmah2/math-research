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
#   example: Person12 = [17, man]
#
# LAYER
#   1 list of persons people in layer[]
#       By construction list is ordered by seniority
#           Acts a priority queue with no aditional overhead
#   example: Layer3 = [[3, woman], [11, man], [9, man], [23, woman]]
#   number of women in each layer is stored in a seperate array
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

λ = .5   # homophily
b = .9  # bias

min_yr = 1  # minimum years to retirement of a new person
max_yr = 15 # maximum years to retirement of a new person

num_layers = 2      # number of layers
L = num_layers - 1  # index of top layer
Layers = []         # list of LAYERs
LayerSize = [130,80,5,3,2,1]          # preset layer size
IC = [0.4,0.3,0.2,0.1,0.05,0.01]    # initial condition
numWomen = []       # number of women in each layer, initialized during initialize layers



### Homophily Function ###
def P(u, v):
    # sigmoid funciton λ
    return 1/(1 + 2.71828**(-λ*(u - v)))



### CREATE A NEW PERSON ###
# inputs:   bool Gender;    0 = female, 1 = male
#           int lls;        lower limit for lifeSpan
#           int uls;        upper limit for lifeSpan
# outputs:  person struct
# effects:  none
def newPerson(gender, lls, uls):
    lifeSpan = random.randint(lls,uls)
    #level = 0
    return [lifeSpan, gender]



### CHOOSE PERSONS TO BE PROMOTED ###
# inputs:   int dest;  index of higher layer
#           int vacancies;  number of vacancies
# outputs:  int[vacancies] idx;  list of indexes of people to be promoted
#           int women;  number of women promoted
#           outputs stored in a tuple
# effects:  none
def selectPromote(dest, vacancies):
    # compute likelyhood of a woman to apply
    # numWomen[dest] holds number of women at layer dest
    homophily = P(numWomen[dest]/LayerSize[dest], numWomen[dest-1]/LayerSize[dest-1])

    women = 0   # number of women selected to be promoted
    count = 0   # number of people selected to be promoted
    idx = []    # indexes of people to be promoted
    # loop until enough people have been found
    while(count < vacancies):
        # look at everyone in lower layer
        # people in the front of the list have been in the layer longest
        # it makes sense to me that they would be looked at first
        # it should also fill higher layer with older people
        for i in range(len(Layers[dest-1])):    # look at all people in lower layer, may not equal layer size
            if(i not in idx): # if the person at i hasn't been promoted
                # compute probability of being promoted
                if(Layers[dest-1][i][1] == 0): # if woman
                    pp = b*homophily
                else: # if man
                    pp = (1-b)*(1-homophily)
                # determine if the person should be promoted
                rand = random.uniform(0, 1) # generate random float from uniform distribution
                # Give a person a (100*pp)% chance of getting promoted
                if(pp > rand and count < vacancies): # promoted as long as there are still spots
                    idx.append(i)
                    count += 1
                    if(Layers[dest-1][i][1] == 0):   # if woman
                        women += 1

    idx.sort()      # sort idx
    for i in range(len(idx)):
        idx[i] -= i # adjust indexes to accommodate for removal of earlier elements

    return [idx, women]



### MOVE TO NEXT YEAR ###
# inputs:
# outputs:  int[num_layers] vacancies;  array holding vacancies at each layer
# effects:  removes people from each layer, updates number of women in each layer
def nextYear():
    vacancies = []
    for i in range(num_layers):     # for every layer
        women = 0       # number of retiring women
        count = 0
        retiring = []   # list of indexes of people retiring
        for j in range(len(Layers[i])):   # for every person
            Layers[i][j][0] -= 1     # decrement yr of this person
            if(Layers[i][j][0] == 0):    # check if at retirment
                if(Layers[i][j][1] == 0):    # if woman decrease woman count
                    women -= 1
                retiring.append(j)     # add person to retirment list
                count += 1
        numWomen[i] += women   # update number of women in layer
        for j in range(len(retiring)):  # remove retired persons
            del Layers[i][retiring[j]-j]  # retiring[j]-j holds index of person to be removed
        vacancies.append(count)
    return vacancies



### FILL VACANCIES ###
# inputs:   int[L] vacancies;    array holding vacancies at each layer
# outputs:  none
# effects:  adds/removes persons to simulate promotion
#           updates number of women in both layers involved for each promotion
def fillVacancies(vacancies):
    global Layers
    global numWomen
    for i in range(L, 0, -1): # for all layers L down to 1
        promoted = selectPromote(i, vacancies[i])   # get promoted people for layer i
        numWomen[i] += promoted[1]     # add number of women promoted to upper layer
        numWomen[i-1] -= promoted[1]   # subtract number of women promoted out of lower layer
        for j in range(vacancies[i]):   # for each person promoted
            # add person to upper layer, promoted[0][j] is index of person to be promoted
            Layers[i].append(Layers[i-1][promoted[0][j]])
            # remove person from lower layer
            del Layers[i-1][promoted[0][j]]
        # number of vacancies at lower level is increases do to promotions
        vacancies[i-1] += vacancies[i]


    # fill lower layer with new people
    # number of women in bottom layer is Layers[0][1], fraction of women in entry pool is always .5
    homophily = P(numWomen[0]/LayerSize[0], .5)
    pp = homophily*b    # probability that the new person promoted is a woman
    women = 0   # number of new women
    for j in range(vacancies[0]):
        rand = random.uniform(0, 1)
        if(pp >= rand): # add new woman
            Layers[0].append(newPerson(0, min_yr, max_yr))
            women += 1
        else:   # add new man
            Layers[0].append(newPerson(1, min_yr, max_yr))
    numWomen[0] += women



### INITIALIZE LAYERS ###
# inputs:   int min_yr, max_yr;     min and max possible years remaining
# outputs:  none
# effects:  clears/initializes Layers[]
#           clears/initializes numWomen
# NOTE: decrease max_yr as you move to higher levels, TESTING ONLY
#       need to find a good way to get different yr for each layer
def initializeLayers(min_yr, max_yr):
    global Layers
    global numWomen
    Layers = [] # clear Layers
    numWomen = []
    for i in range(num_layers): # for all layers
        Layers.append([])   # initialize list structure
        women = 0   # keeps count of number of women
        for j in range(LayerSize[i]):   # fill current layer
            rand = random.uniform(0, 1) # should produce fraction of women consistent with IC
            if(IC[i] >= rand):  # add woman
                Layers[i].append(newPerson(0, min_yr, max_yr))
                women += 1
            else:   # add man
                Layers[i].append(newPerson(1, min_yr, max_yr))
        numWomen.append(women)   # set fraction of women
        max_yr -= 10 # temporary



### COMPUTE FRACTION OF WOMEN
# inputs:   none
# outputs:  float[num_layers] frac; holds fraction of women at each layer
# effects:  none
def computeFraction():
    frac = []
    for i in range(num_layers):
        frac.append(numWomen[i]/LayerSize[i])
    return frac



### PRINT LAYERS ###
# inputs:   none
# outputs:  none
# effects:  none
# prints layer in top down order
def printLayers():
    for i in range(num_layers):
        print('Layer', L-i, ':', Layers[L-i])




initializeLayers(min_yr, max_yr)
#printLayers()
print(numWomen)
print(computeFraction())
for i in range(500):
    #print(i)
    vac = nextYear()
    #print(vac)
    fillVacancies(vac)

#printLayers()
print(numWomen)
print(computeFraction())
