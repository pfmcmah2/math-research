import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.stats
import csv

name_string = 'Film_Industry.csv'
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


T = np.arange(0, years, 1)
plt.xlabel("Years")
plt.ylabel("Fraction of Women")
plt.title("MATH 492")
for i in range(num_layers):
    plt.plot(T, data[i],label = layer_names[i])
plt.legend()
plt.show()
