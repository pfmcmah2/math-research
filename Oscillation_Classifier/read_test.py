import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.stats
import csv

name_string = '2_med_50-1__50-1__full.csv'

res = name_string[2:5]
if(res == 'low'):
    res = .1
if(res =='med'):
    res = .05
if(res =='hig'):
    res = .01

print(res)

x_low  = int(name_string[6:8])
if(name_string[10] == '_'):
    x_high = 100
else:
    x_high = int(name_string[9:11])
y_low  = int(name_string[12:14])
if(name_string[16] == '_'):
    y_high = 100
else:
    y_high = int(name_string[15:17])

x_low  = float(x_low/100)
x_high  = float(x_high/100)
x_high += res
y_low  = float(y_low/100)
y_high  = float(y_high/100)
y_high += res

print(x_low)
print(x_high)
print(y_low)
print(y_high)

size = int((x_high - x_low)/res)
print(size)



with open(name_string, newline='') as myFile:
    reader = csv.reader(myFile)
    reader = list(reader)
    for i in range(size):
        for j in range(size):
            reader[i][j] = float(reader[i][j])
    print(reader)

    x = np.arange(x_low, x_high, res)
    y = np.arange(y_low, y_high, res)
    x, y = np.meshgrid(x, y)


    plt.pcolormesh(x, y, reader)
    plt.xlabel("Bias")
    plt.ylabel("Homophily")
    plt.colorbar() #need a colorbar to show the intensity scale
    plt.show() #boom
