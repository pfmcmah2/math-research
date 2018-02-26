import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.stats
import csv

name_string = '6_050_50-1*_50-1*_late.csv'
directory = 'avg/'

res = int(name_string[2:5])
res = float(res/1000)

print(res)

x_low  = int(name_string[6:8])
if(name_string[10] == '*'):
    x_high = 100
else:
    x_high = int(name_string[9:11])
y_low  = int(name_string[12:14])
if(name_string[16] == '*'):
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

x_size = int((x_high - x_low)/res)
y_size = int((y_high - y_low)/res)



with open(directory + name_string, newline='') as myFile:
    reader = csv.reader(myFile)
    reader = list(reader)
    for i in range(x_size):
        for j in range(y_size):
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
