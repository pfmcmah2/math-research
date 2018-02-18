import matplotlib.pyplot as plt
import numpy as np
import csv

'''
myData = []
for i in range(5):
    myData.append([])
    for j in range(5):
        myData[i].append(j + 5*i)

myFile = open('csvexample3.csv', 'w')
with myFile:
   writer = csv.writer(myFile)
   writer.writerows(myData)
'''

with open('csvexample3.csv', newline='') as myFile:
    reader = csv.reader(myFile)
    reader = list(reader)
    for i in range(5):
        for j in range(5):
            reader[i][j] = float(reader[i][j])
    print(reader)

    x = np.arange(0, 5, 1)
    y = np.arange(0, 5, 1)
    x, y = np.meshgrid(x, y)


    plt.pcolormesh(x, y, reader)
    plt.colorbar() #need a colorbar to show the intensity scale
    plt.show() #boom
