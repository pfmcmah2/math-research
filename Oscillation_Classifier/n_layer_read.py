import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.stats
import csv

with open('6_low_0-1_late.csv', newline='') as myFile:
    reader = csv.reader(myFile)
    reader = list(reader)
    for i in range(6):
        for j in range(6):
            reader[i][j] = float(reader[i][j])
    print(reader)

    x = np.arange(.5, 1.1, .1)
    y = np.arange(.5, 1.1, .1)
    x, y = np.meshgrid(x, y)


    plt.pcolormesh(x, y, reader)
    plt.colorbar() #need a colorbar to show the intensity scale
    plt.show() #boom
