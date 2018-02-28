import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.stats
import csv

with open('6_hig_50-65_65-80_late.csv', newline='') as myFile:
    reader = csv.reader(myFile)
    reader = list(reader)
    for i in range(16):
        for j in range(16):
            reader[i][j] = float(reader[i][j])
    print(reader)

    x = np.arange(.5, .66, .01)
    y = np.arange(.65, .81, .01)
    x, y = np.meshgrid(x, y)


    plt.pcolormesh(x, y, reader)
    plt.colorbar() #need a colorbar to show the intensity scale
    plt.show() #boom
