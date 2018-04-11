import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.stats

T = np.arange(0, 10, 1)
for i in range(5):
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(str(i))
    plt.plot(T, T**i)
    plt.legend()
    plt.show()
