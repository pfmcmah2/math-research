import numpy as np
import matplotlib.pyplot as plt

file_string = 'foo.png'
x = np.arange(0, 1, 0.1);
for i in range(0, 100):
    y = np.sin(i*x)
    plt.plot(x, y)
    out_string = file_string[:3] + str(i) + file_string[3:]
    plt.savefig(out_string)
    plt.clf()
    plt.cla()
    plt.close()

x = 1.

def dx(x):
    return 2*x

for i in range(0, 100):
    #print(.01*dx(x))
    d = .01*dx(np.sqrt(x))
    x += d

print(x)
