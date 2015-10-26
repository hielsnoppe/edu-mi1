import numpy as np
import math
import matplotlib.pyplot as plt
import random as rd

# define fix values
nhid = 10
x = np.arange(-2.0,2.0,0.05)
y = np.zeros((50,len(x)))
a_max = 0.5

# loop for 50 MLP output calculations
i = 0
while (i < 50):
    n = 0
    yi = np.zeros(len(x))
    while (n < nhid):
        w = rd.random()
        ai = rd.random()*a_max
        bi = rd.random()*4-2
        yi = yi + w * np.tanh(ai*x-bi)
        n = n+1

    y[i,:] = yi
    i = i+1

# loop for 50 plots
i = 0
while (i<50):
    plt.plot(x,y[i,:])
    i = i+1

plt.xlabel('x')
plt.ylabel('y')
plt.title('Input-Output Functions of 50 MLPs')
plt.show()
