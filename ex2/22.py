import numpy as np
import math
import matplotlib.pyplot as plt
import random as rd
import csv

# import data
from numpy import genfromtxt
data = genfromtxt('applesOranges.csv', delimiter=',')
x1 = data[:,0]
x2 = data[:,1]
y = data[:,2]
x1 = x1[1:len(x1)]
x2 = x2[1:len(x2)]
y = y[1:len(y)]

################################################################################
### a) plot x1 & x2 in scatter plot
col = [np.zeros(len(y)), np.zeros(len(y)), y]
#plt.scatter(x1,x2,c=col)
plt.grid()
#plt.show()

################################################################################
### b) classification performance with different weight vectors w = [w1 ,w2]

# generate weight vectors
angle = np.arange(0.0,181.0,10.0)
alpha = np.radians(angle)
w1 = np.cos(alpha)
w2 = np.sin(alpha)

# simulate neuron
n = 0
m = 0
predictions = np.zeros((len(w1),len(y)))
rightpred = np.zeros((len(w1),len(y)))

for n in range(len(w1)):
    for m in range(len(y)):
        predictions[n,m] = np.sign(np.dot([w1[n], w2[n]],[x1[m],x2[m]]))
        rightpred[n,m] = predictions[n,m] == (y[m]*2-1)

# evaluate results
i = 0
p = np.zeros(len(w1))
while i<len(w1):
     p[i] = np.sum(rightpred[i,:])/len(y)*100
     i = i+1

# plot alpha and p
plt.plot(angle,p)
plt.xlabel('alpha / degrees')
plt.ylabel('p / %')
plt.show()


################################################################################
### c) find best theta in [-3,3]

# find best angle, [w1,w2]
ind_max = np.where(p == p.max())
print 'best angle (theta = 0):', angle[ind_max]
wfix = np.transpose([[w1[ind_max]],[w2[ind_max]]])

theta = np.arange(-3,3,0.01)

# simulate neuron
n = 0
m = 0
predictions = np.zeros((len(theta),len(y)))
rightpred = np.zeros((len(theta),len(y)))
for n in range(len(theta)):
    for m in range(len(y)):
        predictions[n,m] = np.sign(np.dot(wfix,[x1[m],x2[m]]) - theta[n])
        rightpred[n,m] = predictions[n,m] == (y[m]*2-1)

# evaluate results
i = 0
p = np.zeros(len(theta))
while i<len(theta):
     p[i] = np.sum(rightpred[i,:])/len(y)*100
     i = i+1

# plot theta and p
plt.plot(theta,p)
plt.xlabel('theta')
plt.ylabel('p / %')
plt.show()

# find best angle, [w1,w2]
ind_max = np.where(p == p.max())
print 'best theta (angle = 20 degrees):', theta[ind_max]

################################################################################
### d) plot data points with color according to classification by found parameters

# calculate output for each datapoint


plt.scatter(x1,x2,c=col)
plt.grid()
#plt.show()
