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
plt.scatter(x1[y==0],x2[y==0],c='r',label='$y=0$')
plt.scatter(x1[y==1],x2[y==1],c='g',label='$y=1$')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('a) input data')
plt.legend()
plt.grid()
plt.show()

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
plt.title('b) predictions $p$ with variable weights $w$')
plt.grid()
plt.show()


################################################################################
### c) find best theta in [-3,3]

# find best angle, [w1,w2]
ind_max = np.where(p == p.max())
print 'best angle (theta = 0):', angle[ind_max]
wfix = np.transpose([[w1[ind_max]],[w2[ind_max]]])

theta = np.arange(-3,3,0.1)

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
plt.title('c) predictions $p$ with variable input offset $\Theta$')
plt.grid()
plt.show()

# find best angle, [w1,w2]
ind_max = np.where(p == p.max())
print 'best theta (angle = 20 degrees):', theta[ind_max]
print 'best prediction result:', p[ind_max]

################################################################################
### d) plot data points with color according to classification by found parameters

# extract output for each datapoint
rightpred_best = rightpred[ind_max,:]
Rgb = np.zeros(len(x1))
rGb = np.zeros(len(x2))
rgB = 0.5*rightpred_best

for i in range(len(x1)):
    if rightpred_best[0,0,i] == 1:
        plt.scatter(x1[i],x2[i],c='g')
    else:
        plt.scatter(x1[i],x2[i],c='r')
plt.scatter(x1[rightpred_best[0,0,i]==1],x2[rightpred_best[0,0,i]==1],c='g',label='right predicted')
plt.scatter(x1[rightpred_best[0,0,i]==0],x2[rightpred_best[0,0,i]==0],c='r',label='false predicted')

plt.scatter(wfix[0,0,0],wfix[0,0,1],s=100,c='k',label='w')

plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.title('d) input data and trueness of their predictions')
plt.grid()
plt.show()

################################################################################
### e) looking for best combination of alpha and theta

# simulate neuron
predictions = np.zeros((len(w1),len(y),len(theta)))
rightpred = np.zeros((len(w1),len(y),len(theta)))
for n in range(len(w1)):
    for m in range(len(y)):
        for k in range(len(theta)):
            predictions[n,m,k] = np.sign(np.dot([w1[n], w2[n]],[x1[m],x2[m]])-theta[k])
            #predictions[n,m,k] = np.sign(np.dot([np.cos(np.radians(20)), np.sin(np.radians(20))],[x1[m],x2[m]])-0.2)
            rightpred[n,m,k] = predictions[n,m,k] == (y[m]*2-1)

# evaluate results
i = 0
j = 0
p = np.zeros((len(w1),len(theta)))
while i<len(w1):
    while j<len(theta):
        p[i,j] = np.sum(rightpred[i,:,j])/len(y)*100
        print p[i,j]
        j = j+1
    i = i+1

ind_max = np.where(p == p.max())
print p.max()
