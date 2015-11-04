import numpy as np
import math
import matplotlib.pyplot as plt
import random as rd
import csv

# training data
xn_list = [0.5503,0.9206,0.5359,0.6081,0.0202,0.8545,0.2357,0.4847,0.3996,0.1957]
tn_list = [-0.5894,-0.2507,-0.0468,-0.3402,0.2857,-1.0683,0.8605,-0.0801,0.6837,1.1850]
eta = 0.5

# initialization of one hidden layer with 3 neurons and one output (weight w, bias b)
w_xhl = np.subtract([rd.random(),rd.random(),rd.random()],0.5)
w_hly = np.subtract([rd.random(),rd.random(),rd.random()],0.5)
b_hl = np.subtract([rd.random(),rd.random(),rd.random()],0.5)
b_y = rd.random()-0.5


################################################################################

# forward propagation functions
def fw_prop_in(x):
    S0 = x

def fw_prop_hl(x,w,b):                  #hl refers to hidden layer
    S = np.tanh(np.subtract(w*x,b))
    return S

def fw_prop_out(S,w,b):
    y = np.tanh(np.sum(np.subtract(np.multiply(w,S),b)))
    return y

# derivation of tanh
def sech(x):
    y = np.divide(4*np.square(np.cosh(x)),np.square(np.cosh(2*x)+1))
    return y

# quadratic error
def quad_error(tn,y):
    q_error = 0.5*np.square(tn-y)
    return q_error

# backpropagation
def local_error_out(S,w_hly,b_y):
    l_error_out = sech(np.sum(np.multiply(S,w_hly))-b_y)
    return l_error_out

def local_error_hl(w,S,d_child,w_child):
    l_error_hl = sech(w*S)*d_child*w_child
    return l_error_hl

def local_error_x(x,w_child,d_child):
    l_error_in = sech(x)*(d_child[0]*w_child[0] + d_child[1]*w_child[1] + d_child[2]*w_child[2])
    return l_error_in

# weight updates
def weight_update_x(eta,l_error,w,x):
    Dw = eta*l_error*w*x
    w = w - Dw
    return w

def weight_update_hl(eta,l_error,w,S):
    Dw = eta*l_error*S
    w = w - Dw
    return w

################################################################################
##### ITERATIONS [a),b)]

n_iterations = 0
n_training = 0

E_n = [0]
S_n1 = [0]
S_n2 = [0]
S_n3 = [0]

d_hl = [0,0,0]

while n_iterations <3000:
    # 0: select training data
    xn, tn = xn_list[n_training], tn_list[n_training]
    n_training = (n_training+1) % 10
    # 1: forward propagation
    S0 = xn
    S = fw_prop_hl(S0,w_xhl,b_hl)
    y = fw_prop_out(S,w_hly,b_y)

    # 2: quadratic error
    E = quad_error(tn,y)
    E_n.extend([E])
    S_n1.extend([S[0]])
    S_n2.extend([S[1]])
    S_n3.extend([S[2]])

    # 3: local errors
    dy = local_error_out(S,w_hly,b_y)
    d_hl[0] = local_error_hl(w_xhl[0],S[0],dy,w_hly[0])
    d_hl[1] = local_error_hl(w_xhl[1],S[1],dy,w_hly[1])
    d_hl[2] = local_error_hl(w_xhl[2],S[2],dy,w_hly[2])
    dx = local_error_x(S0,w_xhl,d_hl)

    # 4: update weights
    w_xhl[0] = weight_update_x(eta,dx,w_xhl[0],S0)
    w_xhl[1] = weight_update_x(eta,dx,w_xhl[1],S0)
    w_xhl[2] = weight_update_x(eta,dx,w_xhl[2],S0)

    w_hly[0] = weight_update_hl(eta,d_hl[0],w_hly[0],S[0])
    w_hly[1] = weight_update_hl(eta,d_hl[1],w_hly[1],S[1])
    w_hly[2] = weight_update_hl(eta,d_hl[2],w_hly[2],S[2])

    n_iterations = n_iterations+1



# plot error over iteration numbers
iterations = np.linspace(1.0,3000.0,len(E_n))

plt.plot(iterations,np.sqrt(E_n))
plt.xlabel('Iterations')
plt.ylabel('Quadratic Error')
plt.grid()
plt.show()

# plot final results for activations of hidden units
S_n1 = S_n1[-10:]
S_n2 = S_n2[-10:]
S_n3 = S_n3[-10:]

plt.scatter(xn_list,S_n1)
plt.scatter(xn_list,S_n2,c='r')
plt.scatter(xn_list,S_n3,c='k')
plt.xlabel('Input')
plt.ylabel('Activations S')
plt.grid()
plt.show()

################################################################################
##### Input-Output Function over Input Space [c)]

n_iterations = 0
Input = np.linspace(0.0,1.0,100)
Output = np.zeros(len(Input))
while n_iterations < len(Input):
    # forward propagation
    S0 = Input[n_iterations]
    S = fw_prop_hl(S0,w_xhl,b_hl)
    Output[n_iterations] = fw_prop_out(S,w_hly,b_y)
    n_iterations = n_iterations+1


n_iterations = 0
Output_Train = np.zeros(10)
while n_iterations < 10:
    # forward propagation
    S0 = xn_list[n_iterations]
    S = fw_prop_hl(S0,w_xhl,b_hl)
    Output_Train[n_iterations] = fw_prop_out(S,w_hly,b_y)

    n_iterations = n_iterations+1

plt.plot(Input,Output)
plt.scatter(xn_list,Output_Train,c='b')
plt.scatter(xn_list,tn_list,c='r')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend(['input-output function', 'output of training data', 'original training outputs'])
plt.grid()
plt.show()

################################################################################
##### d)
# 1. the derivation of the quadratic error is simple
# Bishop - 'Pattern Recognition and Machine Learning' 3.1.4:
# 2. the quadratic error function encourages weight values to remain small
# 3. quadratic dependency of w ensures (analytic) closed form of minimization problem (y contains the weights)
