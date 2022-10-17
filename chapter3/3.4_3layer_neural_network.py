import numpy as np
#3.4.2 (0->1)
X=np.array([1.0,0.5])
W1=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
B1=np.array([0.1,0.2,0.3])
print(W1.shape), print(X.shape),print(B1.shape)
A1=np.dot(X,W1)+B1
print(A1)

def sigmoid(x):
    return 1/(1+np.exp(-x))
Z1=sigmoid(A1)
print(Z1)

#3.4.2 (1->2)
W2=np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
B2=np.array([0.1,0.2])
print(Z1.shape),print(W2.shape),print(B2.shape)
A2=np.dot(Z1,W2)+B2
Z2=sigmoid(A2)

#3.4.2 (2->3)
def identity_function(x): #sigma / output layer function
    return x 
W3=np.array([[0.1,0.3],[0.2,0.4]])
B3=np.array([0.1,0.2])
A3=np.dot(Z2,W3)+B3
Y=identity_function(A3)
print(Y)

#3.4.3
def init_network():
    network={}
    network['W1']=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1']=np.array([0.1,0.2,0.3])
    network['W2']=np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['b2']=np.array([0.1,0.2])
    network['W3']=np.array([[0.1,0.3],[0.2,0.4]])
    network['b3']=np.array([0.1,0.2])
    return network
def forward(network, x):
    W1, W2, W3=network['W1'],network['W2'],network['W3']
    b1, b2, b3=network['b1'],network['b2'],network['b3']
    a1=np.dot(x,W1)+b1
    z1=sigmoid(a1)
    a2=np.dot(z1,W2)+b2
    z2=sigmoid(a2)
    a3=np.dot(z2,W3)+b3
    y=identity_function(a3)
    return y
network = init_network()
x=np.array([1.0, 1.5])
y=forward(network, x)
print(network), print(y)