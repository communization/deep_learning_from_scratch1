import numpy as np
import matplotlib.pylab as plt
#3.2.2
def step_function1(x):
    '''if x>0:
        return 1
    else:
        return 0
    '''
    y=x>0
    return y.astype(np.int)  #astype: change data type
x=np.array([-1.0, 1.0, 2.0])
y=step_function1(x)
print(y)

#3.2.3
def step_function(x):
    return np.array(x>0, dtype=np.int)
x=np.arange(-5.0,5.0,0.1)
y=step_function(x)
'''
plt.plot(x,y)
plt.ylim(-0.1, 1.1)
plt.show()
'''
#3.2.4
def sigmoid(x):
    return 1/(1+np.exp(-x))

x=np.array([-1.0, 1.0, 2.0])
print(sigmoid(x))

t=np.array([1.0,2.0,3.0])
print(1.0+t), print(1.0/t)

x=np.arange(-5.0,5.0,0.1)
y=sigmoid(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()

#3.2.7 ReLU
def relu(x):
    return np.maximum(0,x)