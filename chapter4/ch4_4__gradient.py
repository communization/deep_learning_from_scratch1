import numpy as np
import sys, os
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
def _numerical_gradient_no_batch(f,x): #insert (function_2,np.array([X,Y]))
                                    # => calculate each X,Y.
    h=1e-4
    grad=np.zeros_like(x) #create array similar to x
    for idx in range(x.size):
        tmp_val=x[idx]
        #calculate f(x+h)
        x[idx]=float(tmp_val)+h
        fxh1=f(x)

        #calculate f(x-h)
        x[idx]=tmp_val-h
        fxh2=f(x)

        grad[idx]=(fxh1-fxh2)/(2*h)
        x[idx]=tmp_val
    return grad
def numerical_gradient(f,X):
    if X.ndim==1:
        return _numerical_gradient_no_batch(f,X)
    else:
        grad=np.zeros_like(X)
        for idx, x in enumerate(X): #enumerate: get (i, X[i]), idx=i,x=X[i]
            grad[idx]=_numerical_gradient_no_batch(f,x)
        return grad

def function_2(x): #f2=x0^2+x1^2
    if x.ndim==1:
        return np.sum(x**2)
    else:
        return np.sum(x**2,axis=1)
def tangent_line(f,x):
    d=numerical_gradient(f,x)
    print(d)
    y=f(x)-d*x
    return lambda t: d*t+y

x0=np.arange(-2,2.5,0.25)
x1=np.arange(-2,2.5,0.25)
X,Y=np.meshgrid(x0,x1)

X=X.flatten()   #multi dimension array -> 1dimension array
Y=Y.flatten()

grad=numerical_gradient(function_2,np.array([X,Y]))

plt.figure()
plt.quiver(X,Y,-grad[0],-grad[1],angles="xy",color="#666666")
    #, headwidth=10, scale=40,color="#444444")
    # -grad[0] <= to find 0 gradient. arrow indicates direction of low gradient
plt.xlim([-2,2]), plt.ylim([-2,2])
plt.xlabel('x0'),plt.ylabel('x1')
plt.grid()
plt.legend() #범례
plt.draw()
plt.show()
'''
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x=init_x
    for i in range(step_num):
        grad=numerical_gradient(f,x)
        x-=lr*grad
    return x
'''
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append( x.copy() )

        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x, np.array(x_history)
    
def function_2(x):
    return x[0]**2+x[1]**2
init_x=np.array([-3.0,4.0])
#print(gradient_descent(function_2,init_x=init_x,lr=0.1,step_num=100))


lr = 0.1
step_num = 20
x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)

plt.plot( [-5, 5], [0,0], '--b')
plt.plot( [0,0], [-5, 5], '--b')
plt.plot(x_history[:,0], x_history[:,1], 'o')

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()