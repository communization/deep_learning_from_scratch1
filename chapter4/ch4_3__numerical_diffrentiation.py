import numpy as np
import sys, os
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

#4.3.2
def function_1(x):
    return 0.01*x**2+0.1*x
x=np.arange(0.0,20.0,0.1)
y=function_1(x)
plt.xlabel("x"), plt.ylabel("f(x)")
plt.plot(x,y)
plt.show()

def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)
print(numerical_diff(function_1,5)), print(numerical_diff(function_1,10))


