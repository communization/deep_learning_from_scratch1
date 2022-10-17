import numpy as np
import sys, os
import matplotlib.pylab as plt
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

# 4.2.1
def sum_squares_error(y,t):
    return 0.5*np.sum((y-t)**2)

y1=[0.1, 0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
t=[0,0,1,0,0,0,0,0,0,0]
print(sum_squares_error(np.array(y1),np.array(t)))
y2=[0.1, 0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0]
print(sum_squares_error(np.array(y2),np.array(t)))

# 4.2.2
def cross_entropy_error(y,t):
    delta=1e-7
    return -np.sum(t*np.log(y+delta))
print(cross_entropy_error(np.array(y1),np.array(t)))
print(cross_entropy_error(np.array(y2),np.array(t)))

#4.2.3
(x_train,t_train),(x_test,t_test)=\
    load_mnist(normalize=True,one_hot_label=True) #normalize:0~255(False)=>0~1 (when True) 
                                              #flatten : 1x28x28(False)->1demension (when True)
                                        #one_hot_label : number(False), label_one hot incoding(only answer 1) (True)
print(x_train.shape) , print(t_train.shape)
print(x_test.shape), print(t_test.shape)

train_size = x_train.shape[0]
batch_size=10
batch_mask=np.random.choice(train_size, batch_size) # Among 0~60000, get random 10 number  if(60000,10)
x_batch=x_train[batch_mask]
t_batch=t_train[batch_mask]
print(np.random.choice(60000,10)) #test random.choice( , )

#4.2.4 
def cross_entropy_error1(y,t):
    if y.ndim==1:
        t=t.reshape(1,t.size)
        y=y.reshape(1,y.size)
        print(y.shape)
    batch_size=y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size),t]+1e-7))/batch_size
    #np.log(y[np.arange(batch_size),t]) : 
    #  1. np.arange(batch_size) => create 0~ batch_size - 1  array
    #  2. if [0,1,2,3,4] 's answer [2,7,0,9,4] => y[0,2],y[1,7],y[2,0],y[3,9],y[4,4] => match each answer and calculate that only

