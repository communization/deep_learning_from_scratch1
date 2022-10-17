import numpy as np
#3.5.1
a=np.array([0.3, 2.9, 4.0])
exp_a=np.exp(a)
sum_exp_a=np.sum(exp_a)
y= exp_a/sum_exp_a
print(exp_a), print(sum_exp_a), print(y)

def softmax1(a):
    exp_a=np.exp(a)
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a
    return y

#3.5.2 error of overflow (exp)
a=np.array([1010,1000,990])
np.exp(a)/np.sum(np.exp(a))
c=np.max(a)
print(a), print(np.exp(a)/np.sum(np.exp(a))) #overflow
print(a-c),print(np.exp(a-c)/np.sum(np.exp(a-c)))

def softmax(a):
    c=np.max(a)
    exp_a=np.exp(a-c)
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a
    return y

