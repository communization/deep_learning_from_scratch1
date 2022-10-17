import sys, os
sys.path.append(os.pardir)
sys.path.append('E:/C_Data/Desktop/22_1_summer/deep_learning')
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W=np.random.randn(2,3)

    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z=self.predict(x)
        y=softmax(z)
        loss=cross_entropy_error(y,t)
        return loss
net=simpleNet()
print(net.W)  #__init__
x=np.array([0.6, 0.9])
p=net.predict(x)
print(p)
np.argmax(p) #최대값 인덱스.
t=np.array([0,0,1]) #정답 레이블 0,1,2중 (2)가 정답.
print(net.loss(x,t)) #손실함수 구하기

#def f(W):  # 함수의 인수 W는 더미로 만든것. 
    #return net.loss(x,t) #손실함수의 최소를 구하기 위해 f=손실함수로 설정.
f=lambda w:net.loss(x,t) #위와 같은 표현
dW=numerical_gradient(f,net.W)
print(dW)