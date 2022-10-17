import numpy as np
import sys,os
sys.path.append('E:/C_Data/Desktop/22_1_summer/deep_learning')
from common.functions import cross_entropy_error, softmax
X_dot_W=np.array([[0,0,0],[10,10,10]])
B=np.array([1,2,3])
print(X_dot_W)
print(X_dot_W+B)

class SoftmaxWithLoss:
    def __init__(self):
        self.loss=None #손실
        self.y=None #softmax의 출력
        self.t=None #정답레이블(원-핫 벡터)

    def forward(self,x,t):
        self.t=t
        self.y=softmax(x)
        self.loss=cross_entropy_error(self.y,self.t)
        return self.loss
    def backward(self, dout=1):
        batch_size=self.t.shape[0]
        dx=(self.y-self.t)/batch_size
        return dx