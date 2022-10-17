import sys, os
sys.path.append(os.pardir)
sys.path.append('E:/C_Data/Desktop/22_1_summer/deep_learning')
import numpy as np
from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet:
    def __init__(self, input_size, hidden_size,output_size,weight_init_std=0.01):
        #가중치 초기화 #입력층의 뉴런수, 은닉층의 뉴런수, 출력층의 뉴런 수
        self.params={}
        self.params['W1']=weight_init_std*\
            np.random.randn(input_size,hidden_size)
        self.params['b1']=np.zeros(hidden_size)
        self.params['W2']=weight_init_std *\
            np.random.randn(hidden_size, output_size)
        self.params['b2']=np.zeros(output_size)
    def predict(self, x): #추론 시작(가중치로 데이터 예측), x는 이미지 데이터
        W1, W2= self.params['W1'], self.params['W2']
        b1,b2= self.params['b1'],self.params['b2']
        a1=np.dot(x,W1)+b1
        z1=sigmoid(a1)
        a2=np.dot(z1,W2)+b2
        y=softmax(a2)
        return y
    def loss(self, x, t): #x :입력데이터 t:정답레이블
        y=self.predict(x)
        return cross_entropy_error(y,t) #손실함수값
    def accuracy(self, x, t):
        y=self.predict(x)
        y=np.argmax(y, axis=1) #최대값 인덱스
        t=np.argmax(t,axis=1)
        accuracy=np.sum(y==t)/float(x.shape[0])  #입력데이터 개수
        return accuracy
    def numerical_gradient(self,x,t):
        loss_W=lambda W: self.loss(x,t)
        
        grads={}
        grads['W1']=numerical_gradient(loss_W, self.params['W1'])
        grads['b1']=numerical_gradient(loss_W, self.params['b1'])
        grads['W2']=numerical_gradient(loss_W, self.params['W2'])
        grads['b2']=numerical_gradient(loss_W, self.params['b2'])
        return grads

net=TwoLayerNet(input_size=784,hidden_size=100,output_size=10)
print(net.params['W1'].shape),print(net.params['b1'].shape)
print(net.params['W2'].shape),print(net.params['b2'].shape)

x=np.random.rand(100,784) #더미 입력 데이터(100장분량)
t=np.random.rand(100,10)  #더미 정답 레이블(100장분량)

#grads=net.numerical_gradient(x,t) #시간이 오래걸려서 gradient(self,x,t) (오차역전파법)이용
#print(grads['W1'].shape), print(grads['b1'].shape)
#print(grads['W2'].shape), print(grads['b2'].shape)
