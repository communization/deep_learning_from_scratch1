import sys, os
sys.path.append(os.pardir)
sys.path.append('E:/C_Data/Desktop/22_1_summer/deep_learning')
import numpy as np
from dataset.mnist import load_mnist
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict

class TwoLayerNet:
    def __init__(self, input_size,hidden_size,output_size,weight_init_std=0.01):
        #가중치 초기화
        self.params={}
        self.params['W1']=weight_init_std*\
            np.random.randn(input_size,hidden_size) #784x50
        self.params['b1']=np.zeros(hidden_size)
        self.params["W2"]=weight_init_std*\
            np.random.randn(hidden_size,output_size) #50x10
        self.params['b2']=np.zeros(output_size)

        #계층 생성
        self.layers=OrderedDict() #순서가 있는 딕서녀리에 추가한 순서를 기억.->역전파에 이용
        self.layers['Affine1']=\
            Affine(self.params['W1'], self.params['b1']) #affine층의 계산.(dot계산) (순전파, 역전파를 포함)
        self.layers['Relu1']=Relu() #Relu함수(0, x)
        self.layers['Affine2']=\
            Affine(self.params['W2'], self.params['b2'])
        self.lastLayer=SoftmaxWithLoss()#로그함수를 이용한 정답레이블과 시험레이블 오차 계산
    
    def predict(self, x):
        for layer in self.layers.values():
            x=layer.forward(x) #순전파 진행.
        return x
    
    #x : 입력데이터, t : 정답레이블
    def loss(self, x, t):
        y=self.predict(x)
        return self.lastLayer.forward(y,t) #self. softmaxwithloss 클래스 (순전파+오차계산(출력층함수 소프트맥스))
    def accuracy(self,x,t):
        y=self.predict(x)
        y=np.argmax(y, axis=1)
        if t.ndim != 1:t=np.argmax(t,axis=1) # :은 {}와 같은 역할.
        accuracy=np.sum(y==t)/float(x.shape[0]) #정확도는 정답레이블과 y값이 같을때를 기준으로 파악.
        return accuracy
    
    def numerical_gradient(self,x,t): #기존 gradient / {f(x+h)-f(x-h)}/2h 방식으로 구현. 오래걸림.
        loss_W=lambda W: self.loss(x,t)
        grads={}
        grads['W1']=numerical_gradient(loss_W,self.params['W1'])
        grads['b1']=numerical_gradient(loss_W,self.params['b1'])
        grads['W2']=numerical_gradient(loss_W,self.params['W2'])
        grads['b2']=numerical_gradient(loss_W,self.params['b2'])
        return grads

    def gradient(self,x,t): 
        #순전파
        self.loss(x,t) #추론. 0층->2층진행후 출력층 소프트맥스 함수 적용.

        #역전파
        dout=1
        dout=self.lastLayer.backward(dout)

        layers=list(self.layers.values())
        layers.reverse() #순서 역으로 바꾸기.
        for layer in layers:
            dout=layer.backward(dout)
        
        #결과 저장
        grads={}
        grads['W1']=self.layers['Affine1'].dW
        grads['b1']=self.layers['Affine1'].db
        grads['W2']=self.layers['Affine2'].dW
        grads['b2']=self.layers['Affine2'].db
        return grads

#데이터 읽기
(x_train,t_train),(x_test,t_test)=\
    load_mnist(normalize=True,one_hot_label=True)
print(x_train.shape) #60000x784
network = TwoLayerNet(input_size=784,hidden_size=50,output_size=10) #위 클래스 할당.

x_batch=x_train[:3] #[0,0~784],[1,0~784],[2,0~784] 
t_batch=t_train[:3] #처음부터 3번까지 (3개의 데이터 선택.)

grad_numerical=network.numerical_gradient(x_batch,t_batch)
grad_backprop=network.gradient(x_batch,t_batch)

#각 가중치의 절대값을 구한 후, 그 절댓값들의 평균을 낸다.
#오차역전파법이 잘 작동하는지 확인(수치미분과 비교)
for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key]-grad_numerical[key]))
    print(key+":"+str(diff))
    