import sys, os
sys.path.append(os.pardir)
sys.path.append('E:/C_Data/Desktop/22_1_summer/deep_learning')
import numpy as np
import matplotlib.pyplot as plt
from common.functions import *
from common.gradient import numerical_gradient
from dataset.mnist import load_mnist
from ch4_5__algorithm import TwoLayerNet

(x_train,t_train),(x_test,t_test)=\
    load_mnist(normalize=True,one_hot_label=True)
train_loss_list=[]

#하이퍼파라미터
iters_num=10000#반복횟수
train_size=x_train.shape[0]
batch_size=100
learning_rate=0.1
network=TwoLayerNet(input_size=784,hidden_size=50,output_size=10)
for i in range(iters_num):
    #미니배치 획득
    batch_mask=np.random.choice(train_size,batch_size) #train_size개수 중, batch_size개수만큼 array생성(100개) 
    x_batch=x_train[batch_mask] #100개의 데이터 선택
    t_batch=t_train[batch_mask] #60000훈련데이터 중 100개의 데이터
    
    #기울기 계산
    grad=network.numerical_gradient(x_batch,t_batch)
    #grad=network.gradient(x_batch,t_batch) # 성능 개선판

    #매개변수 갱신
    for key in('W1','b1','W2','b2'):
        network.params[key]-=learning_rate*grad[key] #x0=x0-eta*(round f/rount x0) p.131
                                                     #W=w-eta*(round L/round W) p.133
    #학습경과 기록
    loss=network.loss(x_batch,t_batch)
    train_loss_list.append(loss) #array에 요소 추가. (손실함수값을 추가)

