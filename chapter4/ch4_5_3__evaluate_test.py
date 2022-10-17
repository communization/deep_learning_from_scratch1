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

network=TwoLayerNet(input_size=784,hidden_size=50,output_size=10)
#하이퍼 파라미터
iters_num=10000 #반복횟수를 적절히 설정
train_size=x_train.shape[0] #60000
print(train_size)
batch_size=100 #미니배치 크기(표본)
learning_rate=0.1 #eta

train_loss_list=[]
train_acc_list=[]
test_acc_list=[]

#1에폭당 반복 수
iter_per_epoch=max(train_size/batch_size,1) #60000/100=600
print (iter_per_epoch)

for i in range(iters_num):
    #미니배치 획득
    batch_mask=np.random.choice(train_size,batch_size)
    x_batch=x_train[batch_mask] #100x784
    t_batch=t_train[batch_mask] #100x10
    print(x_batch.shape),print(t_batch.shape)
    #기울기 계산
    grad=network.numerical_gradient(x_batch,t_batch)
    #grad=network.gradient(x_batch,t_batch)

    #매개변수 갱신
    for key in ('W1','b1','W2','b2'):
        network.params[key]-=learning_rate*grad[key]
    
    #학습경과 기록
    loss=network.loss(x_batch,t_batch)
    train_loss_list.append(loss)

    #1에폭당 정확도 계산
    if i % iter_per_epoch==0:
        train_acc=network.accuracy(x_train,t_train)
        test_acc=network.accuracy(x_test,t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc |"+str(train_acc)+", "+str(test_acc))

# 그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()