import sys,os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import numpy as np
from PIL import Image
import pickle
#3.6.1

(x_train,t_train),(x_test,t_test)=\
    load_mnist(flatten=True, normalize=False) #normalize:0~255(False)=>0~1 (when True) 
                                              #flatten : 1x28x28(False)->1demension (when True)
                                        #one_hot_label : number(False), label_one hot incoding(True)
print(x_train.shape) , print(t_train.shape)
print(x_test.shape), print(t_test.shape)

def img_show(img):
    pil_img=Image.fromarray(np.uint8(img)) #change into PIL data
    pil_img.show()      
#img=x_train[0]
#label=t_train[0]
img=x_test[0]
label=t_test[0]
print(label), print(img.shape)
img=img.reshape(28,28) #784 -> 28x28
print(img.shape)
img_show(img)

#3.6.2
def get_data():
    (x_train,t_train),(x_test,t_test)=\
        load_mnist(normalize=True, flatten=True,one_hot_label=False)
    return x_test, t_test
def init_network():
    with open("sample_weight.pkl",'rb') as f:
        network=pickle.load(f)
    return network

def sigmoid(x):
    return 1/(1+np.exp(-x))
def softmax(a):
    c=np.max(a)
    exp_a=np.exp(a-c)
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a
    return y

def predict(network, x):
    W1,W2,W3=network['W1'],network['W2'],network['W3']
    b1,b2,b3=network['b1'],network['b2'],network['b3']
    a1=np.dot(x,W1)+b1
    z1=sigmoid(a1)
    a2=np.dot(z1,W2)+b2
    z2=sigmoid(a2)
    a3=np.dot(z2,W3)+b3
    y=softmax(a3)
    return y

x,t=get_data()
print(t[0]) 
network=init_network()
print(network['W1'].shape),print(network['W2'].shape),print(network['W3'].shape)
        #784->50->100->10 (layer)
print("x.shape: "+str(x.shape)) #10000data (784 size)

accuracy_cnt=0
for i in range(len(x)):
    y=predict(network,x[i])
    p=np.argmax(y) #choose the highest probability
    if p==t[i]:   #if highest probability == real number(handwrite_number)
        accuracy_cnt+=1
print("Accuracy: "+str(float(accuracy_cnt)/len(x)))