#초기값을 0으로 한다면 학습이 올바르게 이루어지지 않는다.
#이유는 오차역전파법에서 모든 가중치의 값이 똑같이 갱신되기 때문.
#예를 들어 2층 신경망에서 첫번째와 두번째 층의 가중치가 0이라고 가정한다면,
#순전파 때의 입력층의 가중치가 0이고, 두번째 층의 뉴런에 모두 같은 값 전달.
#두 번째 층의 모든 뉴런에 같은 값이 입력된다는 것은 역전파때 두번째 층의 가중치가 똑같이 갱신된다는 말.
#가중치들은 같은 초기값에서 시작하고 갱신을 거쳐도 여전히 같은 값을 유지하게 된다.
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))
def ReLU(x):
    return np.maximum(0,x)
def tanh(x):
    return np.tanh(x)

input_data=np.random.randn(1000,100)#1000개의 데이터
node_num=100 #각 은닉층의 노드(뉴런) 수
hidden_layer_size=5 #은닉층 5개
activations={} #활성화 결과를 저장.

x=input_data
for i in range(hidden_layer_size):
    if i!=0:
        x=activations[i-1]

    #초깃값을 다양하게 바꿔가며 실험.
    #w=np.random.randn(node_num,node_num)*1
    #w=np.random.randn(node_num,node_num)*0.01
    w=np.random.randn(node_num,node_num)*np.sqrt(1.0/node_num) #Xavier초깃값. 표준편차가 1/sqrtn이 되도록
            #n은 앞층의 노드수, 층이 깊어지면서 다소 일그러지지만, 넓게 분포.
            #이는 sigmoid 대신 tanh함수를 이용하면 개선. tanh함수를 이용하면 말끔한 종모양으로 개선.
            #활성화 함수용으로는 원점에서 대칭인 함수가 바람직.(tanh함수)
            
            #ReLU를 사용할때는 He초깃값, sigmoid나 tanh는 Xavier사용.
    #w=np.random.randn(node_num,node_num)*np.sqrt(2.0/node_num)

    a=np.dot(x,w)
    #활성화함수도 바꿔가며 실험
    #z=sigmoid(a)
    #z=ReLU(a)
    z=tanh(a)
    activations[i]=z

for i,a in activations.items():
    plt.subplot(1,len(activations),i+1)
    plt.title(str(i+1)+"-layer")
    if i!=0: plt.yticks([],[])#눈금 표시하기.
    #plt.xlim(0.1,1)
    #plt.ylim(0,7000)
    plt.hist(a.flatten(),30,range=(0,1)) #히스토그램 출력 #30은 선개수
plt.show()
#위 그래프의 hist값, 활성화의 histogram은 고루 분포되어야 한다.
#층과 층사이에 적당하게 다양한 데이터가 흐르게 해야 신경망 학습이 효율적으로 이루어지기 때문이다.
#반대로 치우친 데이터가 흐르면 기울기 소실이나 표현력 제한 문제에 빠져 학습이 잘 이뤄지지 않는다.
