import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
sys.path.append('E:/C_Data/Desktop/22_1_summer/deep_learning')
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from common.optimizer import *

class SGD:
    def __init__(self,lr=0.01):
        self.lr=lr
    def update(self,params,grads):
        for key in params.keys():
            params[key]-=self.lr*grads[key]
#for i in range(10000):
#---optimizer.update(params,grads)
#업데이트하는데 필요한 것은 매개변수와 기울기만 넘겨주면 된다.

#6.1.7
def f(x, y): #함수 f = x^2/20+y^2
    return x**2 / 20.0 + y**2 


def df(x, y): #기울기 함수 df
    return x / 10.0, 2.0*y

init_pos = (-7.0, 2.0) #초기위치
params = {}
params['x'], params['y'] = init_pos[0], init_pos[1]
grads = {}
grads['x'], grads['y'] = 0, 0


optimizers = OrderedDict() #순서대로 배열입력받기.
optimizers["SGD"] = SGD(lr=0.95)
optimizers["Momentum"] = Momentum(lr=0.1)
optimizers["AdaGrad"] = AdaGrad(lr=1.5)
optimizers["Adam"] = Adam(lr=0.3)

idx = 1

for key in optimizers:
    optimizer = optimizers[key]
    print(optimizers[key])
    x_history = []
    y_history = []
    params['x'], params['y'] = init_pos[0], init_pos[1]
    
    for i in range(30):
        x_history.append(params['x']) #x_history에 저장.
        y_history.append(params['y']) #y_history에 저장.
        
        grads['x'], grads['y'] = df(params['x'], params['y']) #기울기 구하기
        optimizer.update(params, grads) #매개변수와 기울기 전달하여 매개변수 갱신 함수 사용.
    

    x = np.arange(-10, 10, 0.01)
    y = np.arange(-5, 5, 0.01)
    
    X, Y = np.meshgrid(x, y) #격자그리드 만들기.
    Z = f(X, Y)
    
    # 외곽선 단순화
    mask = Z > 7
    Z[mask] = 0 #Z가 7보다 크면 0으로.
    
    # 그래프 그리기
    plt.subplot(2, 2, idx)
    idx += 1
    plt.plot(x_history, y_history, 'o-', color="red")
    plt.contour(X, Y, Z) #등치선 표현.
    plt.ylim(-10, 10)
    plt.xlim(-10, 10)
    plt.plot(0, 0, '+')
    #colorbar()
    #spring()
    plt.title(key)
    plt.xlabel("x")
    plt.ylabel("y")
    
plt.show()