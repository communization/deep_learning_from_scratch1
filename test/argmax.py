import numpy as np
from collections import OrderedDict
a=np.array([[1,2,3,4,5],[1,2,4,4,3],[5,4,3,2,4],[3,4,2,4,3]])
b=np.argmax(a,axis=1) #최대값 인덱스
print(b)
print("a.shape: ", a.shape)

c=np.array([1])
d=np.random.choice(4,2)
print("np.random.choice( ): ",d)

print(a[d])
print((3<0))

x=np.array([[1.0, -0.5],[-2.0,3.0]])
mask=x<=0
print(mask)
x[mask]=0
print(x)

ordered_dic = OrderedDict()
ordered_dic['A'] = 1
ordered_dic['B'] = 2
ordered_dic['C'] = 3
print(ordered_dic.values())
for layer in ordered_dic.values():
    #x=layer.forward(x)
    print(layer)

aa=np.array([[1,2,3,4,5],[6,7,8,9,0],[11,12,13,14,15]])
print(aa.shape)
print(aa[:2])
print(np.sum(aa,axis=0)) #[18 21 24 27 20] 열방향덧셈.
print("aa[1:]  " ,aa[1:])
print(aa[1])
class SGD:
    def __init__(self,lr=0.01):
        self.lr=lr
    def update(self,params):
        params[1]=np.array([1,2, 3, 4 ,6])

optimizer=SGD()
optimizer.update(aa)
print(aa)
print(aa[1,:]) if 1<0 else print(aa[2,:]) #[11 12 13 14 15]