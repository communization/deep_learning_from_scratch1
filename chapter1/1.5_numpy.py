import numpy as np
x= np.array([1.0, 2.0, 3.0])
y= np.array([2.0, 4.0, 6.0])
print(x), print(type(x))
print(x+y), print(x*y), print(x/y) ,print(x/2.0)#element-wise

A=np.array([[1,2],[3,4]])
print(A), print(A.shape), print(A.dtype) #1.5.4
B=np.array([[3,0],[0,6]])
print(A+B), print(A*B) #1.5.4

## 1.5.5
print (A*10)
B=np.array([[10],[20]]) #[10; 20]
print (A*B) #broadcast
B=np.array([[10,20]]) #[10 20]
print (A*B) #broadcast

#1.5.6
X=np.array([[51,55],[14,19],[0,4]])
print(X), print(X[0]), print(X[0][1])
for row in X:
    print(row)
X=X.flatten()
print(X)
print(X[np.array([0,2,4])])

print(X>15) #true false
print(X[X>15]) # get true element

