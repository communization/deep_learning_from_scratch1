import numpy as np
#3.3.1
A=np.array([1,2,3,4])
print(A), print(np.ndim(A)), print(A.shape), print(A.shape[0]) # A.shape -> type: tuple

B=np.array([[1,2],[3,4],[5,6]])
print(B), print(np.ndim(B)), print(B.shape), print(B.shape[1])

#3.3.2 matrix product
A=np.array([[1,2],[3,4]])
B=np.array([[5,6],[7,8]])
print(A.shape), print(B.shape), print(np.dot(A,B))

A=np.array([[1,2,3],[4,5,6]])
B=np.array([[1,2],[3,4],[5,6]])
print(A.shape), print(B.shape), print(np.dot(A,B))

#3.3.2 matrix product error case
C=np.array([[1,2],[3,4]])
#print(C.shape), print(A.shape), print(np.dot(A,C))

A=np.array([[1,2],[3,4],[5,6]])
B=np.array([[7],[8]])
print(A.shape), print(B.shape), print(np.dot(A,B))
A=np.array([[1,2],[3,4],[5,6]])                    #3x2 product 2x1 => 3x1
B=np.array([7,8])
print(A.shape), print(B.shape), print(np.dot(A,B)) #3x2 product 1x2 => 1x3

#3.3.3
X=np.array([1,2])
W=np.array([[1,3,5],[2,4,6]])
Y= np.dot(X,W)
print(X.shape),print(W),print(W.shape),print(Y)    #1x2 product 2x3 => 1x3
