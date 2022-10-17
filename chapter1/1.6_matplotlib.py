import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
#1.6.1
x=np.arange(0,6,0.1)
y=np.sin(x)
plt.plot(x,y)
plt.show()

#1.6.2
y1=np.sin(x)
y2=np.cos(x)
plt.plot(x,y1,label="sin"), plt.plot(x,y2,linestyle="--",label="cos")
plt.xlabel("x"), plt.ylabel("y")
plt.title('sin & cos')
plt.legend() #그래프의 종류 표시
plt.show()

#1.6.3
img= imread('../chapter1/Figure_1.png')
#img= imread('Figure_1.png')
plt.imshow(img)
plt.show()

