#5.4.1 mullayer
class MulLayer:
    def __init__(self):
        self.x=None
        self.y=None
    
    def forward(self,x,y):
        self.x=x
        self.y=y
        out = x*y
        return out
    
    def backward(self,dout):
        dx=dout*self.y #x와 y를 바꾼다.
        dy=dout*self.x
        return dx,dy

apple=100
apple_num=2
tax=1.1

#계층들
mul_apple_layer=MulLayer()
mul_tax_layer=MulLayer()

#순전파
apple_price=mul_apple_layer.forward(apple,apple_num) #100*2
price=mul_tax_layer.forward(apple_price,tax) #200*1.1
print(price) #220

#역전파
dprice=1
dapple_price,dtax=mul_tax_layer.backward(dprice) #dx=dapple_price=1*tax(=self.y) =1.1 / dy=dtax=1*apple_price(=self.x)=200
dapple,dapple_num=mul_apple_layer.backward(dapple_price) #dx=dapple=dapple_price(=1.1)*apple_num(=self.y)=1.1*2  / dy=dapple_num=dapple_price(=1.1)*apple(self.x)=1.1*100
print(dapple,dapple_num,dtax)

#5.4.2 addlayer
class AddLayer:
    def __init__(self):
        pass #아무일도 하지 않는다. (초기화가 필요 없음)
    def forward(self,x,y):
        out=x+y
        return out
    def backward(self,dout):
        dx=dout*1
        dy=dout*1
        return dx,dy

apple=100
apple_num=2
orange=150
orange_num=3
tax=1.1

#게층들
mul_apple_layer=MulLayer()
mul_orange_layer=MulLayer()
add_apple_orange_layer=AddLayer()
mul_tax_layer=MulLayer()

#순전파
apple_price=mul_apple_layer.forward(apple,apple_num) #(1)
orange_price=mul_orange_layer.forward(orange,orange_num) #(2)
all_price=add_apple_orange_layer.forward(apple_price,orange_price) #(3)
price =mul_tax_layer.forward(all_price,tax) #(4)

#역전파
dprice=1
dall_price,dtax=mul_tax_layer.backward(dprice) #(4)
dapple_price, dorange_price=add_apple_orange_layer.backward(dall_price) #(3)
dorange,dorange_num=mul_orange_layer.backward(dorange_price) #(2)
dapple,dapple_num=mul_apple_layer.backward(dapple_price) #(1)

print(price)
print(dapple_num,dapple,dorange,dorange_num,dtax)
