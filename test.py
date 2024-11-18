import split
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
a= [1,2,3,4,100, [3.14,False]]

print("a[-1]", a[-1])
print("a[2:]= ", a[2:]) #slicing
a[-1] = [1,False]
print("a[0:]", a[0:])
print(a[-1][0])

b = (1,2,3,99,[False,2222])  #tuple

dictionary = {"kim": 1, "sam": 22}

print(dictionary, dictionary.keys())

a1="samuel, choi"

a1 = a1+", kim"
print(a1[3])
b2 = a1.split(',')
# print(len(a), size(a)) # size ---> indicates #of EVERY elements


A = "hello"
B = {"Samuel": 77, "Elosie": 100}
C = '3.141592'
D = 3.1415

print(list(A), list(B.keys()), str(D))

for data in range(100):
    print(data,end='')

print(" ")
for data in range(0,100,10):
    print(data,end='')

for data in B.keys():
    print(data)


raw_data = [[1,0],[2,2],[40,200]]

all_data = [x for x in raw_data]
first = [y[0] for y in raw_data]
print(all_data)
print(first)

quiz = [int(x/2) for x in range(10)]

print(quiz)


data1 = 1
while data1<10:
    print(data1)
    data1 = data1+1


def random_function(x,y, count = "2"):


    return x*y,x*x,int(count)*100
    

result,result2,new = random_function(10,11)
print(result,result2,new)
result,result2,new = random_function(10,11,1000)
print(result,result2,new)


#immutable -> number, char, tuple.etc (will not be cahnged through function)
#mutable -> list,dict,numpy (can be changed through function)


f = lambda x: x+10

for i in range(10):
    print("lambda:", f(i))

i=10
j=200

def print_hello():
    print("hello samuel")

def whateverfunction(i,j):
    print(i+j)


fx = lambda x,y: whateverfunction(i,j)      #doesn't necesarily to use x and y
fy = lambda x,y: print_hello()

fx(1,1)
fy(1100010,321)

def print_name():  
    print("external func")

class Person:
    count= 0

    def __init__(self,name):
        self.name = name
        self.__something = name+"private"
        Person.count = Person.count+1
        print(self.name,"is initialized")

    def work(self, company):
        print(self.name,"is working at", company)

    def sleep(self, hour):
        print(self.name,"slept",hour)

    def __setprivate(self):
        return self.__something
    def print_name(self):
        print("class method printname")
    def call_print(self):
        self.print_name()

    @classmethod
    def getcount(cls):
        return cls.count

obj = Person("samuel")
obj2 = Person("eloise")
obj3 = Person("eloise")
obj.work("google")
obj.sleep("9")
#print(obj.__setprivate())##error
print(obj.getcount())



'''
def calc(list_data):    #try except syntax

    try:
        sum = list_data[0]+list_data[1]

        if(sum<0):
            raise Exception("sum is less than 0")
        
    except IndexError as err:
        print(err)
    finally:
        sum


list_data = [-1,-100]

calc(list_data)
'''


with open("./README.md", 'w') as f:
    f.write("this is me!!!!!!!")
    #doens't need to close 


A = np.array([[1,1],[2,2]])
B = np.array([[1,1,4],[5,2,4]])
#print(A+B)
print(A.shape, B.shape, A.ndim)

D= np.array( [[1 ,1],[2 ,2],[2, 2]] )

#C = A.reshape(3,2)
#print(C,C.shape,C.ndim)
#print(D,D.ndim)


E = np.dot(A,B)       # why use dot product ----> we can get the expected result's shape wheras +-/* can't
print(E)

K = B.T
print("k: ", K)

A1 = np.array([10,20,30,40,50,60]).reshape(3,2)
print("A1: ",A1)
print("A[0:-1, 1:2]=", A1[0:-1, 1:2])



J = np.array([[10,20,30],[40,50,60]])
row_add = np.array([1,1,1]).reshape(1,3)

P = np.concatenate((J,row_add),axis=0)  #axis=0 ()
print(P)



#loaded_data = np.loadtxt('./data.csv',delimiter = ',', dtype=float32) #store data.csv as a 25*4matrix

#x_data = loaded_data[:, 0:-1] # save everything except for the last colum
#y_data = loaded_data[:, [-1]] # save only colum-2



X = np.array([[22,4,6], [1,2,3],[0,5,8]])

print(np.max(X,axis=0))


temp1 = [x*2 for x in range (-10,10)]
temp2 = np.random.rand(20)

plt.title('EXAMPLE')
plt.grid()
plt.plot(temp1,temp2,color='b')
#plt.show()

def func(x,y):
    return 2*x+3*x*y+y**3

def numerical_derivative(func,x,y):
    
    deltax=1e-4
    ((func(x+deltax,y)-func(x-deltax,y))/(2*deltax))
    print("x: ",((func(x+deltax,y)-func(x-deltax,y))/(2*deltax)))
    print("y: ",((func(x,y+deltax)-func(x,y-deltax))/(2*deltax))) 

def f(input):
    #temp = x.reshape(1,0)
    w = input[0,0]
    x = input[0,1]
    y = input[1,0]
    z = input[1,1]

    return (w*x + x*y*z + 3*w + z*np.power(y,2))

def numerical_derivative(f,x):
    delta_x = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x,flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index

        tmp_val = x[idx]
        x[idx] = float(tmp_val) + delta_x
        fx1 = f(x)

        x[idx] = tmp_val -delta_x
        fx2 = f(x)

        grad[idx] = (fx1 - fx2)/(2*delta_x)
        print("grad:\n ", grad)
        x[idx] = tmp_val

        it.iternext()
    return grad

x = np.array([[1.0, 2.0],[3.0,4.0]])
print("numerical_derivative: \n", numerical_derivative(f, x))
