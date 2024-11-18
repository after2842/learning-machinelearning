import numpy as np

class LogicGate:
    def __init__(self, gatename, xdata,tdata):
        self.name = gatename
        self.__xdata = xdata.reshape(4,2)
        self.__tdata = tdata.reshape(4,1)

        self.__w2 = np.random.rand(2,6)
        self.__b2 = np.random.rand(6)

        self.__w3 = np.random.rand(6,1)
        self.__b3 = np.random.rand(1)       #??????????

        self.__learning_rate = 1e-2


    

    def sigmoid(slef,x):
        return 1/(1+np.exp(-x))
    
    def feed_forward(self):
        delta = 1e-7

        z2 = np.dot(self.__xdata,self.__w2) + self.__b2
        a2 = self.sigmoid(z2)

        z3 = np.dot(a2,self.__w3) + self.__b3
        y = self.sigmoid(z3)

        return -np.sum(self.__tdata*np.log(y+delta) + (1-self.__tdata)*np.log(1-y+delta))
    
    def train(self):

        E = lambda x: self.feed_forward()

        for step in range(10001):
            self.__w2 = self.__w2- self.__learning_rate*self.numerical_derivative(E,self.__w2)
            self.__b2 = self.__b2 - self.__learning_rate*self.numerical_derivative(E,  self.__b2)

            self.__w3 = self.__w3- self.__learning_rate*self.numerical_derivative(E,self.__w3)
            self.__b3 = self.__b3 - self.__learning_rate*self.numerical_derivative(E,  self.__b3)
            if(step%400 ==0):
                print("step = ",step, " error value = ")

    def numerical_derivative(self,E,x):
        delta_x = 1e-9
        grad = np.zeros_like(x)

        it = np.nditer(x,flags=['multi_index'], op_flags=['readwrite'])
        #E = lambda x: self.__loss_func()
        while not it.finished:
            idx = it.multi_index
            #print("entire W matrix:",x)
            tmp_val = x[idx]
            x[idx] = float(tmp_val) + delta_x
            #print("fist x[idx]: ", x[idx] )
            fx1 = E(x)


            x[idx] = tmp_val - delta_x
            #print("second x[idx]: ", x[idx] )
            fx2 = E(x)

            grad[idx] = (fx1 - fx2)/(2*delta_x)         #collecting the change rate of w1,w2,w3. which means how much would each element affect the E() as we move w1,w2,w3,b while other are fixed so that we could update how much we should adjust them
            #print("grad:\n ", grad)
            x[idx] = tmp_val

            it.iternext()
        #print("////////////while end//////////////")
        return grad        
    
    def predict(self, input_data):
        z2 = np.dot(input_data,self.__w2)+self.__b2
        a2 = self.sigmoid(z2)

        z3 = np.dot(a2,self.__w3)+self.__b3
        y = self.sigmoid(z3)

        if y > 0.5:
            result = 1
        else:
            result = 0
        return y, result

xdata = np.array([[0,0],[0,1],[1,0],[1,1]])
tdata = np.array([0,1,1,0])

AND_obj = LogicGate("AndGate",xdata,tdata)
AND_obj.train()

test_data = np.array([[0,0],[0,1],[1,0],[1,1]])

for input_data in test_data:
    (sigmoid_val, logical_val) = AND_obj.predict(input_data)
    print(input_data, "==", logical_val, "$n")