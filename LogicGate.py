import numpy as np

class LogicGate:
    def __init__(self, gatename, xdata,tdata):
        self.name = gatename
        self.__xdata = xdata.reshape(4,2)
        self.__tdata = tdata.reshape(4,1)

        self.__w = np.random.rand(2,1)
        self.__b = np.random.rand(1)

        self.__learning_rate = 1e-2

    def __loss_func(self):

        delta = 1e-7
        z = np.dot(self.__xdata,self.__w) +self.__b
        y = self.sigmoid(z)
        return -np.sum(self.__tdata*np.log(y+delta) + (1-self.__tdata)*np.log(1-y+delta))
    

    def sigmoid(slef,x):
        return 1/(1+np.exp(-x))
    
    def train(self):
        learning_rate = 1e-5
        E = lambda x: self.__loss_func()

        for step in range(8001):
            self.__w = self.__w- learning_rate*self.numerical_derivative(E,self.__w)
            self.__b = self.__b - learning_rate*self.numerical_derivative(E,  self.__b)

            if(step%1000 ==0):
                print("step = ",step, " error value = ")

    def numerical_derivative(self,E,x):
        delta_x = 1e-9
        grad = np.zeros_like(x)

        it = np.nditer(x,flags=['multi_index'], op_flags=['readwrite'])
        E = lambda x: self.__loss_func()
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
        z = np.dot(input_data,self.__w)+self.__b
        y = self.sigmoid(z)

        if y > 0.5:
            result = 1
        else:
            result = 0
        return y, result

xdata = np.array([[0,0],[0,1],[1,0],[1,1]])
tdata = np.array([0,0,0,1])

AND_obj = LogicGate("AndGate",xdata,tdata)
AND_obj.train()

test_data = np.array([[0,0],[0,1],[1,0],[1,1]])

for input_data in test_data:
    (sigmoid_val, logical_val) = AND_obj.predict(input_data)
    print(input_data, "==", logical_val, "$n")