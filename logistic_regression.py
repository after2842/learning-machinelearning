import numpy as np


loaded_data = np.loadtxt('./example.csv', delimiter=',', dtype=np.float32)

x_data = loaded_data[: , 0:-1]
t_data = loaded_data[ : , [-1]]


def loss_func(X,t):
    delta = 1e-7
    z = np.dot(X,W) +b
    y = sigmoid(z)
    return -np.sum(t*np.log(y+delta) + (1-t)*np.log(1-y+delta))


def numerical_derivative(E, x):
    delta_x = 1e-9
    grad = np.zeros_like(x)

    it = np.nditer(x,flags=['multi_index'], op_flags=['readwrite'])

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

def sigmoid(x):
    return 1/(1+np.exp(-x))

def predict(x):
    z = np.dot(x,W)+b
    y = sigmoid(z)

    if y > 0.5:
        result = 1
    else:
        result = 0
    return y, result

learning_rate = 1e-5

E = lambda x : loss_func(x_data, t_data)

W = np.random.rand(2,1)
b = np.random.rand(1)


for step in range(1000):
    W = W- learning_rate*numerical_derivative(E,W)
    b = b - learning_rate*numerical_derivative(E, b)

    if(step%100 ==0):
        print("step = ",step, " error value = ")

test_data = np.array([10,1])
print("predict: ",predict(test_data), W[0],W[1])