import numpy as np

loaded_data = np.loadtxt('./sueng_scores.csv', delimiter=',', dtype=np.float32)

x_data = loaded_data[: , 0:-1]
t_data = loaded_data[ : , [-1]]

W = np.random.rand(3,1)
b = np.random.rand(1)

print("W = ", W, "W.shape = ", W.shape, "b.sahpe = ", b.shape)

def loss_func(x, t):
    y = np.dot(x, W) + b
    return (np.sum((t-y)**2)/len(x))

def numerical_derivative(E, x):
    delta_x = 1e-4
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

def error_val(x, t):
    y = np.dot(x,W) + b

    return (np.sum((t-y)**2)/len(x))

def predict(x):
    y = np.dot(x,W)+b
    return y

learning_rate = 1e-5

E = lambda x : loss_func(x_data, t_data)

print("initial error value = ",error_val(x_data, t_data), "initial W = ", W, "Wn,",",b = ", b)

for step in range(100):
    W = W-learning_rate*numerical_derivative(E,W)
    b = b - learning_rate*numerical_derivative(E, b)

    if(step%10 ==0):
        print("step = ",step, " error value = ", error_val(x_data, t_data))

test_data = np.array([100,100,100])
print("predict: ",predict(test_data))





'''
오차함수 E는 W와 b에 라는 독립변수에 의해 결정되는 함수이다. 
다시, W는 3개의 변수 w1,w2,w3로 이루어져 있다. 따라서 
오차함수 E는, 나머지 변수(w2,w3,b)를 고정했을 때 w1에 따라서 결정되는 2차함수이다.  E'(w1) 는 국어점수가 오차함수에 미치는 영향의 변화 방향과 크기를 나타낸다. 
마찬가지로 w2,w3도 각각 진행하면 각각 영어와 수학점수가 오차함수에 미치는 영향의 변화 방향과 크기를 나타낸다.  E(W,b)는 사실상 E(w1,w2,w3,b)인 네개의 독립변수를 갖는 다변수함수라고 할 수 있다.
그렇게해서 얻어낸 새로운 Wmatrix는 (원래W-학습계수*E를 W에대해 편미분한 값) 한단계 학습된 W이다. 

결론적으로, E를 W에 대해서 편미분 하는 이유는. W의 각 성분 w1,w2,w3가 오차함수에 영향을 미치는 정도를 계산해서 영향을 미치는 정도만큼 W의 성분들을 조절하기 위함이다.
'''