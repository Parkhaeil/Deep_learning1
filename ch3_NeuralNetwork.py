# def step_function(x): # float만 가능하고 배열은 안된다는 단점이 존재
#     if x > 0 :
#         return 1
#     else
#         return 0
import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    return np.array(x>0, dtype=int)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

def sigmoid(x):
    return 1 / (1 + np.exp(-x)) # x가 넘파이 배열이어도 정상작동, 브로드캐스트 덕분
    # 브로드캐스트란? 넘파이 배열과 스칼라 값의 연산을 가능하게 해주는 기능

def relu(x):
    return np.maximum(0,x)

### 신경망에서 행렬의 곱이 사용되므로 익혀놓자!

# 이것은 일차원 배열의 예시
# A = np.array([1,2,3,4])
# np.ndim(A)      >>> 1
# A.shape         >>> (4,)
# A.shape[0]      >>> 4


# B = np.array([[1,2],[3,4],[5,6]])
# np.ndim(B)  >>> 2
# B.shape     >>> (3,2) , 차원이 두 개
# 이차원 배열은 행렬이라고 한다

# np.dot(A,B)

### 3.4.3 3층 신경망 구현 정리
def identity_function(x):
    return x

def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1)
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    y  = identity_function(a3)

    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)


###3.5.1 소프트맥스 구현하기 (a가 1000만 넘어가도 오버플로우 발생)
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a -c) # 오버플로우 대책
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a 
    
    return y