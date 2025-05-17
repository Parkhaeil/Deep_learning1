# def AND(x1, x2):
#     w1, w2, theta = 0.5, 0.5, 0.7
#     tmp = x1*w1 + x2+w2
#     if tmp <= theta:
#         return 0
#     elif tmp > theta:
#         return 1

# 가중치와 편향을 도입한 AND 게이트
import numpy as np

def AND(x1, x2):
    x = np.array([x1,x2])   
    w = np.array([0.5,0.5]) # 가중치는 입력이 결과에 주는 영향력을 조절하는 매개변수
    b = -0.7                # 편향은 뉴런이 활성화를 얼마나 쉽게 하느냐를 조절하는 매개변수
    tmp = np.sum(x*w) + b
    if tmp <= 0:
        return 0
    else:
        return 1
    
def NAND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([-0.5,-0.5]) # AND랑 가중치와 편향만 다르다
    b = 0.7
    tmp = np.sum(x*w) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def OR(x1, x2):
    x = np.array([x1,x2])
    w = np.array([0.5, 0.5]) # AND랑 가중치와 편향만 다르다
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def XOR(x1, x2):
    s1 = NAND(x1,x2)
    s2 = OR(x1,x2)
    y =  AND(s1,s2)
    return y

