
import numpy as np
from PIL import Image

import sys, os
sys.path.append(os.pardir) # 부모 디렉터리 파일을 가져올 수 있도록 설정
from dataset.mnist import load_mnist


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img)) # 넘파이로 저장된 이미지 데이터를 PIL용 데이터 객체로 변환
                                             # unsigned int 8 bit
    pil_img.show()

(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]

print(label) # 정답이 5

print(img.shape) # (784,) 신경망의 입력은 벡터여야 하기 때문
img = img.reshape(28,28) # 이미지 출력할 때는 reshape 해줘야 함
print(img.shape) # (28, 28)

img_show(img)