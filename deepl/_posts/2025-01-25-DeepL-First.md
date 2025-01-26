---
layout: post
title:  "DeepL #신경망학습 - 신경망을 공부할때 어떤게 중요할까요?"
date:   2025-01-26 11:00:00 +0900
categories: 
  - studylog
  - deepl
description: >
  신경망에 대한 기초 내용입니다.

---

# DeepL #신경망학습 - 신경망을 공부할때 어떤게 중요할까요?



## 🎖️ 중심 내용


---

<aside>
🔥 역전파, 손실함수, 옵티마이저
</aside>

## 🧑🏻‍💻 정리 내용

---

### ☑️ 어떤 Optimizer를 사용해야하는가?


![image1](http://Moomin03.github.io/deepl/img/2025-01-25/image.png)


### ☑️ Gradient Vanishing은 어떻게 막는가?

<aside>
☝🏻

역전파 과정에서 손실 함수와 활성화 함수는 중요한 역할을 하는데, 여기서 중요한 것은 각 가중치가 오차에 미치는 영향을 알기 위해 각각에 대해 미분을 해야 한다는 점입니다. 손실 함수의 미분은 오차가 얼마나 클지, 즉 예측값과 실제값의 차이에 대해 계산을 진행하고 활성화 함수의 미분은 그 결과가 오차에 어떻게 영향을 미쳤는지 계산합니다. 왜냐하면, 활성화 함수는 각 뉴런의 출력값을 결정하기 때문입니다.

먼저 tanh 함수입니다. tanh의 미분값을 보면 0일때 1로 최대값을 가지고, 다른 값들일때는 0과 1 사이의 작은 값들을 가지게 됩니다. 이런 값들이 각 층에서 곱해지면, 아마도 0에 가까워지게되면서 기울기 소실 문제가 발생합니다.

$$
tanh(x) = \frac{e^x-e^{-x}}{e^x+e^{-x}}
$$

$$
tanh'(x) = 1-\frac{(e^x-e^{-x})^2}{(e^x+e^{-x})^2}
$$
</aside>

![image2](http://Moomin03.github.io/deepl/img/2025-01-25/tanh.jpg)
<aside>

다음은 시그모이드 함수입니다. 시그모이드 함수도 tanh 함수와 유사한 문제를 띄죠.

$$
sigmoid(x) = \frac{1}{1+e^{-x}}
$$

$$
sigmoid'(x) = \frac{1}{1+e^{-x}}*\frac{e^{-x}}{1+e^{-x}}
$$
</aside>

![image3](http://Moomin03.github.io/deepl/img/2025-01-25/sigmoid.jpg)

<aside>
따라서 ReLU 함수를 사용하긴 하는데, 양수일때 미분값이 무조건 1을 갖는다는 점은 긍정적이지만, 음수인 값에 대해서는 0을 갖는다는 문제도 여전히 가지고 있기는 합니다.

$$
ReLU =  \begin{Bmatrix}
0 & x<0 \\
x & x>0 \\
\end{Bmatrix}
$$

$$
ReLU' =  \begin{Bmatrix}
0 & x<0 \\
1 & x>0 \\
\end{Bmatrix}
$$
</aside>

![image4](http://Moomin03.github.io/deepl/img/2025-01-25/relu.jpg)

<aside>
따라서 이런 문제를 해결하기 위해서 LeakyReLU를 사용하거나, 다른 형태의 ReLU함수를 사용하기도 합니다.

</aside>

### ☑️ 일반적인 지도학습의 과정

<aside>
☝🏻

[순서1] : 모든 커넥션에 임의에 가중치를 부여합니다.

[순서2] : 가중치를 이용하여 라벨(Y)과 예측값(Prediction)의 에러(Error)를 계산합니다.

[순서3] : 에러(Error)를 각 가중치로 미분하여 기울기(Gradient)를 구합니다.

[순서4] : 기울기에 학습을 곱하고 이 값을 기존 가중치에서 빼주어 가중치를 업데이트합니다.

[순서4] → [순서2]로 돌아갑니다.

</aside>

### ☑️ 에러를 각 가중치로 미분하는 과정

<aside>
☝🏻

일반적인 회귀에서 손실함수는 아래와 같습니다.

$$
MSE = L = \frac{(y_{i}-\hat y_{i})^2}{n}
$$

각각의 가중치가 L(손실함수)에 어떤 영향을 미치는지 알고자 L을 w에 대해여 편미분을 진행합니다.

$$
\frac{\partial L}{\partial w_{j}}
$$

그전에 알아두어야할 내용이, L에는 w가 없는데 어떻게 편미분을 진행하는지에 대한 부분입니다.

$$
\hat y_{i} = \sum_{j=1}^{m}x_{ij}w_{j} + b
$$

예측값이 이미 w에 대한 식이 들어가있으므로 편미분을 진행할 수 있겠죠?

$$
\frac{\partial L}{\partial w_{j}} = \frac{\partial}{\partial w_{j}}\left [ \frac{1}{n}\sum_{i=1}^{n}(y_{i}-\hat y_{i})^2 \right ]
$$

오차의 제곱에 대해서 편미분을 진행하면

$$
\frac{\partial}{\partial w_{j}}(y_{i}-\hat y_{i})^2 = 2(y_{i}-\hat y_{i})*\frac{\partial}{\partial w_{j}}(y_{i}-\hat y_{i})
$$

오차에 대해서 편미분을 진행하면

$$
\frac{\partial}{\partial w_{j}}(y_{i}-\hat y_{i}) = -w_{j}
$$

다음과 같은 식이 되므로 따라서 편미분 값은 아래와 같습니다.

$$
\frac{\partial L}{\partial w_{j}} = -\frac{2}{n}\sum_{i=1}^{n}(y_{i}-\hat y_{i})w_{j}
$$

</aside>

### ☑️ 가중치 업데이트

<aside>
☝🏻

가중치의 업데이트는 아래와 같이 정의되고 Eta는 학습률을 의미한다.

$$
w_{i+1} = w_{i}-\eta \frac{\partial L}{\partial w}
$$

왜 기존 가중치에서 미분값을 빼냐면?

MSE 식은 2차 함수의 형태, 즉 포물선 형태를 가지는데 임의의 점이 기울기가 0인 지점으로 향하기 위해서는 기울기의 값을 빼주어야함.

</aside>

## 🎁 향후 공부 방향

---

😀 내일은 역전파의 계산 과정에 대해서 서술할 예정!