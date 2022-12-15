## 개발 환경
### 환경
OS : Ubuntu 18.04<br>
Python : 3.8.5<br>
Tensorflow : 2.5.0<br>
Cuda : 11.3<br>
gym : 0.26.2<br>

### 개발환경 변경에 따른 코드 변경
```python
# 1. gym에서 Pendulum-v0 지원 불가
env_name = 'Pendulum-v1'
->
env_name = 'Pendulum-v0'

# 2. env.reset()
state = self.env.reset()
->
state, info = self.env.reset()

# 3. env.step()
next_state, reward, done, _ = self.env.step(action)
->
next_state, reward, term, trunc, info = self.env.step(action)
done = term or trunc
```



## A2C

### 변경사항
- Episode 200 -> 300 | 200 episode에서는 학습이 잘 안되어 변경하였음


### A2C - a) 
```
중간층이 64-32-16으로 되어있는 데 실험을 통해서 최적 구조를 찾아보세요
(성능비교 테이블과 x축은 에피소드 y축이 보상인 학습 곡선을 그려서 비교분석하세요.)
```

### A2C - b) 
```
a)에서 찾은 최적구조를 아래와 같이 합쳐진 정책 신경망과 가치 신경망의 형태로 변경하여 성능을 비교하세요. 
(x축은 에피소드 y축이 보상인 학습 곡선을 그려서 비교하세요.)
```

### A2C - c) 
```
변경된 코드도 제출하세요.
```