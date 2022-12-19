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
env_name = 'Pendulum-v0'
->
env_name = 'Pendulum-v1'

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


### Question - a) 
```
중간층이 64-32-16으로 되어있는 데 실험을 통해서 최적 구조를 찾아보세요
(성능비교 테이블과 x축은 에피소드 y축이 보상인 학습 곡선을 그려서 비교분석하세요.)
```
<b> 실험 세팅 </b>
- <b>Node Options : 4, 16, 32, 64, 128</b>
    - 기존 64, 32, 16 Node에 4와 128을 선택하는 경우의 수를 추가하였음
    - Node는 각 Layer의 Node 수를 의미함

- <b> Layer Options </b>
    - 3층 Layer는 각각 Node Options에서 하나를 선택할 수 있음
    - 모든 Combination을 만들고, Selection Criteria를 만들었음
    - Selection Criteria
        - Layer 1 Node > Layer 2 Node >= Layer 3 Node
        - Layer 1 Node >= 32
        - Layer 2 Node >= 16
        - Lyaer 3 Node <= 16
    - Finally, 12 Selected Layer Options are selected
    
<b> 실험 </b>
```python
bash a2c/a2c_run.sh
```

<b> 결과 </b>
- Reward Plot

<img src = 'https://github.com/JayHong99/2022_Fall_RL_Assignment/blob/master/a2c/Results/total.png' width='800px' height='500px'>

- Reward Plot : 10 episode 단위로 average하고, 해당 구간 내의 reward의 std를 표시함
- 128 -> 64 -> 16의 Model Architecture 선정
    - 빠른 학습 속도
    - 가장 높은 reward (-0.03)

<img src = 'https://github.com/JayHong99/2022_Fall_RL_Assignment/blob/master/a2c/Results/scaled_total.png' width='800px' height='500px'>

![best Model](./a2c/Results/best_model.gif "Best Model")

### Question - b) 
```
a)에서 찾은 최적구조를 합쳐진 정책 신경망과 가치 신경망의 형태로 변경하여 성능을 비교하세요. 
(x축은 에피소드 y축이 보상인 학습 곡선을 그려서 비교하세요.)
```
<img src = 'https://github.com/JayHong99/2022_Fall_RL_Assignment/blob/master/a2c/Results/scaled_compare.png' width='800px' height='500px'>

### Question - c) 
```
변경된 코드도 제출하세요. 
python a2c/a2c_main2.py
python a2c/a2c_learn2.py # 해당 부분에서 수정
python a2c/a2c_load_play2.py
```

## A3C

### Question -a ) 
```
동기적 방법과 비동기적 방법을 비교분석하세요. 
(성능비교 테이블과 x축은 에피소드 y축이 보상인 학습곡선을 그려서 비교분석하세요.).
```

- Intermediate Model : 64 -> 32 -> 16
<img src = 'https://github.com/JayHong99/2022_Fall_RL_Assignment/blob/master/Compare_A2C_A3C/Model_64_32_16/scaled_total_reward.png' width='800px' height='500px'>
    - Synchronous 
<img src = 'https://github.com/JayHong99/2022_Fall_RL_Assignment/blob/master/Compare_A2C_A3C/Model_64_32_16/A2C.gif' width='800px' height='500px'>
    - Asynchronous
<img src = 'https://github.com/JayHong99/2022_Fall_RL_Assignment/blob/master/Compare_A2C_A3C/Model_64_32_16/A3C.gif' width='800px' height='500px'>

- Intermediate Model : 128 -> 64 -> 16
<img src = 'https://github.com/JayHong99/2022_Fall_RL_Assignment/blob/master/Compare_A2C_A3C/Model_128_64_16/scaled_total_reward.png' width='800px' height='500px'>
    - Synchronous 
<img src = 'https://github.com/JayHong99/2022_Fall_RL_Assignment/blob/master/Compare_A2C_A3C/Model_128_64_16/A2C.gif' width='800px' height='500px'>
    - Asynchronous
<img src = 'https://github.com/JayHong99/2022_Fall_RL_Assignment/blob/master/Compare_A2C_A3C/Model_128_64_16/A3C.gif' width='800px' height='500px'>

### Question - b) 
```
 기존 그래디언트 병렬화 방법을 데이터 병렬화 방법으로 변경하고 비교분석하세요. 
 (성능비교 테이블과 x축은 에피소드 y축이 보상인 학습곡선을 그려서 비교분석하세요.).
```
<img src = 'https://github.com/JayHong99/2022_Fall_RL_Assignment/blob/master/A3CGradient/Results/scaled_compare.png' width='800px' height='500px'>
- Integrated Model
<img src = 'https://github.com/JayHong99/2022_Fall_RL_Assignment/blob/master/A3CGradient/Results/Integrated_A3C/project.gif' width='800px' height='500px'>


### Question - c) 
```
변경된 코드도 제출하세요.
```