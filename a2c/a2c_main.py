# A2C main
# coded by St.Watermelon

## 에이전트를 학습하고 결과를 도시하는 파일
# 필요한 패키지 임포트
from a2c_learn import A2Cagent
import gym
import itertools


def main(depth1, depth2, depth3):

    max_episode_num = 1000   # 최대 에피소드 설정
    env_name = 'Pendulum-v1'
    env = gym.make(env_name, max_episode_steps=300)  # 환경으로 OpenAI Gym의 pendulum-v0 설정
    agent = A2Cagent(env, depth1 = depth1, depth2 = depth2, depth3 = depth3)

    # 학습 진행
    agent.train(max_episode_num)

    # 학습 결과 도시
    agent.plot_result()

if __name__=="__main__":
    num_layers = 3
    node_options = [4, 16, 32, 64, 128]
    layer_possibilities = [node_options] * num_layers # 3 Layer 선택 필요
    layer_node_permutations = list(itertools.product(*layer_possibilities))

    layer_node_permutations = [[d1, d2, d3] for d1, d2, d3 in layer_node_permutations if d1 > d2 >= d3]
    layer_node_permutations = [[d1, d2, d3] for d1, d2, d3 in layer_node_permutations if d1 >= 32 and d2 >= 16 and d3 <= 16]
    
    for depth1, depth2, depth3 in layer_node_permutations:
        print(f"Depth 1 : {depth1}, Depth 2 : {depth2}, Depth 3 : {depth3}")
        # main(depth1, depth2, depth3)
    print(len(layer_node_permutations))