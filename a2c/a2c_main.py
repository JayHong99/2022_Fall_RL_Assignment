# A2C main
# coded by St.Watermelon

## 에이전트를 학습하고 결과를 도시하는 파일
# 필요한 패키지 임포트
from a2c_learn import A2Cagent
import gym
import itertools


def main(actor_d1, actor_d2, actor_d3, critic_d1, critic_d2, critic_d3):

    max_episode_num = 1000   # 최대 에피소드 설정
    env_name = 'Pendulum-v1'
    env = gym.make(env_name, max_episode_steps=300)  # 환경으로 OpenAI Gym의 pendulum-v0 설정
    agent = A2Cagent(env, # A2C 에이전트 객체
                        gamma=0.95, batch_size=32, actor_lr=0.0001, critic_lr=0.001,
                        actor_d1=actor_d1, actor_d2=actor_d2, actor_d3= actor_d3,
                        critic_d1=critic_d1, critic_d2=critic_d2, critic_d3=critic_d3
                        )   

    # 학습 진행
    agent.train(max_episode_num)

    # 학습 결과 도시
    agent.plot_result()

if __name__=="__main__":
    num_layers = 3
    node_options = [4, 16, 32, 64]
    layer_possibilities = [node_options] * num_layers # 3 Layer 선택 필요
    layer_node_permutations = list(itertools.product(*layer_possibilities))

    layer_node_permutations = [[d1,d2,d3] for d1,d2,d3 in layer_node_permutations if d1 > d2 > d3]
    actor_node_permutations = layer_node_permutations
    critic_node_permutations = layer_node_permutations

    for actor_d1, actor_d2, actor_d3 in actor_node_permutations:
        for critic_d1, critic_d2, critic_d3 in critic_node_permutations:
            print(f"actor_d1: {actor_d1}, actor_d2: {actor_d2}, actor_d3: {actor_d3}, critic_d1: {critic_d1}, critic_d2: {critic_d2}, critic_d3: {critic_d3}")
            main(actor_d1, actor_d2, actor_d3, critic_d1, critic_d2, critic_d3)