# A2C load and play (tf2 version)
# coded by St.Watermelon

## 학습된 신경망 파라미터를 가져와서 에이전트를 실행시키는 파일
# 필요한 패키지 임포트
import gym
import tensorflow as tf
from a2c_learn import A2Cagent
import cv2
import itertools


def main(depth1, depth2, depth3):

    env_name = 'Pendulum-v1'
    env = gym.make(env_name, max_episode_steps=400, render_mode='rgb_array')

    agent = A2Cagent(env, depth1 = depth1, depth2 = depth2, depth3 = depth3)

    agent.load_weights()  # 신경망 파라미터를 가져옴
    video_writer = cv2.VideoWriter(agent.save_path.joinpath('project').with_suffix('.mp4').as_posix(),
                                cv2.VideoWriter_fourcc(*'DIVX'), 15, (500,500))

    time = 0
    state, info = env.reset() # 환경을 초기화하고 초기 상태 관측

    while True:
        time_image = env.render()
        video_writer.write(time_image)
        
        action = agent.actor(tf.convert_to_tensor([state], dtype=tf.float32))[0][0] # 행동 계산
        state, reward, term, trunc, _ = env.step(action)  # 환경으로 부터 다음 상태, 보상 받음
        time += 1
        
        print('Time: ', time, 'Reward: ', reward)
        done = term or trunc
        if done:
            break
    env.close()
    video_writer.release()

if __name__=="__main__":
    num_layers = 3
    node_options = [4, 16, 32, 64]
    layer_possibilities = [node_options] * num_layers # 3 Layer 선택 필요
    layer_node_permutations = list(itertools.product(*layer_possibilities))

    layer_node_permutations = [[d1, d2, d3] for d1, d2, d3 in layer_node_permutations if d1 > d2 >= d3]
    layer_node_permutations = [[d1, d2, d3] for d1, d2, d3 in layer_node_permutations if d1 >= 32 and d2 >= 16 and d3 <= 16]
    for depth1, depth2, depth3 in layer_node_permutations:
        print(f"Depth 1 : {depth1}, Depth 2 : {depth2}, Depth 3 : {depth3}")
        main(depth1,depth2,depth3)