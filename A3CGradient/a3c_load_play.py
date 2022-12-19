# A3C load_play (tf2 version)
# coded by St.Watermelon
## 학습된 신경망 파라미터를 가져와서 에이전트를 실행시키는 파일

# 필요한 패키지 임포트
import gym
import tensorflow as tf
from a3c_learn import A3Cagent
import cv2
import itertools
import imageio

def main():

    env_name = 'Pendulum-v1'
    env = gym.make(env_name, max_episode_steps=300, render_mode='rgb_array')

    agent = A3Cagent(env_name) # A3C 에이전트 객체
    # 글로벌 신경망 파라미터 가져옴
    agent.load_weights()
    video_writer = cv2.VideoWriter(agent.save_path.joinpath('project').with_suffix('.mp4').as_posix(),
                                cv2.VideoWriter_fourcc(*'DIVX'), 15, (500,500))
    time = 0
    state, info = env.reset() # 환경을 초기화하고, 초기 상태 관측
    images = []

    while True:
        time_image = env.render()
        video_writer.write(time_image)
        images.append(time_image)

        # 행동 계산
        action = agent.global_actor(tf.convert_to_tensor([state], dtype=tf.float32))[0][0]
        # 환경으로부터 다음 상태, 보상 받음
        state, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        time += 1

        print('Time: ', time, 'Reward: ', reward, time_image.shape)

        if done:
            break

    env.close()
    video_writer.release()
    imageio.mimsave(agent.save_path.joinpath('project').with_suffix('.gif').as_posix(), images, fps=15)

if __name__=="__main__":
    main()
