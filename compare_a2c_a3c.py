from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import shutil

model_name = '128_64_16'

a2c_path = Path('a2c').joinpath('Results').joinpath(f'Model_{model_name}')
a3c_path = Path('A3CGradient').joinpath('Results', f'Model_{model_name}')

save_path = Path('Compare_A2C_A3C').joinpath(f'Model_{model_name}')
save_path.mkdir(exist_ok = True, parents = True)


def draw_plot(data, title, xlabel, ylabel, model_name, filename):
    """Draw plot of data and save it to file."""
    plt.clf()
    fig, ax = plt.subplots()
    sns.lineplot(data, x = 'episodes', y = 'rewards', ax=ax, label=model_name)
    ax.set_ylim(-2500, 0)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid()
    fig.savefig(filename)
    plt.close(fig)

    plt.clf()
    fig, ax = plt.subplots()
    sns.lineplot(data, x = 'episodes_10', y = 'rewards', ax=ax, label=model_name)
    ax.set_ylim(-2500, 0)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid()
    fig.savefig(filename.with_name('scaled_' + filename.name))
    plt.close(fig)

def draw_total_plot(data, title, xlabel, ylabel, filename):
    """Draw plot of data and save it to file."""
    plt.clf()
    fig, ax = plt.subplots(figsize = (10,8))
    custom_palette = sns.color_palette("Paired", 12)
    for model_name in data.columns[2:]:
        sns.lineplot(data, x = 'episodes', y = model_name, ax=ax, label=model_name, palette=custom_palette)
    ax.set_ylim(-3000, 0)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid()
    fig.savefig(filename)
    plt.close()

    plt.clf()
    fig, ax = plt.subplots(figsize = (10,8))
    custom_palette = sns.color_palette("Paired", 12)
    for model_name in data.columns[2:]:
        sns.lineplot(data, x = 'episodes_10', y = model_name, ax=ax, label=model_name, palette=custom_palette)
    ax.set_ylim(-3000, 0)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid()
    fig.savefig(filename.with_name('scaled_' + filename.name))
    plt.close()

total_rewards = []


for model_name, path in [['A2C', a2c_path], ['A3C', a3c_path]] : 
    reward_path = list(path.glob('*reward.txt'))[0]
    gif_path = list(path.glob('*.gif'))[0]
    shutil.copyfile(gif_path, save_path.joinpath(f"{model_name}.gif"))

    with reward_path.open() as f:
        rewards = np.loadtxt(f)
    rewards = pd.DataFrame(rewards, columns = ['rewards']).reset_index().rename(columns = {'index' : 'episodes'})
    rewards['episodes_10'] = rewards['episodes'] // 10 * 10
    total_rewards.append([rewards, model_name])
    draw_plot(rewards, 'Rewards by Episode', 'Episode', 'Reward', model_name, save_path.joinpath(f'{model_name}_reward.png'))
    


total_reward_df = pd.DataFrame(range(1007), columns = ['episodes'])
total_reward_df['episodes_10'] = total_reward_df['episodes'] // 10 * 10
print(total_rewards)
for rewards, model_name in total_rewards:
    if rewards['rewards'].shape[0] == 1000 : 
        rewards = list(rewards['rewards'].values) + [rewards['rewards'].values[-1]]*7
    else : 
        rewards = list(rewards['rewards'].values)
    total_reward_df[model_name] = rewards
draw_total_plot(total_reward_df, 'Rewards by Episode', 'Episode', 'Reward', save_path.joinpath(f'total_reward.png'))