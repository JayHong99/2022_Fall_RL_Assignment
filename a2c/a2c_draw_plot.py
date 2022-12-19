from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

total_rewards = []
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

    # plt.clf()
    # fig, ax = plt.subplots()
    # sns.lineplot(data, x = 'episodes_10', y = 'rewards', ax=ax, label=model_name)
    # ax.set_ylim(-2500, 0)
    # ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    # ax.grid()
    # fig.savefig(filename.with_name('scaled_' + filename.name))
    # plt.close(fig)

def load_rewards(path : Path) : 
    """Load rewards from a file."""
    with path.open() as f:
        rewards = np.loadtxt(f)
    rewards = pd.DataFrame(rewards, columns = ['rewards']).reset_index().rename(columns = {'index' : 'episodes'})
    rewards['episodes_10'] = rewards['episodes'] // 10 * 10
    return rewards

def load_and_draw(path : Path, title : str, xlabel : str, ylabel : str, model_name, filename : str) :
    """Load rewards from file and draw plot."""
    rewards = load_rewards(path)
    total_rewards.append([rewards, model_name])
    draw_plot(rewards, title, xlabel, ylabel, model_name,  filename)

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

def main():
    """Load rewards from file and draw plot."""
    #reward_paths = Path('a2c/Results').rglob('Model*/*reward.txt')
    #reward_paths = Path('a2c/Results/Integrated_A2C').rglob('*reward.txt')
    reward_paths = Path('a2c/Results').rglob('*.reward.txt')
    for path in reward_paths:
        load_and_draw(path, 'Rewards by Episode', 'Episode', 'Reward', path.parent.name, path.with_suffix('.png'))
    
    total_reward_df = pd.DataFrame(range(1000), columns = ['episodes'])
    total_reward_df['episodes_10'] = total_reward_df['episodes'] // 10 * 10
    for rewards, model_name in total_rewards:
        total_reward_df[model_name] = rewards['rewards'].values
    draw_total_plot(total_reward_df, 'Rewards by Episode', 'Episode', 'Reward', Path('a2c/Results/total.png'))

def main2() : 
    """
    Draw plot for question 2
    """
    reward_path1 = Path('a2c/Results/Model_128_64_16/pendulum_epi_reward.txt')
    reward_path2 = Path('a2c/Results/Integrated_A2C/pendulum_epi_reward.txt')
    for path in [reward_path1, reward_path2]:
        load_and_draw(path, 'Rewards by Episode', 'Episode', 'Reward', path.parent.name, path.with_suffix('.png'))
    
    total_reward_df = pd.DataFrame(range(1000), columns = ['episodes'])
    total_reward_df['episodes_10'] = total_reward_df['episodes'] // 10 * 10
    for rewards, model_name in total_rewards:
        total_reward_df[model_name] = rewards['rewards'].values
    draw_total_plot(total_reward_df, 'Rewards by Episode', 'Episode', 'Reward', Path('a2c/Results/compare.png'))

if __name__ == "__main__" :
    # main()
    main2()
