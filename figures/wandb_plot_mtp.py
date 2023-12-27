import wandb
import glob
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage import gaussian_filter1d

WANDB_DIR = '/alpha/overcooked_rl/my_wandb_log/mtp'
GROUP = 'MTP'

layout_names = ['Cramped Room',
           'Coordination Ring',
           'Asymmetric Advantages',
           'Soup Coordination']

layouts= ['cramped_room',
           'coordination_ring',
           'asymmetric_advantages',
           'soup_coordination']

p2c = {'place_onion_in_pot': '#33658a',
       'deliver_soup': '#758e4f',
       'place_onion_and_deliver_soup': '#f6ae2d',
       'random': '#f26419',
       'place_tomato_in_pot': '#7209b7',
       'mixed_order': '#8d99ae'}

def get_legend_handles(axs):
    handles = []
    labels = []
    for ax in axs:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    return handles, labels


# 设置字体和样式
plt.style.use('seaborn-whitegrid')  # 一个清晰、美观的样式
mpl.rcParams['font.family'] = 'Arial'  # 设置字体为 Arial
mpl.rcParams['font.size'] = 18  # 设置基本字体大小为 12 点
mpl.rcParams['axes.labelsize'] = 18  # 设置坐标轴标签的字体大小
# mpl.rcParams['axes.titlesize'] = 16  # 设置坐标轴标题的字体大小
mpl.rcParams['xtick.labelsize'] = 24  # 设置x轴刻度标签的字体大小
mpl.rcParams['ytick.labelsize'] = 24  # 设置y轴刻度标签的字体大小
mpl.rcParams['legend.fontsize'] = 24  # 设置图例的字体大小
# mpl.rcParams['image.cmap'] = 'viridis'

api = wandb.Api()
num_episodes = 2000
num_seeds = 5
runs = api.runs(f"wanghm/overcooked_rl")
group_runs = [run for run in runs if run.group == 'MTP']


fig, axs = plt.subplots(1, 4, figsize=(32, 5))
for i, ax in enumerate(axs.flat):
    layout = layouts[i]
    layout_name = layout_names[i]
    for script_policy_name in p2c:
        reward_list = []
        num_runs = 0
        for run in group_runs:
            if num_runs > 5:  #只运行5个seed
                break
            if run.state == "finished" and run.name.startswith(f'mtp_ppo_{layout}_{script_policy_name}_seed'):
                print(f"{run.id}:{run.name}")
                num_runs += 1
                # num_ep = run.config['num_episodes']
                history = run.history(samples=num_episodes)[['_step', 'ep_reward']]
                ep_sparse_r = history['ep_reward'].to_numpy()
                reward_list.append(ep_sparse_r)
        if len(reward_list) == 0:
            continue
        rewards_array = np.array(reward_list)
        mean_rewards = np.mean(rewards_array, axis=0)
        mean_rewards = gaussian_filter1d(mean_rewards, sigma=5)
        std_rewards = np.std(rewards_array, axis=0)
        std_rewards = gaussian_filter1d(std_rewards, sigma=5)
        episodes = np.arange(1, num_episodes+1) * 600
        ax.plot(episodes, mean_rewards, color=p2c[script_policy_name], label=script_policy_name)
        # plt.plot(episodes, mean_rewards)
        ax.fill_between(episodes, mean_rewards-std_rewards, mean_rewards+std_rewards, alpha=0.2, color=p2c[script_policy_name])
        # plt.fill_between(episodes, mean_rewards-std_rewards, mean_rewards+std_rewards, alpha=0.2)
    # ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.grid(axis='x')
    ax.set_xlabel('Environment steps', fontsize=24)
    ax.set_ylabel('Mean episode reward', fontsize=24)
    ax.set_title(layout_name, fontsize=28)

# 调整子图之间的间距
plt.tight_layout()
fig.subplots_adjust(bottom=0.27,
                    wspace=0.22
                    )
handles, labels = get_legend_handles(axs)
fig.legend(handles, labels,
           loc='lower center',
           bbox_to_anchor=(0.5, 0),
           fancybox=True,
           shadow=True,
           ncol=6)
fig.savefig(f'mtp_training.pdf', bbox_inches='tight')
fig.show()




