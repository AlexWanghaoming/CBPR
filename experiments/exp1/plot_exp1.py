import math
from scipy import stats
import wandb
import re
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage import gaussian_filter1d
import argparse
# add = 'http://127.0.0.1:7890'
# os.environ['http_proxy'] = add
# os.environ['https_proxy'] = add

WANDB_DIR = '/alpha/overcooked_rl/my_wandb_log/exp1'
parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--switch_freq', type=str, default='inter1')
args = parser.parse_args()

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


a2c = {
       'okr': '#d90429',
       # 'BCP': '#9d4edd', # 紫色
       'BCP': '#3f7d20',
       # 'FCP': '#219ebc', # 蓝色
       'FCP': '#9d4edd',
       'SP': '#f6ae2d',
       }
layout_names = ['Cramped Room',
           'Coordination Ring',
           'Asymmetric Advantages',
           'Soup Coordination']

layouts= ['cramped_room',
           'coordination_ring',
           'asymmetric_advantages',
           'soup_coordination']

api = wandb.Api()
num_episodes = 50
num_seeds = 5
runs = api.runs(f"wanghm/overcooked_rl")
group_runs = [run for run in runs if run.group == 'exp1']

fig, axs = plt.subplots(1, 4, figsize=(32, 5))
for i, ax in enumerate(axs.flat):
    layout = layouts[i]
    layout_name = layout_names[i]
    for algorithm in a2c:
        reward_list = []
        num_runs = 0
        for run in group_runs:
            if num_runs > num_seeds:  #只运行5个seed
                break
            if run.state == "finished" and \
                    re.match(f'{algorithm}_{layout}_{args.switch_freq}_seed\d+$', run.name):
                print(f"{run.id}:{run.name}")
                num_runs+=1
                num_ep = run.config['num_episodes']
                # print(num_ep)
                history = run.history(samples=num_episodes)[['_step', 'ep_reward']]
                ep_sparse_r = history['ep_reward'].tolist()
                reward_list.append(ep_sparse_r)
        rewards_array = np.array(reward_list)
        mean_rewards = np.mean(rewards_array, axis=0)
        std_rewards = np.std(rewards_array, axis=0)
        ste_rewards = np.std(rewards_array, axis=0)/np.sqrt(num_seeds)
        # 计算置信区间
        sem = stats.sem(reward_list)
        confidence = 0.95
        interval = stats.t.interval(confidence, len(reward_list) - 1, loc=mean_rewards, scale=sem)
        episodes = np.arange(1, num_episodes+1)
        lab = 'CBPR' if algorithm == 'okr' else algorithm
        ax.plot(episodes, mean_rewards, color=a2c[algorithm], label=lab, linewidth=2)
        ax.fill_between(episodes, mean_rewards-std_rewards, mean_rewards+std_rewards, alpha=0.2, color=a2c[algorithm])
    ax.grid(axis='x')
    ax.set_xlabel('Episodes', fontsize=24)
    ax.set_ylabel('Mean episode reward', fontsize=24)
    ax.set_title(layout_name, fontsize=28)
# 调整子图之间的间距
plt.tight_layout()
fig.subplots_adjust(bottom=0.28,
                    wspace=0.2
                    )
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc='lower center',
           bbox_to_anchor=(0.5, 0),
           fancybox=True,
           shadow=True,
           ncol=4)
fig.savefig(f'exp1_{args.switch_freq}_sd.pdf', bbox_inches='tight')
fig.show()




