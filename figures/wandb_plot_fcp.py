import wandb
import glob
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage import gaussian_filter1d


WANDB_DIR = '/alpha/overcooked_rl/my_wandb_log'
GROUP = 'FCP'

# 设置字体和样式
plt.style.use('seaborn-whitegrid')  # 一个清晰、美观的样式
mpl.rcParams['font.family'] = 'Arial'  # 设置字体为 Arial
mpl.rcParams['font.size'] = 14  # 设置基本字体大小为 12 点
mpl.rcParams['axes.labelsize'] = 20  # 设置坐标轴标签的字体大小
mpl.rcParams['axes.titlesize'] = 20  # 设置坐标轴标题的字体大小
mpl.rcParams['xtick.labelsize'] = 14  # 设置x轴刻度标签的字体大小
mpl.rcParams['ytick.labelsize'] = 14  # 设置y轴刻度标签的字体大小
mpl.rcParams['legend.fontsize'] = 13  # 设置图例的字体大小
# mpl.rcParams['image.cmap'] = 'viridis'


l2c = {'cramped_room': '#FF0000',
       'asymmetric_advantages': '#0000FF',
       'coordination_ring': '#008000',
       'marshmallow_experiment': '#8B4513'}


runs = glob.glob(f"{WANDB_DIR}/*run*")
run_ids = [x.split('-')[-1] for x in runs]
print(runs)
print(run_ids)
api = wandb.Api()

num_episodes = 50000

plt.figure(figsize=(8, 5))
for layout_name in l2c:
    reward_list = []
    num_runs = 0
    for run_id in run_ids:
        if num_runs>5:
            break
        try:
            run = api.run(f"wanghm/overcooked_rl/{run_id}")
        except wandb.errors.CommError:
            continue
        if run.state == "finished" and run.name.startswith(f'fcp_{layout_name}_seed'):
            print(f"{run_id}:{run.name}")
            num_runs += 1
            num_ep = run.config['num_episodes']
            print(num_ep)
            history = run.history(samples=num_episodes)[['_step', 'ep_reward']]
            ep_sparse_r = history['ep_reward'].tolist()
            reward_list.append(ep_sparse_r)
            # print(len(ep_sparse_r))

    rewards_array = np.array(reward_list)
    mean_rewards = np.mean(rewards_array, axis=0)
    mean_rewards = gaussian_filter1d(mean_rewards, sigma=5)  # 平滑处理
    std_rewards = np.std(rewards_array, axis=0)
    std_rewards = gaussian_filter1d(std_rewards, sigma=5)
    episodes = np.arange(1, num_episodes+1) * 600
    plt.plot(episodes, mean_rewards, color=l2c[layout_name], label=layout_name)
    # plt.plot(episodes, mean_rewards)
    plt.fill_between(episodes, mean_rewards-std_rewards, mean_rewards+std_rewards, alpha=0.2, color=l2c[layout_name])
    # plt.fill_between(episodes, mean_rewards-std_rewards, mean_rewards+std_rewards, alpha=0.2)
    plt.xlabel('Environment steps')
    plt.ylabel('Mean episode reward')
    # plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    # plt.gca().xaxis.set_major_formatter(FuncFormatter(format_ticks))
plt.legend(loc='best')
plt.grid(axis='x')
plt.tight_layout()
plt.savefig(f'fcp_training.pdf', bbox_inches='tight')
plt.show()



