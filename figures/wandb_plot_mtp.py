import wandb
import glob
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# import matplotlib.font_manager
# fonts = sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist])
# for font in fonts:
#     print(font)
# exit()

# 设置字体和样式
plt.style.use('seaborn-whitegrid')  # 一个清晰、美观的样式
mpl.rcParams['font.family'] = 'Arial'  # 设置字体为 Arial
mpl.rcParams['font.size'] = 14  # 设置基本字体大小为 12 点
mpl.rcParams['axes.labelsize'] = 20  # 设置坐标轴标签的字体大小
mpl.rcParams['axes.titlesize'] = 20  # 设置坐标轴标题的字体大小
mpl.rcParams['xtick.labelsize'] = 14  # 设置x轴刻度标签的字体大小
mpl.rcParams['ytick.labelsize'] = 14  # 设置y轴刻度标签的字体大小
mpl.rcParams['legend.fontsize'] = 16  # 设置图例的字体大小
mpl.rcParams['image.cmap'] = 'viridis'


LAYOUT = 'coordination_ring'
GROUP = 'MTP'
SOURCE_DIR = '/alpha/overcooked_rl'
# WANDB_PATH = SOURCE_DIR + '/algorithms/baselines/wandb'
WANDB_PATH = SOURCE_DIR + '/algorithms/wandb'

p2c = {'place_onion_in_pot': '#0d0887', 'deliver_soup': '#7201a8', 'place_onion_and_deliver_soup': '#ed7953', 'random': '#fdca26'}


runs = glob.glob(f"{WANDB_PATH}/run*")
run_ids = [x.split('-')[-1] for x in runs]
print(runs)
print(run_ids)
api = wandb.Api()
num_episodes = 2000


plt.figure(figsize=(8, 5))
for script_policy_name in p2c:
    reward_list = []
    for run_id in run_ids:
        try:
            run = api.run(f"wanghm/overcooked_rl/{run_id}")
        except wandb.errors.CommError:
            continue
        if run.state == "finished" and run.name.startswith(f'mtp_ppo_{LAYOUT}_{script_policy_name}_seed'):
            print(run.name)
            # num_ep = run.config['num_episodes']
            history = run.history(samples=num_episodes)[['_step', 'ep_reward']]
            ep_sparse_r = history['ep_reward'].to_numpy()
            reward_list.append(ep_sparse_r)

    rewards_array = np.array(reward_list)
    mean_rewards = np.mean(rewards_array, axis=0)
    std_rewards = np.std(rewards_array, axis=0)
    episodes = np.arange(1, num_episodes+1)
    # plt.plot(episodes, mean_rewards, color=p2c[script_policy_name])
    plt.plot(episodes, mean_rewards)
    # plt.fill_between(episodes, mean_rewards-std_rewards, mean_rewards+std_rewards, alpha=0.2, color=p2c[script_policy_name])
    plt.fill_between(episodes, mean_rewards-std_rewards, mean_rewards+std_rewards, alpha=0.2)
    plt.xlabel('Episode')
    plt.ylabel('Mean episode reward')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
# plt.legend()
plt.grid(axis='y')
plt.tight_layout()
# plt.savefig(f'aaa.pdf', bbox_inches='tight')
plt.show()




