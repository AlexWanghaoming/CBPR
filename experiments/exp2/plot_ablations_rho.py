import wandb
import glob
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage import gaussian_filter1d
from scipy import stats
import argparse
import math

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
# parser.add_argument('--layout', default='soup_coordination')
parser.add_argument('--layout', default='coordination_ring')
args = parser.parse_args()


# 设置字体和样式
plt.style.use('seaborn-whitegrid')  # 一个清晰、美观的样式
mpl.rcParams['font.family'] = 'Arial'  # 设置字体为 Arial
mpl.rcParams['font.size'] = 18  # 设置基本字体大小为 12 点
mpl.rcParams['axes.labelsize'] = 22  # 设置坐标轴名称的字体大小
# mpl.rcParams['axes.titlesize'] = 16  # 设置坐标轴标题的字体大小
mpl.rcParams['xtick.labelsize'] = 22  # 设置x轴刻度标签的字体大小
mpl.rcParams['ytick.labelsize'] = 22  # 设置y轴刻度标签的字体大小
mpl.rcParams['legend.fontsize'] = 16  # 设置图例的字体大小


groups = ['low', 'medium', 'high']
subgroups = ['rho0.9', 'rho0.95', 'rho0.99']

a2c = {
    'rho0.9': '#ffea00',
    'rho0.95': '#ffaa00',
       'rho0.99': '#ff7b00',
       }
api = wandb.Api()
num_episodes = 50
num_seeds = 5
runs = api.runs(f"wanghm/overcooked_rl") # the name of wandb project
group_runs = [run for run in runs if run.group == 'exp2_3']

plt.figure(figsize=(8, 5))
group_mean = []
group_interval = []

for level in groups:
    sub_group_mean = []
    sub_group_interval = []
    for ablation in a2c:
        for run in group_runs:
            match_name = f'okr_{args.layout}_{level}_seed0_Q20_{ablation}'
            if run.state == "finished" and run.name == match_name:
                print(f"{run.id}:{run.name}")
                num_ep = run.config['num_episodes']
                history = run.history(500000)[['_step', 'ep_reward']] # 获取wandb中记录的50000条数据
                ep_sparse_r = history['ep_reward'].tolist()
                ep_sparse_r = [x for x in ep_sparse_r if not math.isnan(x)]
                mean_r = np.mean(ep_sparse_r)
                # 计算置信区间
                sem = stats.sem(ep_sparse_r)
                confidence = 0.95
                interval = stats.t.interval(confidence, len(ep_sparse_r) - 1, loc=mean_r, scale=sem)
                sub_group_mean.append(mean_r)
                sub_group_interval.append(interval[1] - mean_r)
                break
    group_mean.append(sub_group_mean)
    group_interval.append(sub_group_interval)
group_mean = np.array(group_mean,dtype=float)
group_interval = np.array(group_interval,dtype=float)

# 创建图形和轴
fig, ax = plt.subplots()
bar_width = 0.2
n_subgroups = len(subgroups)
index = np.arange(len(groups))
# 生成每个子组的柱状图和误差线
for i in range(n_subgroups):
    ax.bar(index + i * bar_width, group_mean[:, i], bar_width, yerr=group_interval[:, i],
           label=fr'$\rho$={subgroups[i][3:]}', color=list(a2c.values())[i], capsize=5)

# 添加标签和标题
ax.set_ylabel('Mean episode reward')
ax.set_xticks(index + bar_width)
ax.set_xticklabels(['Low', 'Medium', 'High'])
if args.layout == 'cramped_room' or args.layout == 'coordination_ring':
    ax.legend(loc='upper left', ncol=1, columnspacing=0.05)
    # plt.ylim(0, 350)

plt.grid(axis='x')
plt.tight_layout()
plt.ylim(0, 170)
plt.savefig(f'{args.layout}_ablations_rho_rebuttal.pdf', bbox_inches='tight')
plt.show()




