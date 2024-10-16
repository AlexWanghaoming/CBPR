import wandb
import glob
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage import gaussian_filter1d
from scipy import stats
import argparse


WANDB_DIR = 'my_wandb_log/exp2_2'
parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--layout', default='soup_coordination')
# parser.add_argument('--layout', default='marshmallow_experiment')
args = parser.parse_args()


# 设置字体和样式
plt.style.use('seaborn-whitegrid') 
mpl.rcParams['font.family'] = 'Arial' 
mpl.rcParams['font.size'] = 18  
mpl.rcParams['axes.labelsize'] = 24  
# mpl.rcParams['axes.titlesize'] = 16 
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24  
mpl.rcParams['legend.fontsize'] = 16  # 
# mpl.rcParams['image.cmap'] = 'viridis'


groups = ['low', 'medium', 'high']
subgroups = ['CBPR', 'BCP', 'FCP', 'SP']
a2c = {
      'okr': '#d90429',
       'BCP': '#3f7d20',
       'FCP': '#9d4edd',
       'SP': '#f6ae2d',
       }


api = wandb.Api()
num_episodes = 50
num_seeds = 5
runs = api.runs(f"wanghm/overcooked_rl")
group_runs = [run for run in runs if run.group == 'exp2_2']
plt.figure(figsize=(8, 5))
group_mean = []
group_interval = []

for level in groups:
    sub_group_mean = []
    sub_group_interval = []
    for algorithm in a2c:
        for run in group_runs:
            if algorithm == 'okr':
                match_name = f'okr_{args.layout}_{level}_seed0_Q20_rho0.1'
                if level == 'high':
                    match_name = f'okr_{args.layout}_{level}_seed0_Q20_rho0.9'

            else:
                match_name = f'{algorithm}_{args.layout}_{level}_seed'
            if run.state == "finished" and match_name in run.name:
                print(f"{run.id}:{run.name}")
                num_ep = run.config['num_episodes']
                history = run.history(samples=num_episodes)[['_step', 'ep_reward']]
                ep_sparse_r = history['ep_reward'].tolist()
                mean_r = np.mean(ep_sparse_r)
                # calculate confidence interval
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

fig, ax = plt.subplots()
bar_width = 0.2
n_subgroups = len(subgroups)
index = np.arange(len(groups))
for i in range(n_subgroups):
    ax.bar(index + i * bar_width, group_mean[:, i], bar_width, yerr=group_interval[:, i],
           label=subgroups[i], color=list(a2c.values())[i], capsize=5)

ax.set_ylabel('Mean episode reward')
# ax.set_title('Grouped Bar Chart with Confidence Intervals')
ax.set_xticks(index + bar_width)
ax.set_xticklabels(['Low', 'Medium', 'High'])
# plt.ylim(0, 340)
if args.layout == 'cramped_room':
    plt.ylim(0, 350)
    ax.legend(loc='upper left',
              ncol=2)
plt.grid(axis='x')
plt.tight_layout()
# plt.savefig(f'{args.layout}_exp2_2.pdf', bbox_inches='tight')
plt.show()




