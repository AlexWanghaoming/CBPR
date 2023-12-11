import pickle
import seaborn as sns
import os, sys
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
import pandas as pd

# 设置字体和样式
plt.style.use('seaborn-whitegrid')  # 一个清晰、美观的样式
mpl.rcParams['font.family'] = 'Arial'  # 设置字体为 Arial
mpl.rcParams['font.size'] = 14  # 设置基本字体大小为 12 点
mpl.rcParams['axes.labelsize'] = 18  # 设置坐标轴标签的字体大小
mpl.rcParams['axes.titlesize'] = 18  # 设置坐标轴标题的字体大小
mpl.rcParams['xtick.labelsize'] = 16  # 设置x轴刻度标签的字体大小
mpl.rcParams['ytick.labelsize'] = 14  # 设置y轴刻度标签的字体大小
mpl.rcParams['legend.fontsize'] = 13  # 设置图例的字体大小
# mpl.rcParams['image.cmap'] = 'viridis'

def find_files(directory, filename):
    matches = []
    for root, dirnames, filenames in os.walk(directory):
        for file in filenames:
            if file == filename:
                matches.append(os.path.join(root, file))
    return matches


groups = ['cramped_room', 'coordination_ring', 'asymmetric_advantages', 'marshmallow_experiment']
subgroups = ['CBPR', 'BCP', 'FCP', 'SP']
a2c = {
      'CBPR': '#d90429',
       'BCP': '#3f7d20',
       'FCP': '#9d4edd',
       'SP': '#f6ae2d',
       }

all_data = []
for group in groups:
    for subgroup in subgroups:
        print(f"{group}-{subgroup}:")
        num = 0
        # 这里我们使用随机数据作为示例
        for path in find_files(os.path.join('real_human_test', group, subgroup), 'result.pkl'):
            print(path)
            num+=1
            with open(path, 'rb') as f:
                # 加载pickle文件
                data = pickle.load(f)
                print(data['final_reward'])
            all_data.append([group, subgroup, data['final_reward']])
        print(num)
df = pd.DataFrame(all_data, columns=['Group', 'Subgroup', 'Value'])

# 绘制分组的boxplot
plt.figure(figsize=(8, 5))
ax = sns.boxplot(x='Group',
                 y='Value',
                 hue='Subgroup',
                 data=df,
                 palette=a2c,
                 showfliers=False,
                 boxprops=dict(edgecolor='none'))

ax.set_xticklabels(['Cramped\nRoom', 'Coordination\nRing', 'Asymmetric\nAdvantages', 'Marshmallow\nExperiment'])
ax.set_xlabel('')
ax.set_ylabel('Episodic reward')

# 获取每个子组盒子的位置
group_positions = ax.get_xticks()  # 主组的位置
n_subgroups = len(df['Subgroup'].unique())  # 子组的数量
width = 0.8  # 盒子的总宽度
subgroup_width = width / n_subgroups  # 每个子组的宽度

# 添加显著性注释
for i, group in enumerate(df['Group'].unique()):
    group_data = df[df['Group'] == group]
    subgroups = group_data['Subgroup'].unique()
    # 基础高度
    base_y = df['Value'].max() + 10
    # 第一个子组与其他子组比较
    for j in range(1, len(subgroups)):
        data1 = group_data[group_data['Subgroup'] == subgroups[0]]['Value']
        data2 = group_data[group_data['Subgroup'] == subgroups[j]]['Value']
        # t_stat, p_value = stats.ttest_ind(data1, data2)
        t_stat, p_value = stats.mannwhitneyu(data1, data2, alternative='greater')
        # 确定显著性级别
        if p_value < 0.001:
            sig = '***'
        elif p_value < 0.01:
            sig = '**'
        elif p_value < 0.05:
            sig = '*'
        else:
            sig = 'n.s.'
        # 计算注释位置
        x1 = group_positions[i] - width / 2 + subgroup_width / 2
        x2 = group_positions[i] - width / 2 + j * subgroup_width + subgroup_width / 2
        y = base_y + (j - 1) * 40  # 增加垂直位置
        h, col = 10, 'k'   # h是显著性标记中短竖线的高度
        # 画线和添加文本
        plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
        plt.text((x1 + x2) * 0.5, y+h, sig, ha='center', va='bottom', color=col)

# plt.grid(axis='x')
plt.tight_layout()
plt.legend(title='',frameon=False)
plt.savefig(f'human_test_exp3.pdf', bbox_inches='tight')
plt.show()