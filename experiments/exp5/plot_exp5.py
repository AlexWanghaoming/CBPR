import matplotlib.pyplot as plt
import seaborn as sns

mtx1 = [[9.0, 66.0, 33.0],
        [212.0, 265.0, 266.0],
        [192.0, 269.0, 171.0]]

mtx2 = [[0, 181, 113],
        [151, 172, 206],
        [169.0, 184, 188]]

mtx3 = [[300, 300, 282],
        [260, 260, 250],
        [120, 240, 242]]

mtx4 = [[544.6, 459.8, 501.6],
        [740.0, 524.5, 742.1],
        [774.8, 670.9, 741.0]]


def plot_heat(mtx, nrow, ncol):
    # 使用seaborn绘制热图
    plt.figure(figsize=(ncol, nrow))  # 设置图像大小
    sns.heatmap(mtx,
                annot=True,
                fmt=".1f",
                cmap='coolwarm',
                cbar=True,
                yticklabels=['BCP', 'FCP', 'CBPR'], #设置行名
                xticklabels=['BCP', 'FCP', 'CBPR', 'Mean']
                )
    # # 设置标题和坐标轴标签
    # plt.xlabel('Column')
    # plt.ylabel('Row')
    # 显示图像
    plt.show()

for mtx in [mtx1, mtx2, mtx3, mtx4]:
    plot_heat(mtx,3, 4)