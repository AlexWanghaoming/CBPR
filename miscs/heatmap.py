import matplotlib.pyplot as plt
import seaborn as sns


def plot_heat(mtx, nrow, ncol):
    # 使用seaborn绘制热图
    plt.figure(figsize=(ncol, nrow))  # 设置图像大小
    sns.heatmap(mtx, annot=True, cmap='coolwarm', cbar=True)
    # # 设置标题和坐标轴标签
    # plt.xlabel('Column')
    # plt.ylabel('Row')
    # 显示图像
    plt.show()


def plot_combined_heatmaps(matrix_dict, nrow, ncol, save_path):
    # 设置图像大小
    plt.figure(figsize=(ncol*2, nrow*2))  # 假设每个热图的宽度为ncol，总宽度需要乘以热图的数量

    # 遍历字典并为每个矩阵绘制热图
    for i, (title, mtx) in enumerate(matrix_dict.items(), 1):
        plt.subplot(2, len(matrix_dict)//2, i)  # 创建子图，1行len(matrix_dict)列，当前是第i个
        sns.heatmap(mtx,
                    annot=True,
                    cmap='coolwarm',
                    cbar=False
                    # cbar=i == len(matrix_dict)# 仅在最后一个热图显示颜色条
                    )
        plt.title(title)  # 设置每个热图的标题

    # 显示图像
    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    # plt.savefig(save_path, bbox_inches='tight', format='pdf')
    plt.show()
