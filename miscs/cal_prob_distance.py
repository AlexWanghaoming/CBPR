import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

# 计算两个概率分布的JS散度
def js_divergence(p, q):
    # 计算M，即P和Q的平均
    m = (p + q) / 2
    # 计算JS散度
    return (entropy(p, m) + entropy(q, m)) / 2

# 计算两个概率分布的总变异距离
def total_variation_distance(p, q):
    return np.sum(np.abs(p - q)) / 2

# 示例概率矩阵
p = np.array([[0.1, 0.2, 0.3, 0.4],
              [0.1, 0.2, 0.3, 0.4],
              [0.1, 0.2, 0.3, 0.4]])

q = np.array([[0.3, 0.3, 0.2, 0.2],
              [0.3, 0.3, 0.2, 0.2],
              [0.3, 0.3, 0.2, 0.2]])

def cal_tvd_js(p:np.ndarray, q:np.ndarray):
    # 将矩阵展平为向量
    p_flat = p.flatten()
    q_flat = q.flatten()
    # 计算JS散度
    js_div = js_divergence(p_flat, q_flat)
    # print(f"JS Divergence: {js_div}")
    # 计算总变异距离
    tvd = total_variation_distance(p_flat, q_flat)
    # print(f"Total Variation Distance: {tvd}")
    # 使用scipy的jensenshannon函数直接计算JS散度
    # js_div_scipy = jensenshannon(p_flat, q_flat) ** 2  # jensenshannon函数返回的是距离的平方根
    # print(f"JS Divergence (using scipy): {js_div_scipy}")
    # print(f"Total Variation Distance: {tvd}, JS Divergence: {js_div}")
    return tvd, js_div

cal_tvd_js(p, q)
